"""SayEye 서버 — 노트북(또는 GPU 서버)에서 돌리고, 보행자 · 보호자는 폰 브라우저로 접속한다.

    python server.py --device 0                 # GPU: weights/ 의 팀 가중치 + 임태규 v12.1 트리거
    python server.py --device cpu               # GPU 없는 노트북

카메라는 보행자 폰 카메라(스마트폰 전용). 폰 카메라는 HTTPS 주소가 필요하다
(Cloudflare 터널 주소를 쓰거나 --https).
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import io
import ipaddress
import hashlib
import hmac
import json
import os
import socket
import sys
import secrets
import time
from pathlib import Path

import cv2
import segno
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR") or APP)   # 확인번호 · 보호자 연결 · 푸시 키 · 알림 기록. Cloud Run 에선 버킷을 붙여 재시작해도 남게
sys.path.insert(0, str(APP))

from alerts import AlertCenter  # noqa: E402
from app_tts import AppTTS  # noqa: E402
from imu_fusion import IMU  # noqa: E402
from push import Push  # noqa: E402
from sensors import SensorLog  # noqa: E402
from pipeline import Config, LivePipeline  # noqa: E402
from sources import SourceManager  # noqa: E402


MOUNT_RATIO = {"eye": 0.93, "neck": 0.82, "chest": 0.72, "waist": 0.60, "hand": 0.70}   # 키 대비 카메라 높이
MOUNT_KO = {"eye": "눈높이", "neck": "목걸이", "chest": "가슴", "waist": "허리", "hand": "손에 들고"}
CHILD_AGE = 13


class Profile:
    """보행자 처음 설정: 만 나이 · 키 · 카메라 위치 → 카메라 높이(트리거) 와 만 13세 미만 보호(보호자가 항상 봄).
    나이는 설정한 날부터 1년마다 저절로 한 살씩 올린다. 13세 미만에서 13세 이상으로 바꾸는 건 보호자 화면에서만."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.d = {}

    def age(self) -> int | None:
        if self.d.get("age") is None:
            return None
        return int(self.d["age"]) + int((time.time() - self.d.get("age_ts", time.time())) // (365.25 * 86400))

    def child(self) -> bool:
        a = self.age()
        return a is not None and a < CHILD_AGE

    def cam_h(self) -> float | None:
        h, m = self.d.get("height_cm"), self.d.get("mount")
        if not h or m not in MOUNT_RATIO:
            return None
        return round(float(h) / 100 * MOUNT_RATIO[m], 2)

    def update(self, m: dict, by_guardian: bool = False) -> str | None:
        """바꾸면 None, 거절하면 이유."""
        new = dict(self.d)
        if m.get("age") is not None:
            try:
                age = int(m["age"])
            except (TypeError, ValueError):
                return "나이를 숫자로 넣어 주세요"
            if not 3 <= age <= 120:
                return "나이를 확인해 주세요"
            if self.child() and age >= CHILD_AGE and not by_guardian:
                return "만 13세 미만 보호는 보호자 화면에서만 풀 수 있어요"
            new.update(age=age, age_ts=time.time())
        if not by_guardian:
            if m.get("height_cm") is not None:
                try:
                    h = float(m["height_cm"])
                except (TypeError, ValueError):
                    return "키를 숫자로 넣어 주세요"
                if not 80 <= h <= 230:
                    return "키를 확인해 주세요"
                new["height_cm"] = round(h)
            if m.get("mount") is not None:
                if m["mount"] not in MOUNT_RATIO:
                    return "카메라 위치를 골라 주세요"
                new["mount"] = m["mount"]
        self.d = new
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, ensure_ascii=False), encoding="utf-8")
        return None

    def public(self) -> dict:
        return {"type": "profile", "age": self.age(), "height_cm": self.d.get("height_cm"), "mount": self.d.get("mount"),
                "mount_ko": MOUNT_KO.get(self.d.get("mount")), "cam_h": self.cam_h(), "child": self.child(),
                "set": self.age() is not None and self.cam_h() is not None}


class Pairing:
    """보호자 연결: 보행자 앱이 1회용 링크(10분)를 만들어 카톡 등으로 보내면, 보호자 기기가 그 링크로 오래 쓰는 토큰을 받는다.
    시각장애인이 번호를 읽어 줄 필요가 없고, 한 번 연결한 보호자 기기는 계속 연결된다. 서버엔 토큰의 해시만 둔다."""
    TTL = 600

    def __init__(self, path: Path):
        self.path, self.codes = path, {}
        try:
            self.tokens = set(json.loads(path.read_text()).get("tokens", []))
        except Exception:
            self.tokens = set()

    @staticmethod
    def _h(t: str) -> str:
        return hashlib.sha256(t.encode()).hexdigest()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"tokens": sorted(self.tokens)}))

    def new_code(self) -> str:
        now = time.time()
        self.codes = {c: e for c, e in self.codes.items() if e > now}
        code = secrets.token_urlsafe(12)
        self.codes[code] = now + self.TTL
        return code

    def redeem(self, code: str) -> str | None:
        exp = self.codes.pop(code, None)          # 한 번 쓰면 사라진다
        if not exp or exp < time.time():
            return None
        tok = secrets.token_urlsafe(24)
        self.tokens.add(self._h(tok))
        self._save()
        return tok

    def ok(self, tok: str) -> bool:
        return bool(tok) and self._h(tok) in self.tokens

    def clear(self):
        self.tokens.clear()
        self.codes.clear()
        self._save()


GUARDIAN_TYPES = {"alert", "status", "loc", "share", "share_denied", "g_msg_ok", "profile"}   # 보호자 화면은 알림 · 상태 · 위치만 받는다 (박스 · 음성 문장은 안 받음)


class Hub:
    """파이프라인 스레드 → 접속한 브라우저로 메시지 전달. 보행자/개발 화면은 전부, 보호자 화면은 알림·상태만."""

    def __init__(self):
        self.clients: dict[asyncio.Queue, str] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    def broadcast(self, msg: dict):
        if self.loop is None or not self.clients:
            return
        s = json.dumps(msg, ensure_ascii=False)
        self.loop.call_soon_threadsafe(self._fanout, s, msg.get("type"))

    def _fanout(self, s: str, typ):
        for q, role in list(self.clients.items()):
            if role == "guardian" and typ not in GUARDIAN_TYPES:
                continue
            if role != "guardian" and typ in ("alert", "loc"):
                continue
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(s)


def lan_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # 실제로 보내지 않음 — 나가는 인터페이스 주소만 얻는다
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def ensure_cert(ips: list[str]) -> tuple[str, str]:
    """자체 서명 인증서 (localhost + 이 노트북의 LAN 주소). 주소가 바뀌면 다시 만든다."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    d = DATA / "certs"
    d.mkdir(parents=True, exist_ok=True)
    crt, key, meta = d / "insight.crt", d / "insight.key", d / "insight.json"
    want = sorted(set(ips) | {"127.0.0.1"})
    if crt.exists() and key.exists() and meta.exists() and json.loads(meta.read_text()) == want:
        return str(crt), str(key)
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "insight-local")])
    san = x509.SubjectAlternativeName([x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in want])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=800)).add_extension(san, critical=False)
            .sign(k, hashes.SHA256()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    meta.write_text(json.dumps(want))
    return str(crt), str(key)


def load_pin(arg: str | None) -> str:
    """보호자 확인번호. 지정하지 않으면 한 번 만들어 certs/guardian_pin.txt 에 두고 계속 쓴다."""
    if arg:
        return str(arg)
    f = DATA / "certs" / "guardian_pin.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    import secrets
    pin = "".join(secrets.choice("0123456789") for _ in range(6))
    f.write_text(pin)
    return pin


def build_app(cfg: Config, source: str | None = None, pin: str = "000000", alerts_dir: Path | None = None,
              session_dir: Path | None = None, key: str | None = None):
    hub = Hub()
    app = FastAPI(title="SayEye")
    state = {"gws": set()}
    pairing = Pairing(DATA / "certs" / "guardians.json")
    profile = Profile(DATA / "certs" / "profile.json")
    tts = AppTTS(DATA / "tts_cache")
    scene_url = (os.environ.get("SCENE_URL") or "").rstrip("/")   # "주변 설명해줘" 풍경 서버 (SJ 말로 듣는 풍경). 비우면 안 씀

    _tok = {"v": None, "exp": 0.0}

    def scene_headers() -> dict:
        """풍경 서버는 공개하지 않고 앱 서버만 부를 수 있게 닫아 둔다 → Cloud Run 에선 서버 계정의 신분 토큰을 붙인다.
        (PC 에서 직접 띄운 풍경 서버는 토큰 없이)"""
        if not os.environ.get("K_SERVICE") or not scene_url:
            return {}
        if time.time() > _tok["exp"]:
            import urllib.parse
            import urllib.request
            u = ("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience="
                 + urllib.parse.quote(scene_url, safe=""))
            with urllib.request.urlopen(urllib.request.Request(u, headers={"Metadata-Flavor": "Google"}), timeout=5) as r:
                _tok["v"], _tok["exp"] = r.read().decode(), time.time() + 3000
        return {"Authorization": "Bearer " + _tok["v"]}

    async def wake_scene(force=False):
        """보행을 시작하면 풍경 서버(VLM)도 같이 깨운다. 걷는 동안 4분마다 다시 불러 잠들지 않게 한다.
        풍경 서버는 큰 모델을 올리느라 깨어나는 데 몇 분 걸리므로, 정작 "주변 설명해줘"라고 할 때 바로 답하게 하려는 것."""
        if not scene_url:
            return
        now = time.time()
        if not force and now - state.get("scene_ping", 0) < 240:
            return
        state["scene_ping"] = now

        def ping():
            import urllib.request
            try:
                req = urllib.request.Request(scene_url + "/api/health", headers=scene_headers())
                with urllib.request.urlopen(req, timeout=600) as r:
                    state["scene_ready"] = json.loads(r.read() or b"{}").get("vlm", True)
            except Exception as e:
                state["scene_ready"] = False
                print(f"[풍경 서버] 깨우기 실패: {e!r}")
        asyncio.get_running_loop().run_in_executor(None, ping)

    def apply_profile():
        IMU.set_height(profile.d.get("height_cm"))       # 보폭 = 키 × 0.415
        h = profile.cam_h()
        trig = getattr(state.get("pipe"), "trigger", None)
        if h and hasattr(trig, "set_cam_h"):
            trig.set_cam_h(h)
        if "alerts" in state:
            state["alerts"].set_always(profile.child())
        hub.broadcast(profile.public())
    sensors = SensorLog(session_dir or (APP / "sessions"))

    def key_ok(q, cookies):
        return not key or hmac.compare_digest(str(q.get("k") or cookies.get("sayeye_k") or ""), key)

    @app.middleware("http")
    async def _key_guard(request: Request, call_next):
        """공개 주소(터널)에서 영상 · 위치가 새지 않게 접속 키를 요구한다. ?k=키 로 한 번 열면 쿠키로 기억."""
        if not key_ok(request.query_params, request.cookies):
            return JSONResponse({"error": "접속 키가 필요해요"}, status_code=401)
        resp = await call_next(request)
        if key and request.query_params.get("k"):
            resp.set_cookie("sayeye_k", key, httponly=True, secure=True, samesite="strict", max_age=86400 * 7)
        return resp

    def check_pin(given: str | None):
        if not guardian_ok(given):
            raise HTTPException(401, "확인번호가 달라요")

    def need_share():
        if not state["alerts"].sharing():
            raise HTTPException(403, "보행자가 공유를 허락해야 볼 수 있어요")

    def guardian_ok(given: str | None) -> bool:
        """보호자 인증: 연결 링크로 받은 토큰, 또는 보조로 남긴 6자리 확인번호."""
        return bool(given) and (hmac.compare_digest(str(given), str(pin)) or pairing.ok(str(given)))

    @app.on_event("startup")
    async def _start():
        hub.loop = asyncio.get_running_loop()
        alerts = AlertCenter(alerts_dir or (DATA / "alerts"), hub.broadcast)
        push = Push(DATA / "certs", alerts.dir)
        alerts.on_new = push.on_alert

        def emit(msg):
            if msg.get("type") == "status":
                msg["sensors"] = sensors.summary()
                alerts.on_status(msg)
            hub.broadcast(msg)
        pipe = LivePipeline(cfg, emit)
        pipe.on_warn = alerts.on_warn
        pipe.on_frame = alerts.clips.push
        src = SourceManager(pipe, hub.broadcast)
        state.update(pipe=pipe, src=src, alerts=alerts, push=push)
        apply_profile()          # 저장된 처음 설정: 카메라 높이 · 만 13세 미만 보호
        state["prewarm"] = asyncio.create_task(tts.prewarm())   # 경고 문장 음성을 미리 (처음 한 번만 1~2분)
        if source:
            src.use_url(source)

    @app.on_event("shutdown")
    async def _stop():
        state["src"].close()
        state["pipe"].stop()

    @app.get("/api/health")
    async def health():
        p, s = state["pipe"], state["src"]
        return {"ok": True, "source": s.mode, "url": s.url, "perception": p.perception.name,
                "trigger": getattr(p.trigger, "name", "?")}

    @app.get("/api/sensors")
    async def api_sensors():
        return sensors.summary()

    @app.get("/preview.mjpg")
    async def preview():
        async def gen():
            last = None
            while True:
                jpg = state["pipe"].preview_jpeg
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.06)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live.mjpg")
    async def live(pin: str | None = None):
        """보호자 실시간 영상 (인식 박스 포함, 약 7 fps). 보는 동안만 서버가 미리보기를 만든다.
        보행자가 공유 중일 때만 열리고(SOS 30분 / 허락 10분), 보는 동안 보행자 앱이 '보호자가 보고 있어요'라고 알린다."""
        check_pin(pin)
        need_share()
        pipe = state["pipe"]

        async def gen():
            pipe.live_viewers += 1
            if pipe.live_viewers == 1:
                hub.broadcast({"type": "viewing", "on": True})
            last = None
            try:
                while True:
                    if not guardian_ok(pin) or not state["alerts"].sharing():   # 연결을 끊거나 공유가 끝나면 끊는다
                        break
                    jpg = pipe.preview_jpeg
                    if jpg is not None and jpg is not last:
                        last = jpg
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    await asyncio.sleep(0.14)
            finally:
                pipe.live_viewers = max(0, pipe.live_viewers - 1)
                if pipe.live_viewers == 0:
                    hub.broadcast({"type": "viewing", "on": False})
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/location")
    async def api_location(pin: str | None = None):
        check_pin(pin)
        al = state["alerts"]
        return {"loc": al.loc if al.sharing() else None, "share": al.share_state()}

    @app.post("/api/share/request")
    async def share_request(pin: str | None = None):
        """보호자가 '위치 · 영상 보기'를 누르면 보행자 폰이 음성으로 허락을 묻는다."""
        check_pin(pin)
        al = state["alerts"]
        if al.sharing():
            return {"ok": True, "share": al.share_state()}
        now = time.time()
        if now - state.get("ask_ts", 0) < 30:
            return {"ok": True, "asked": True}
        state["ask_ts"] = now
        hub.broadcast({"type": "share_req"})
        return {"ok": True, "asked": True}

    @app.get("/api/push/key")
    async def push_key():
        p = state["push"]
        return {"ok": p.ok, "key": p.public if p.ok else None}

    @app.post("/api/push/subscribe")
    async def push_sub(request: Request, pin: str | None = None):
        check_pin(pin)
        j = await request.json()
        try:
            state["push"].subscribe(j.get("sub") or {}, j.get("sens", "caution"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/push/unsubscribe")
    async def push_unsub(request: Request, pin: str | None = None):
        check_pin(pin)
        j = await request.json()
        state["push"].unsubscribe(j.get("endpoint", ""))
        return {"ok": True}

    @app.get("/api/alerts")
    async def api_alerts(pin: str | None = None, since: int = 0):
        check_pin(pin)
        return {"alerts": state["alerts"].list(since), "active": state["alerts"].active}

    @app.get("/api/alerts/{aid}.jpg")
    async def api_alert_snap(aid: int, pin: str | None = None):
        check_pin(pin)
        b = state["alerts"].snap(aid)
        if b is None:
            raise HTTPException(404)
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/alerts/{aid}.mp4")
    async def api_alert_clip(aid: int, pin: str | None = None):
        check_pin(pin)
        p = state["alerts"].clips.path(aid)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/tts")
    async def api_tts(t: str):
        """앱이 말할 문장 → 자연스러운 음성(MP3). 같은 문장은 저장해 두었다가 바로 준다. 실패하면 앱이 폰 기본 음성으로 말한다."""
        p = await tts.get(t)
        if p is None:
            raise HTTPException(503, "음성을 만들 수 없어요")
        return FileResponse(p, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.post("/api/scene")
    async def api_scene(request: Request):
        """"주변 설명해줘" — 보행자 폰이 알아들은 말 + 그 순간 사진을 보내면 풍경 서버(SJ 말로 듣는 풍경)의 /api/text 로 넘긴다.
        같은 보행자는 같은 session 이라 "더 자세히" · "간판에 뭐라고 써 있어?" 가 앞 사진 · 설명에 이어진다."""
        if not scene_url:
            raise HTTPException(503, "풍경 서버가 아직 연결되지 않았어요")
        form = await request.form()
        text = " ".join(str(form.get("text") or "").split())[:200]
        if not text:
            raise HTTPException(400, "말이 비었어요")
        files = {}
        img = form.get("image")
        if img is not None and hasattr(img, "read"):
            files["image"] = ("frame.jpg", await img.read(), "image/jpeg")
        sid = state.setdefault("scene_session", secrets.token_hex(8))
        import httpx
        try:
            async with httpx.AsyncClient(timeout=90) as cl:
                r = await cl.post(scene_url + "/api/text", headers=await asyncio.to_thread(scene_headers),
                                  data={"session": sid, "text": text, "want_image": "false"}, files=files or None)
        except Exception as e:
            asyncio.get_running_loop().create_task(wake_scene(force=True))
            raise HTTPException(503, f"풍경 서버가 준비 중이에요 ({type(e).__name__})")
        if r.status_code != 200:
            asyncio.get_running_loop().create_task(wake_scene(force=True))
            raise HTTPException(503, f"풍경 서버가 준비 중이에요 ({r.status_code})")
        j = r.json()
        return {"answer": j.get("answer") or "", "intent": (j.get("intent") or {}).get("intent"),
                "level": j.get("level"), "timing": j.get("timing")}

    @app.post("/api/pair")
    async def api_pair(code: str):
        """보호자가 연결 링크를 열면 1회용 코드를 오래 쓰는 토큰으로 바꿔 준다."""
        tok = pairing.redeem(code)
        if not tok:
            raise HTTPException(410, "연결 링크가 만료됐거나 이미 쓰였어요")
        hub.broadcast({"type": "paired", "count": len(pairing.tokens)})   # 보행자 앱이 "보호자가 연결됐어요" 라고 말한다
        return {"token": tok}

    @app.get("/api/latest.jpg")
    async def api_latest(pin: str | None = None):
        check_pin(pin)
        need_share()
        im = state["pipe"].last_frame
        if im is None:
            raise HTTPException(404, "아직 영상이 없어요")
        s_ = min(1.0, 640 / max(im.shape[:2]))
        small = cv2.resize(im, (round(im.shape[1] * s_), round(im.shape[0] * s_))) if s_ < 1 else im
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        role = ws.query_params.get("role", "pedestrian")
        await ws.accept()
        if not key_ok(ws.query_params, ws.cookies):
            await ws.send_text(json.dumps({"type": "auth_error"}))
            await ws.close(code=4401)
            return
        if role == "guardian" and not guardian_ok(ws.query_params.get("pin")):
            await ws.send_text(json.dumps({"type": "auth_error"}))
            await ws.close(code=4401)
            return
        if role == "guardian":
            state["gws"].add(ws)
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        hub.clients[q] = role
        p, s = state["pipe"], state["src"]
        await ws.send_text(json.dumps({"type": "hello", "source": s.mode, "url": s.url, "default_url": source,
                                       "perception": p.perception.name, "trigger": getattr(p.trigger, "name", "?"),
                                       "weights": Path(str(p.cfg.weights)).name,
                                       # 보호자 확인번호는 보행자 화면에만 보낸다 (앱 열 때마다 화면에 띄움)
                                       **({"pin": str(pin), "paired": len(pairing.tokens)} if role == "pedestrian" else {}),
                                       "share": state["alerts"].share_state(), "profile": profile.public()},
                                      ensure_ascii=False))

        async def sender():
            while True:
                await ws.send_text(await q.get())

        task = asyncio.create_task(sender())
        try:
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes") is not None:
                    if role != "guardian":
                        await wake_scene()                           # 걷는 동안 4분마다 (잠들지 않게)
                        got = s.on_phone_frame(m["bytes"])
                        sensors.frame("phone", got[0] if got else None, got[1] if got else len(m["bytes"]))
                elif m.get("text"):
                    try:
                        cmd = json.loads(m["text"])
                    except json.JSONDecodeError:
                        continue
                    if cmd.get("type") == "sensors" and role != "guardian":
                        sensors.add(cmd)
                        IMU.add(cmd.get("motion"))           # 자이로 · 가속도 → 프레임 시각에 맞춰 트리거에
                    elif cmd.get("type") == "set_source":
                        if cmd.get("mode") == "phone":
                            await asyncio.to_thread(s.use_phone)
                            await wake_scene(force=True)             # 보행 시작 → 풍경 서버(VLM)도 같이 깨우기
                        elif cmd.get("mode") == "url" and cmd.get("url"):
                            await asyncio.to_thread(s.use_url, cmd["url"])
                    elif cmd.get("type") == "sos" and role != "guardian":
                        state["alerts"].on_sos(p.last_frame)
                    elif cmd.get("type") == "client_event" and role != "guardian":
                        state["alerts"].on_client_event(cmd, p.last_frame)
                    elif cmd.get("type") == "profile" and role in ("pedestrian", "guardian"):
                        why = profile.update(cmd, by_guardian=(role == "guardian"))
                        if why:
                            await ws.send_text(json.dumps({"type": "profile_denied", "why": why}, ensure_ascii=False))
                        else:
                            apply_profile()
                            if role == "guardian":
                                await ws.send_text(json.dumps(profile.public(), ensure_ascii=False))
                    elif cmd.get("type") == "share_allow" and role == "pedestrian":
                        from alerts import SHARE_ASK_S
                        state["alerts"].share(SHARE_ASK_S, "allowed")
                    elif cmd.get("type") == "share_deny" and role == "pedestrian":
                        hub.broadcast({"type": "share_denied"})
                    elif cmd.get("type") == "share_stop" and role == "pedestrian":
                        state["alerts"].share_stop()
                    elif cmd.get("type") == "g_msg" and role == "guardian":
                        text = " ".join(str(cmd.get("text") or "").split())[:120]   # 보호자 → 보행자 음성 메시지
                        now = time.time()
                        if text and now - state.get("gmsg_ts", 0) >= 3:
                            state["gmsg_ts"] = now
                            hub.broadcast({"type": "g_msg", "text": text})
                            await ws.send_text(json.dumps({"type": "g_msg_ok"}))
                    elif cmd.get("type") == "pair_new" and role == "pedestrian":
                        # 보행자 화면에 QR 로 띄워 보호자가 자기 폰 카메라로 찍게 한다 (멀리 있으면 같은 링크를 공유)
                        proto = ws.headers.get("x-forwarded-proto") or ("https" if ws.url.scheme == "wss" else "http")
                        url = f"{proto}://{ws.headers.get('host')}/guardian?pair={pairing.new_code()}" + (f"&k={key}" if key else "")
                        buf = io.BytesIO()
                        segno.make(url, error="m").save(buf, kind="svg", scale=8, border=3, dark="#000", light="#fff")
                        await ws.send_text(json.dumps({"type": "pair_link", "url": url, "ttl": Pairing.TTL,
                                                       "qr": buf.getvalue().decode()}))
                    elif cmd.get("type") == "unpair_all" and role == "pedestrian":
                        pairing.clear()
                        for g in list(state["gws"]):
                            try:
                                await g.send_text(json.dumps({"type": "auth_error", "reason": "unpaired"}))
                                await g.close(code=4401)
                            except Exception:
                                pass
                        state["gws"].clear()
                        await ws.send_text(json.dumps({"type": "unpaired"}))
                    elif cmd.get("type") == "gps" and role == "pedestrian":
                        state["alerts"].on_location(cmd)
        except Exception:
            pass
        finally:
            task.cancel()
            hub.clients.pop(q, None)
            state["gws"].discard(ws)

    @app.get("/")
    async def index():
        return FileResponse(APP / "web" / "index.html")

    @app.get("/guardian")
    async def guardian():
        return FileResponse(APP / "web" / "guardian.html")

    @app.get("/dev")
    async def dev():
        return FileResponse(APP / "web" / "dev.html")

    app.mount("/", StaticFiles(directory=APP / "web"), name="web")
    return app


def main():
    ap = argparse.ArgumentParser(description="SayEye 서버")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ["PORT"]) if os.environ.get("PORT") else None, help="기본: --https면 8443, 아니면 8080")
    ap.add_argument("--https", action="store_true", help="자체 서명 HTTPS (터널 없이 같은 와이파이에서 폰 카메라를 쓸 때)")
    ap.add_argument("--weights", default="auto", help="auto(weights/ 폴더에서 찾기) 또는 .pt 경로")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cpu"), help="cpu 또는 GPU 번호(0)")
    ap.add_argument("--trigger", default="nav", help="nav(임태규 v12.1, 기본) | b0")
    ap.add_argument("--conf", type=float, default=None, help="YOLO 신뢰도 (b0 만. nav 는 자체 기준)")
    ap.add_argument("--imgsz", type=int, default=None, choices=[640, 960, 1280], help="nav YOLO 입력 크기 (기본 CPU 640 · GPU 960)")
    ap.add_argument("--no-crop", action="store_true", help="nav 의 잘라 보기(추가 추론)를 끈다 — 느린 CPU 에서")
    ap.add_argument("--no-face-blur", action="store_true", help="보호자 영상 · 저장 사진의 얼굴 가리기를 끈다")
    ap.add_argument("--cam-h", type=float, default=None, help="(b0 · 예전 hitbox 용, nav 는 쓰지 않음)")
    ap.add_argument("--vlm", default="none", help=argparse.SUPPRESS)          # 예전 명령 호환용 (경고 VLM 기능은 뺐다)
    ap.add_argument("--max-fps", type=float, default=15.0)
    ap.add_argument("--guardian-pin", default=os.environ.get("GUARDIAN_PIN") or None, help="보호자 확인번호 (기본: 처음 한 번 만들어 certs/guardian_pin.txt 에 저장)")
    ap.add_argument("--session-dir", default=None, help="센서 기록 폴더 (기본 code/app/sessions)")
    ap.add_argument("--key", default=os.environ.get("SAYEYE_KEY") or None, help="접속 키. 공개 터널로 열 때 꼭 지정 (https://주소/?k=키)")
    a = ap.parse_args()
    if a.cam_h:
        os.environ["CAM_H"] = str(a.cam_h)

    from pipeline import find_weights
    w = find_weights(a.weights)
    if w is None:
        print("\n[오류] 팀 가중치(.pt)를 찾지 못했어요.\n"
              f"  {APP / 'weights'} 폴더에 final_10class_best.pt 를 넣고 다시 실행하세요.\n"
              "  (또는 --weights 경로/파일.pt)\n")
        sys.exit(1)
    if a.no_face_blur:
        os.environ["FACE_BLUR"] = "0"
    cfg = Config(perception="yolo", weights=str(w), device=a.device, conf=a.conf, trigger=a.trigger,
                 nav_imgsz=a.imgsz, nav_crop=not a.no_crop, max_fps=a.max_fps)
    port = a.port or (8443 if a.https else 8080)
    ips = lan_ips()
    kw = {}
    scheme = "http"
    if a.https:
        crt, key = ensure_cert(ips)
        kw = dict(ssl_certfile=crt, ssl_keyfile=key)
        scheme = "https"
    urls = [f"{scheme}://{ip}:{port}" for ip in ips] or [f"{scheme}://localhost:{port}"]
    pin = load_pin(a.guardian_pin)
    print("\n보행자 화면 (폰 브라우저):")
    for u in urls:
        print("  ", u)
    print(f"보호자 화면: {urls[0]}/guardian   확인번호: {pin}")
    print(f"개발자 화면: {urls[0]}/dev")
    if a.key:
        print(f"접속 키가 켜져 있어요. 주소 뒤에 ?k={a.key} 를 붙여 여세요.")
    if not a.https:
        print("   폰 카메라 · 푸시 알림은 HTTPS 가 필요해요 → cloudflared tunnel --url " + f"http://localhost:{port}")
    print(f"\n가중치={w.name} · 트리거={a.trigger} · 장치={a.device}\n")
    app = build_app(cfg, None, pin, session_dir=Path(a.session_dir) if a.session_dir else None, key=a.key)
    uvicorn.run(app, host=a.host, port=port, log_level="warning", **kw)


if __name__ == "__main__":
    main()

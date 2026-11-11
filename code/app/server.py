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
import re
import socket
import sys
import secrets
import threading
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
from bus import Bus  # noqa: E402
from nav import Nav  # noqa: E402
from imu_fusion import ImuBuffer  # noqa: E402
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


UID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
MAX_ACTIVE = int(os.environ.get("MAX_USERS", "8"))      # 동시에 판단 장치를 돌리는 보행자 수 상한 (GPU 하나 기준)
IDLE_S = 600                                            # 보행자가 10분 동안 연결이 없으면 그 사람의 판단 장치를 내린다


def _pin_file(path: Path, given: str | None = None) -> str:
    if given:
        return str(given)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text().strip():
        return path.read_text().strip()
    pin = "".join(secrets.choice("0123456789") for _ in range(6))
    path.write_text(pin)
    return pin


class User:
    """보행자 한 명. 판단 장치(YOLO + 트리거) · 센서 · 처음 설정 · 보호자 연결 · 알림 · 푸시 · 공유 · 메시지 통로가 모두 따로다.
    'default' 는 예전 단일 사용자 자리(이전 데이터 · 테스트 호환) — DATA 바로 아래를 그대로 쓴다."""

    def __init__(self, uid: str, cfg: Config, loop, pin: str | None = None,
                 alerts_dir: Path | None = None, session_dir: Path | None = None):
        self.uid, self.cfg = uid, cfg
        self.root = DATA if uid == "default" else DATA / "users" / uid
        self.hub = Hub()
        self.hub.loop = loop
        self.gws: set = set()
        self.state: dict = {}
        self.pairing = Pairing(self.root / "certs" / "guardians.json")
        self.profile = Profile(self.root / "certs" / "profile.json")
        self.pin = _pin_file(self.root / "certs" / "guardian_pin.txt", pin)
        self.imu = ImuBuffer()
        self.sensors = SensorLog(session_dir or (self.root / "sessions"))
        self.alerts = AlertCenter(alerts_dir or (self.root / "alerts"), self.hub.broadcast)
        self.push = Push(DATA / "certs", self.alerts.dir)          # 푸시 키는 서버 하나, 구독은 사람마다
        self.alerts.on_new = self.push.on_alert
        self.alerts.set_always(self.profile.child())      # 서버가 다시 켜져도 13세 미만 보호는 바로 (보행자 연결 전에도)
        self.imu.set_height(self.profile.d.get("height_cm"))
        self.pipe = self.src = None
        self.ped = 0                        # 지금 연결된 보행자 화면 수
        self.last_ped = time.time()
        self.lock = threading.Lock()

    # 판단 장치 (무거움: 모델을 올림) — 보행자가 연결하면 켜고, 오래 비면 내린다
    def start_pipe(self):
        with self.lock:
            if self.pipe is not None:
                return

            def emit(msg):
                if msg.get("type") == "status":
                    msg["sensors"] = self.sensors.summary()
                    self.alerts.on_status(msg)
                self.hub.broadcast(msg)
            pipe = LivePipeline(self.cfg, emit)
            pipe.on_warn = self.alerts.on_warn
            pipe.on_frame = self.alerts.clips.push
            if hasattr(pipe.perception, "imu"):
                pipe.perception.imu = self.imu          # 이 사람 폰의 센서만 이 사람 판단에
            if hasattr(pipe.perception, "indoor"):
                pipe.perception.indoor = bool(self.state.get("indoor"))   # 실내 모드 (기본 꺼짐)
            self.src = SourceManager(pipe, self.hub.broadcast)
            self.pipe = pipe
        self.apply_profile()

    def stop_pipe(self):
        with self.lock:
            if self.src is not None:
                self.src.close()
            if self.pipe is not None:
                self.pipe.stop()
            self.pipe = self.src = None

    def apply_profile(self):
        self.imu.set_height(self.profile.d.get("height_cm"))       # 보폭 = 키 × 0.415
        h = self.profile.cam_h()
        trig = getattr(self.pipe, "trigger", None)
        if h and hasattr(trig, "set_cam_h"):
            trig.set_cam_h(h)
        self.alerts.set_always(self.profile.child())
        self.hub.broadcast(self.profile.public())

    def guardian_ok(self, given: str | None) -> bool:
        """보호자 인증: 연결 링크(QR)로 받은 토큰, 또는 보조 6자리 확인번호."""
        return bool(given) and (hmac.compare_digest(str(given), str(self.pin)) or self.pairing.ok(str(given)))


def build_app(cfg: Config, source: str | None = None, pin: str = "000000", alerts_dir: Path | None = None,
              session_dir: Path | None = None, key: str | None = None):
    app = FastAPI(title="SayEye")
    users: dict[str, User] = {}
    shared = {}
    tts = AppTTS(DATA / "tts_cache")
    bus = Bus()                                   # 버스 도착 (서울 · TAGO), 키는 DATA_GO_KR_KEY
    nav = Nav()                                   # 도보 길 안내 (TMAP_KEY, 없으면 OpenStreetMap)
    scene_url = (os.environ.get("SCENE_URL") or "").rstrip("/")   # "주변 설명해줘" 풍경 서버 (SJ 말로 듣는 풍경). 비우면 안 씀

    def get_user(uid: str | None, create: bool = True) -> User | None:
        uid = uid if uid and UID_RE.match(uid) else "default"
        u = users.get(uid)
        if u is None and create:
            loop = shared.get("loop")
            u = users[uid] = User(uid, cfg, loop, pin if uid == "default" else None,
                                  alerts_dir if uid == "default" else None, session_dir if uid == "default" else None)
        return u

    def find_guardian(given: str | None) -> User | None:
        """보호자 토큰 · 확인번호 → 그 보행자."""
        if not given:
            return None
        for u in list(users.values()):
            if u.guardian_ok(given):
                return u
        return None

    def guardian_user(given: str | None) -> User:
        u = find_guardian(given)
        if u is None:
            raise HTTPException(401, "확인번호가 달라요")
        return u

    def need_share(u: User):
        if not u.alerts.sharing():
            raise HTTPException(403, "보행자가 공유를 허락해야 볼 수 있어요")

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
        """보행을 시작하면 풍경 서버(VLM)도 같이 깨운다. 누군가 걷는 동안 4분마다 다시 불러 잠들지 않게 한다 (풍경 서버는 모두가 같이 씀)."""
        if not scene_url:
            return
        now = time.time()
        if not force and now - shared.get("scene_ping", 0) < 240:
            return
        shared["scene_ping"] = now

        def ping():
            import urllib.request
            try:
                req = urllib.request.Request(scene_url + "/api/health", headers=scene_headers())
                with urllib.request.urlopen(req, timeout=600) as r:
                    shared["scene_ready"] = json.loads(r.read() or b"{}").get("vlm", True)
            except Exception as e:
                shared["scene_ready"] = False
                print(f"[풍경 서버] 깨우기 실패: {e!r}")
        asyncio.get_running_loop().run_in_executor(None, ping)

    def key_ok(q, cookies):
        return not key or hmac.compare_digest(str(q.get("k") or cookies.get("sayeye_k") or ""), key)

    @app.middleware("http")
    async def _key_guard(request: Request, call_next):
        """공개 주소에서 영상 · 위치가 새지 않게 접속 키를 요구한다. ?k=키 로 한 번 열면 쿠키로 기억."""
        if not key_ok(request.query_params, request.cookies):
            return JSONResponse({"error": "접속 키가 필요해요"}, status_code=401)
        resp = await call_next(request)
        path = request.url.path
        if path in ("/", "/guardian", "/dev") or path.endswith((".html", ".js", ".css", ".webmanifest")):
            # 폰 브라우저가 옛 화면을 그대로 쓰지 않게 매번 새 버전인지 확인 (옛 화면은 사용자 번호를 안 보내 모두가 한 사람으로 묶였다)
            resp.headers["Cache-Control"] = "no-cache"
        if key and request.query_params.get("k"):
            resp.set_cookie("sayeye_k", key, httponly=True, secure=True, samesite="strict", max_age=86400 * 7)
        return resp

    def boot(user: User):
        """판단 장치를 뒤에서 켜고, 다 되면 그 보행자 화면에 'ready' 를 보낸다 (그때부터 영상을 받는다)."""
        if user.pipe is not None or user.state.get("booting"):
            if user.pipe is not None:
                user.hub.broadcast({"type": "ready"})
            return
        user.state["booting"] = True

        async def run():
            try:
                await asyncio.to_thread(user.start_pipe)
                p = user.pipe
                if user.state.pop("want_source", None) == "phone":
                    await asyncio.to_thread(user.src.use_phone)
                user.hub.broadcast({"type": "ready", "perception": p.perception.name, "trigger": getattr(p.trigger, "name", "?")})
            except Exception as e:
                print(f"[boot] {user.uid[:8]} 실패: {e!r}")
            finally:
                user.state["booting"] = False
        asyncio.get_running_loop().create_task(run())

    def warm_gpu():
        """서버가 켜질 때 한 번: GPU(CUDA) 준비와 모델 첫 추론을 미리 해 둔다 → 첫 보행자의 판단 장치가 빨리 켜진다."""
        try:
            if str(cfg.device) == "cpu":
                return
            import numpy as np
            from ultralytics import YOLO
            m = YOLO(str(cfg.weights))
            for _ in range(2):
                m.predict(np.zeros((960, 540, 3), np.uint8), imgsz=960, device=cfg.device, verbose=False)
            print("[warm] GPU 준비 끝", flush=True)
        except Exception as e:
            print(f"[warm] 건너뜀: {e!r}", flush=True)

    async def _janitor():
        """10분 동안 보행자 연결이 없는 사람의 판단 장치를 내린다 (GPU 메모리 · 계산을 비움). 기록 · 보호자 연결은 그대로."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            for u in list(users.values()):
                if u.pipe is not None and u.ped == 0 and now - u.last_ped > IDLE_S:
                    await asyncio.to_thread(u.stop_pipe)

    @app.on_event("startup")
    async def _start():
        shared["loop"] = asyncio.get_running_loop()
        get_user("default")
        root = DATA / "users"
        if root.is_dir():                    # 이전에 쓰던 사람들 (보호자 토큰 · 확인번호로 찾을 수 있게, 가벼운 부분만)
            for d in root.iterdir():
                if d.is_dir() and UID_RE.match(d.name):
                    get_user(d.name)
        shared["prewarm"] = asyncio.create_task(tts.prewarm())   # 경고 문장 음성을 미리 (처음 한 번만 1~2분)
        shared["janitor"] = asyncio.create_task(_janitor())
        shared["warm"] = asyncio.get_running_loop().run_in_executor(None, warm_gpu)
        if source:
            u = get_user("default")
            await asyncio.to_thread(u.start_pipe)
            u.src.use_url(source)

    @app.on_event("shutdown")
    async def _stop():
        for u in list(users.values()):
            u.stop_pipe()

    @app.get("/api/health")
    async def health():
        active = [u for u in users.values() if u.pipe is not None]
        p = active[0].pipe if active else None
        return {"ok": True, "users": len(users), "active": len(active),
                "perception": p.perception.name if p else None, "trigger": getattr(p.trigger, "name", None) if p else None}

    @app.get("/api/sensors")
    async def api_sensors(u: str | None = None):
        return get_user(u).sensors.summary()

    @app.get("/preview.mjpg")
    async def preview(u: str | None = None):
        user = get_user(u)

        async def gen():
            last = None
            while True:
                jpg = user.pipe.preview_jpeg if user.pipe else None
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.06)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live.mjpg")
    async def live(pin: str | None = None):
        """보호자 실시간 영상 (인식 박스 포함, 약 7 fps). 그 보행자가 공유 중일 때만 (SOS 30분 / 허락 10분 / 13세 미만),
        보는 동안 보행자 앱이 '보호자가 보고 있어요'라고 알린다."""
        user = guardian_user(pin)
        need_share(user)
        pipe = user.pipe
        if pipe is None:
            raise HTTPException(404, "보행자가 지금 안내를 쓰고 있지 않아요")

        async def gen():
            pipe.live_viewers += 1
            if pipe.live_viewers == 1:
                user.hub.broadcast({"type": "viewing", "on": True})
            last = None
            try:
                while True:
                    if not user.guardian_ok(pin) or not user.alerts.sharing():   # 연결을 끊거나 공유가 끝나면 끊는다
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
                    user.hub.broadcast({"type": "viewing", "on": False})
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/location")
    async def api_location(pin: str | None = None):
        al = guardian_user(pin).alerts
        return {"loc": al.loc if al.sharing() else None, "share": al.share_state()}

    @app.post("/api/share/request")
    async def share_request(pin: str | None = None):
        """보호자가 '위치 · 영상 보기'를 누르면 보행자 폰이 음성으로 허락을 묻는다."""
        user = guardian_user(pin)
        al = user.alerts
        if al.sharing():
            return {"ok": True, "share": al.share_state()}
        now = time.time()
        if now - user.state.get("ask_ts", 0) < 30:
            return {"ok": True, "asked": True}
        user.state["ask_ts"] = now
        user.hub.broadcast({"type": "share_req"})
        return {"ok": True, "asked": True}

    @app.get("/api/push/key")
    async def push_key():
        p = get_user("default").push
        return {"ok": p.ok, "key": p.public if p.ok else None}

    @app.post("/api/push/subscribe")
    async def push_sub(request: Request, pin: str | None = None):
        user = guardian_user(pin)
        j = await request.json()
        try:
            user.push.subscribe(j.get("sub") or {}, j.get("sens", "caution"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/push/unsubscribe")
    async def push_unsub(request: Request, pin: str | None = None):
        user = guardian_user(pin)
        j = await request.json()
        user.push.unsubscribe(j.get("endpoint", ""))
        return {"ok": True}

    @app.get("/api/alerts")
    async def api_alerts(pin: str | None = None, since: int = 0):
        al = guardian_user(pin).alerts
        return {"alerts": al.list(since), "active": al.active}

    @app.get("/api/alerts/{aid}.jpg")
    async def api_alert_snap(aid: int, pin: str | None = None):
        b = guardian_user(pin).alerts.snap(aid)
        if b is None:
            raise HTTPException(404)
        return Response(b, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/alerts/{aid}.mp4")
    async def api_alert_clip(aid: int, pin: str | None = None):
        p = guardian_user(pin).alerts.clips.path(aid)
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/tts")
    async def api_tts(t: str):
        """앱이 말할 문장 → 자연스러운 음성(MP3). 모두가 같이 쓰는 저장본. 실패하면 앱이 폰 기본 음성으로 말한다."""
        p = await tts.get(t)
        if p is None:
            raise HTTPException(503, "음성을 만들 수 없어요")
        return FileResponse(p, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.post("/api/scene")
    async def api_scene(request: Request):
        """"주변 설명해줘" — 보행자 폰이 알아들은 말 + 그 순간 사진을 보내면 풍경 서버(SJ 말로 듣는 풍경)의 /api/text 로 넘긴다.
        보행자마다 session 이 달라서 "더 자세히" · "간판에 뭐라고 써 있어?" 가 그 사람의 앞 사진 · 설명에 이어진다."""
        if not scene_url:
            raise HTTPException(503, "풍경 서버가 아직 연결되지 않았어요")
        form = await request.form()
        user = get_user(str(form.get("u") or request.query_params.get("u") or ""))
        text = " ".join(str(form.get("text") or "").split())[:200]
        if not text:
            raise HTTPException(400, "말이 비었어요")
        files = {}
        img = form.get("image")
        if img is not None and hasattr(img, "read"):
            files["image"] = ("frame.jpg", await img.read(), "image/jpeg")
        sid = user.state.setdefault("scene_session", hashlib.sha256((user.uid + secrets.token_hex(4)).encode()).hexdigest()[:16])
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

    @app.get("/api/nav")
    async def api_nav(q: str, lat: float, lon: float):
        """"인하대역 가자" — 장소를 찾아 걷는 길(갈림길 · 횡단보도 안내)을 돌려준다."""
        try:
            return await asyncio.to_thread(nav.plan, q[:60], lat, lon)
        except LookupError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(502, f"길을 찾지 못했어요 ({type(e).__name__})")

    @app.get("/api/nav/route")
    async def api_nav_route(lat: float, lon: float, dlat: float, dlon: float, name: str = ""):
        """길에서 벗어났을 때 지금 자리에서 다시 찾기."""
        try:
            return await asyncio.to_thread(nav.route, lat, lon, dlat, dlon, name[:60])
        except Exception as e:
            raise HTTPException(502, f"길을 다시 찾지 못했어요 ({type(e).__name__})")

    @app.get("/api/bus")
    async def api_bus(lat: float, lon: float, route: str | None = None):
        """"7016번 버스 언제 와" · "버스 언제 와" — 가장 가까운 정류장의 도착 예정 (서울은 서울시 API, 인천 등은 TAGO)."""
        try:
            return await asyncio.to_thread(bus.near, lat, lon, route)
        except PermissionError as e:
            raise HTTPException(503, str(e))
        except LookupError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(502, f"버스 정보를 받지 못했어요 ({type(e).__name__})")

    @app.post("/api/pair")
    async def api_pair(code: str):
        """보호자가 연결 링크(QR)를 열면 1회용 코드를 그 보행자의 토큰으로 바꿔 준다."""
        for user in list(users.values()):
            if code in user.pairing.codes:
                tok = user.pairing.redeem(code)
                if tok:
                    user.hub.broadcast({"type": "paired", "count": len(user.pairing.tokens)})   # "보호자가 연결됐어요"
                    return {"token": tok}
        raise HTTPException(410, "연결 링크가 만료됐거나 이미 쓰였어요")

    @app.get("/api/latest.jpg")
    async def api_latest(pin: str | None = None):
        user = guardian_user(pin)
        need_share(user)
        im = user.pipe.last_frame if user.pipe else None
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
        if role == "guardian":
            user = find_guardian(ws.query_params.get("pin"))
            if user is None:
                await ws.send_text(json.dumps({"type": "auth_error"}))
                await ws.close(code=4401)
                return
            user.gws.add(ws)
        else:
            user = get_user(ws.query_params.get("u"))
            if user.pipe is None and sum(1 for x in users.values() if x.pipe is not None) >= MAX_ACTIVE:
                await ws.send_text(json.dumps({"type": "server_busy"}))
                await ws.close(code=4429)
                return
            boot(user)                  # 판단 장치는 뒤에서 준비 (모델 올리기에 수 초~수십 초) — 그동안에도 연결 · 인사는 바로
            user.ped += 1
            user.last_ped = time.time()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        user.hub.clients[q] = role
        p, s = user.pipe, user.src
        await ws.send_text(json.dumps({"type": "hello", "source": s.mode if s else None, "url": s.url if s else None,
                                       "default_url": source,
                                       "perception": p.perception.name if p else None,
                                       "trigger": getattr(p.trigger, "name", "?") if p else None,
                                       "weights": Path(str(cfg.weights)).name,
                                       # 보호자 확인번호는 보행자 화면에만 보낸다
                                       **({"pin": str(user.pin), "paired": len(user.pairing.tokens)} if role == "pedestrian" else {}),
                                       "share": user.alerts.share_state(), "profile": user.profile.public()},
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
                if role != "guardian":
                    user.last_ped = time.time()
                    p, s = user.pipe, user.src
                    if p is None:                    # 아직 준비 중이거나, 오래 비어서 내려간 뒤 다시 쓰기 시작
                        boot(user)
                if m.get("bytes") is not None:
                    if role != "guardian" and s is not None:
                        await wake_scene()                           # 걷는 동안 4분마다 (잠들지 않게)
                        got = s.on_phone_frame(m["bytes"])
                        user.sensors.frame("phone", got[0] if got else None, got[1] if got else len(m["bytes"]))
                elif m.get("text"):
                    try:
                        cmd = json.loads(m["text"])
                    except json.JSONDecodeError:
                        continue
                    t = cmd.get("type")
                    if t == "sensors" and role != "guardian":
                        user.sensors.add(cmd)
                        user.imu.add(cmd.get("motion"))      # 자이로 · 가속도 → 프레임 시각에 맞춰 이 사람의 트리거에
                    elif t == "set_source" and role != "guardian" and s is None:
                        user.state["want_source"] = cmd.get("mode")   # 준비가 끝나면 적용
                        if cmd.get("mode") == "phone":
                            await wake_scene(force=True)
                    elif t == "set_source" and role != "guardian":
                        if cmd.get("mode") == "phone":
                            await asyncio.to_thread(s.use_phone)
                            await wake_scene(force=True)             # 보행 시작 → 풍경 서버(VLM)도 같이 깨우기
                        elif cmd.get("mode") == "url" and cmd.get("url"):
                            await asyncio.to_thread(s.use_url, cmd["url"])
                    elif t == "sos" and role != "guardian":
                        user.alerts.on_sos(p.last_frame if p else None)
                    elif t == "client_event" and role != "guardian":
                        user.alerts.on_client_event(cmd, p.last_frame if p else None)
                    elif t == "profile" and role in ("pedestrian", "guardian"):
                        why = user.profile.update(cmd, by_guardian=(role == "guardian"))
                        if why:
                            await ws.send_text(json.dumps({"type": "profile_denied", "why": why}, ensure_ascii=False))
                        else:
                            user.apply_profile()
                            if role == "guardian":
                                await ws.send_text(json.dumps(user.profile.public(), ensure_ascii=False))
                    elif t == "share_allow" and role == "pedestrian":
                        from alerts import SHARE_ASK_S
                        user.alerts.share(SHARE_ASK_S, "allowed")
                    elif t == "share_deny" and role == "pedestrian":
                        user.hub.broadcast({"type": "share_denied"})
                    elif t == "share_stop" and role == "pedestrian":
                        user.alerts.share_stop()
                    elif t == "g_msg" and role == "guardian":
                        text = " ".join(str(cmd.get("text") or "").split())[:120]   # 보호자 → 그 보행자에게만 음성 메시지
                        now = time.time()
                        if text and now - user.state.get("gmsg_ts", 0) >= 3:
                            user.state["gmsg_ts"] = now
                            user.hub.broadcast({"type": "g_msg", "text": text})
                            await ws.send_text(json.dumps({"type": "g_msg_ok"}))
                    elif t == "pair_new" and role == "pedestrian":
                        # 보행자 화면에 QR 로 띄워 보호자가 자기 폰 카메라로 찍게 한다 (멀리 있으면 같은 링크를 공유)
                        proto = ws.headers.get("x-forwarded-proto") or ("https" if ws.url.scheme == "wss" else "http")
                        url = f"{proto}://{ws.headers.get('host')}/guardian?pair={user.pairing.new_code()}" + (f"&k={key}" if key else "")
                        buf = io.BytesIO()
                        segno.make(url, error="m").save(buf, kind="svg", scale=8, border=3, dark="#000", light="#fff")
                        await ws.send_text(json.dumps({"type": "pair_link", "url": url, "ttl": Pairing.TTL,
                                                       "qr": buf.getvalue().decode()}))
                    elif t == "unpair_all" and role == "pedestrian":
                        user.pairing.clear()
                        for g in list(user.gws):
                            try:
                                await g.send_text(json.dumps({"type": "auth_error", "reason": "unpaired"}))
                                await g.close(code=4401)
                            except Exception:
                                pass
                        user.gws.clear()
                        await ws.send_text(json.dumps({"type": "unpaired"}))
                    elif t == "indoor" and role == "pedestrian":
                        user.state["indoor"] = bool(cmd.get("on"))
                        if user.pipe is not None and hasattr(user.pipe.perception, "indoor"):
                            user.pipe.perception.indoor = user.state["indoor"]
                    elif t == "gps" and role == "pedestrian":
                        user.alerts.on_location(cmd)
        except Exception:
            pass
        finally:
            task.cancel()
            user.hub.clients.pop(q, None)
            user.gws.discard(ws)
            if role != "guardian":
                user.ped = max(0, user.ped - 1)
                user.last_ped = time.time()

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
    app.state.users = users
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

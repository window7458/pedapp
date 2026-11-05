"""SayEye 서버 — 노트북(또는 GPU 서버)에서 돌리고, 보행자 · 보호자는 폰 브라우저로 접속한다.

    python server.py --device 0                 # weights/ 의 팀 가중치 + v5_29 히트박스 + VLM(localhost:8000)
    python server.py --device cpu --vlm none    # GPU 없는 노트북: VLM 설명 없이 경고만

카메라는 보행자 폰 카메라. 폰 카메라는 HTTPS 주소가 필요하다
(상시 서버는 deploy/ 의 Tailscale Funnel 주소, 같은 와이파이 시험은 --https).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import ipaddress
import hmac
import json
import os
import socket
import sys
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from alerts import AlertCenter  # noqa: E402
from push import Push  # noqa: E402
from sensors import SensorLog  # noqa: E402
from pipeline import Config, LivePipeline  # noqa: E402
from sources import SourceManager  # noqa: E402


GUARDIAN_TYPES = {"alert", "status", "loc"}   # 보호자 화면은 알림 · 상태 · 위치만 받는다 (박스 · 음성 문장은 안 받음)


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

    d = APP / "certs"
    d.mkdir(exist_ok=True)
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
    f = APP / "certs" / "guardian_pin.txt"
    f.parent.mkdir(exist_ok=True)
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    import secrets
    pin = "".join(secrets.choice("0123456789") for _ in range(6))
    f.write_text(pin)
    return pin


KEY_PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SayEye 접속 키</title>
<style>body{font:18px system-ui,sans-serif;margin:0;padding:32px 20px;background:#F3F4FB;color:#14182B}
main{max-width:420px;margin:auto}input,button{font:inherit;width:100%;box-sizing:border-box;min-height:56px;border-radius:14px;
margin-top:12px;padding:0 16px}input{border:2px solid #9AA3C7;background:#fff}button{border:0;background:#2F5BEA;color:#fff;font-weight:700}
@media (prefers-color-scheme:dark){body{background:#10131F;color:#EEF0FA}input{background:#1B2033;color:#EEF0FA}}</style></head>
<body><main><h1>SayEye</h1><p>처음 한 번만 접속 키를 넣어 주세요. 이 기기에서는 다음부터 묻지 않아요.</p>
<form method="get"><label for="k">접속 키</label><input id="k" name="k" autocomplete="current-password" autofocus required>
<button type="submit">열기</button></form></main></body></html>"""


def build_app(cfg: Config, pin: str = "000000", alerts_dir: Path | None = None,
              session_dir: Path | None = None, key: str | None = None):
    hub = Hub()
    state = {}
    sensors = SensorLog(session_dir or (APP / "sessions"))

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()
        alerts = AlertCenter(alerts_dir or (APP / "alerts"), hub.broadcast)
        push = Push(APP / "certs", alerts.dir)
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
        try:
            yield
        finally:
            src.close()
            pipe.stop()
            sensors.close()

    app = FastAPI(title="SayEye", lifespan=lifespan)

    def key_ok(q, cookies):
        return not key or hmac.compare_digest(str(q.get("k") or cookies.get("sayeye_k") or ""), key)

    @app.middleware("http")
    async def _key_guard(request: Request, call_next):
        """공개 주소(터널)에서 영상 · 위치가 새지 않게 접속 키를 요구한다. ?k=키 로 한 번 열면 쿠키로 기억."""
        if not key_ok(request.query_params, request.cookies):
            if "text/html" in request.headers.get("accept", ""):          # 화면을 열려는 경우: 키 입력 화면
                return Response(KEY_PAGE, status_code=401, media_type="text/html; charset=utf-8")
            return JSONResponse({"error": "접속 키가 필요해요"}, status_code=401)
        resp = await call_next(request)
        if key and request.query_params.get("k"):
            # 홈 화면 앱(시작 주소에 키가 없음)이 계속 열리도록 오래 기억한다 (브라우저 상한 약 400일)
            resp.set_cookie("sayeye_k", key, httponly=True, secure=True, samesite="lax", max_age=86400 * 400)
        return resp

    def check_pin(given: str | None):
        if not given or not hmac.compare_digest(str(given), str(pin)):
            raise HTTPException(401, "확인번호가 달라요")

    @app.get("/api/health")
    async def health():
        p, s = state["pipe"], state["src"]
        return {"ok": True, "source": s.mode, "perception": p.perception.name, "fps": p.fps(),
                "trigger": getattr(p.trigger, "name", "?"), "vlm": p.vlm_name, "vlm_up": p.vlm_up,
                "scene": p.scene_name}

    @app.get("/api/sensors")
    async def api_sensors(pin: str | None = None):
        check_pin(pin)
        return sensors.summary()

    @app.get("/api/live.mjpg")
    async def live(pin: str | None = None):
        """보호자 실시간 영상 (인식 박스 포함, 약 7 fps). 보는 동안만 서버가 미리보기를 만든다."""
        check_pin(pin)
        pipe = state["pipe"]

        async def gen():
            pipe.live_viewers += 1
            last = None
            try:
                while True:
                    jpg = pipe.preview_jpeg
                    if jpg is not None and jpg is not last:
                        last = jpg
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    await asyncio.sleep(0.14)
            finally:
                pipe.live_viewers = max(0, pipe.live_viewers - 1)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/api/location")
    async def api_location(pin: str | None = None):
        check_pin(pin)
        return {"loc": state["alerts"].loc}

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

    @app.get("/api/latest.jpg")
    async def api_latest(pin: str | None = None):
        check_pin(pin)
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
        if role not in ("pedestrian", "guardian"):
            await ws.close(code=4400)
            return
        if role == "guardian" and not (ws.query_params.get("pin") and hmac.compare_digest(ws.query_params["pin"], str(pin))):
            await ws.send_text(json.dumps({"type": "auth_error"}))
            await ws.close(code=4401)
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        hub.clients[q] = role
        p, s = state["pipe"], state["src"]
        await ws.send_text(json.dumps({"type": "hello", "source": s.mode,
                                       "vlm": p.vlm_name if p.vlm_up else "none", "scene": p.scene_name,
                                       "perception": p.perception.name, "trigger": getattr(p.trigger, "name", "?"),
                                       "weights": Path(str(p.cfg.weights)).name},
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
                    if role == "pedestrian":
                        s.on_phone_frame(m["bytes"])
                        sensors.frame("phone", None, len(m["bytes"]))
                elif m.get("text"):
                    try:
                        cmd = json.loads(m["text"])
                    except json.JSONDecodeError:
                        continue
                    if cmd.get("type") == "sensors" and role != "guardian":
                        sensors.add(cmd)
                    elif cmd.get("type") == "set_source" and role == "pedestrian":
                        s.use_phone()
                    elif cmd.get("type") == "set_scene" and role == "pedestrian":
                        p.set_scene(bool(cmd.get("on", True)))
                    elif cmd.get("type") == "sos" and role != "guardian":
                        state["alerts"].on_sos(p.last_frame)
                    elif cmd.get("type") == "client_event" and role != "guardian":
                        state["alerts"].on_client_event(cmd)
                    elif cmd.get("type") == "gps" and role == "pedestrian":
                        state["alerts"].on_location(cmd)
        except Exception:
            pass
        finally:
            task.cancel()
            hub.clients.pop(q, None)

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


def env(name: str, default=None):
    """상시 서버(systemd · deploy/sayeye.env)는 명령줄 대신 SAYEYE_* 환경변수로 설정한다."""
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def main():
    ap = argparse.ArgumentParser(description="SayEye 서버")
    ap.add_argument("--host", default=env("SAYEYE_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=env("SAYEYE_PORT"), help="기본: --https면 8443, 아니면 8080")
    ap.add_argument("--https", action="store_true", help="자체 서명 HTTPS (같은 와이파이에서 폰 카메라를 시험할 때)")
    ap.add_argument("--weights", default=env("SAYEYE_WEIGHTS", "auto"), help="auto(weights/ 폴더에서 찾기) 또는 .pt 경로")
    ap.add_argument("--device", default=env("SAYEYE_DEVICE", "cpu"), help="cpu 또는 GPU 번호(0)")
    ap.add_argument("--trigger", default=env("SAYEYE_TRIGGER", "hitbox"), help="hitbox(v5_29) | b0")
    ap.add_argument("--conf", type=float, default=env("SAYEYE_CONF"), help="YOLO 신뢰도 (기본 hitbox 0.2)")
    ap.add_argument("--cam-h", type=float, default=env("SAYEYE_CAM_H"), help="카메라 높이(m). 폰을 가슴 앞에 들면 1.3~1.5, 기본 1.70")
    ap.add_argument("--vlm", default=env("SAYEYE_VLM", "server"), choices=["server", "none"],
                    help="server: vLLM(Qwen2.5-VL) 설명 · 특별 상황. 나중에 켜져도 자동으로 붙어요")
    ap.add_argument("--vlm-url", default=env("SAYEYE_VLM_URL", "http://localhost:8000/v1"), help="p0/serve_vllm.py 가 띄운 주소")
    ap.add_argument("--vlm-model", default=env("SAYEYE_VLM_MODEL", "qwen2_5-vl-7b"))
    ap.add_argument("--scene-interval", type=float, default=float(env("SAYEYE_SCENE_INTERVAL", 5.0)), help="특별 상황을 묻는 최소 간격(초)")
    ap.add_argument("--max-fps", type=float, default=float(env("SAYEYE_MAX_FPS", 15.0)))
    ap.add_argument("--guardian-pin", default=env("SAYEYE_GUARDIAN_PIN"), help="보호자 확인번호 (기본: 처음 한 번 만들어 certs/guardian_pin.txt 에 저장)")
    ap.add_argument("--session-dir", default=env("SAYEYE_SESSION_DIR"), help="센서 기록 폴더 (기본 code/app/sessions)")
    ap.add_argument("--alerts-dir", default=env("SAYEYE_ALERTS_DIR"), help="경고 기록 폴더 (기본 code/app/alerts)")
    ap.add_argument("--key", default=env("SAYEYE_KEY"), help="접속 키. 공개 주소로 열 때 꼭 지정 (https://주소/?k=키)")
    ap.add_argument("--public-url", default=env("SAYEYE_PUBLIC_URL"), help="공개 주소 (안내 출력용)")
    a = ap.parse_args()
    if a.cam_h:
        os.environ["CAM_H"] = str(a.cam_h)         # hitbox_core 가 import 될 때 읽는다

    from pipeline import find_weights, vlm_alive
    w = find_weights(a.weights)
    if w is None:
        print("\n[오류] 팀 가중치(.pt)를 찾지 못했어요.\n"
              f"  {APP / 'weights'} 폴더에 final_10class_best.pt 를 넣고 다시 실행하세요.\n"
              "  (또는 --weights 경로/파일.pt)\n", flush=True)
        sys.exit(1)
    vlm, scene = a.vlm, ("openai" if a.vlm == "server" else "none")
    if vlm == "server" and not vlm_alive(a.vlm_url):
        print(f"[알림] VLM 서버({a.vlm_url})가 아직 응답하지 않아요. 켜질 때까지 경고만 내고, 켜지면 자동으로 설명이 붙어요.",
              flush=True)
    cfg = Config(perception="yolo", weights=str(w), device=a.device, conf=a.conf, trigger=a.trigger, vlm=vlm,
                 vlm_url=a.vlm_url, vlm_model=a.vlm_model, scene=scene, scene_url=a.vlm_url,
                 scene_interval=a.scene_interval, max_fps=a.max_fps)
    port = int(a.port or (8443 if a.https else 8080))
    kw = {}
    scheme = "http"
    ips = lan_ips()
    if a.https:
        crt, key = ensure_cert(ips)
        kw = dict(ssl_certfile=crt, ssl_keyfile=key)
        scheme = "https"
    pin = load_pin(a.guardian_pin)
    base = a.public_url.rstrip("/") if a.public_url else None
    if a.host in ("127.0.0.1", "localhost"):
        ips = []                                   # 이 기계 안에서만 열림 (공개는 Tailscale Funnel 이 맡음)
    urls = [base] if base else ([f"{scheme}://{ip}:{port}" for ip in ips] or [f"{scheme}://localhost:{port}"])
    q = f"?k={a.key}" if a.key else ""
    print("\n보행자 화면 (폰 브라우저):")
    for u in urls:
        print(f"   {u}/{q}")
    print(f"보호자 화면: {urls[0]}/guardian{q}   확인번호: {pin}")
    print(f"개발자 화면: {urls[0]}/dev{q}")
    if not a.https and not base:
        print("   폰 카메라 · 푸시 알림은 HTTPS 가 필요해요 → 상시 서버는 deploy/README.md 참고")
    print(f"\n가중치={w.name} · 트리거={a.trigger} · VLM={a.vlm_url if vlm == 'server' else '꺼짐'} · 장치={a.device}\n", flush=True)
    app = build_app(cfg, pin, alerts_dir=Path(a.alerts_dir) if a.alerts_dir else None,
                    session_dir=Path(a.session_dir) if a.session_dir else None, key=a.key)
    uvicorn.run(app, host=a.host, port=port, log_level="warning", proxy_headers=True, **kw)


if __name__ == "__main__":
    main()

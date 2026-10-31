"""Insight 앱 서버 — 노트북에서 돌리고, 폰은 브라우저로 접속한다.

    python server.py --https --perception demo --vlm mock --scene mock      # 모델 없이 UI · 흐름 시험
    python server.py --https --perception yolo --device 0 --vlm server --scene openai   # 서버(A5000) 실제 모델

폰이 카메라를 쓰려면 HTTPS가 필요하다 (브라우저 규칙). --https 면 자체 서명 인증서를 만들어 쓴다.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import ipaddress
import hmac
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from alerts import AlertCenter  # noqa: E402
from pipeline import Config, LivePipeline  # noqa: E402
from sources import SourceManager  # noqa: E402


GUARDIAN_TYPES = {"alert", "status"}          # 보호자 화면은 알림과 상태만 받는다 (박스 · 음성 문장은 안 받음)


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
            if role != "guardian" and typ == "alert":
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


def build_app(cfg: Config, source: str | None, pin: str = "000000", alerts_dir: Path | None = None):
    hub = Hub()
    app = FastAPI(title="Insight")
    state = {}

    def check_pin(given: str | None):
        if not given or not hmac.compare_digest(str(given), str(pin)):
            raise HTTPException(401, "확인번호가 달라요")

    @app.on_event("startup")
    async def _start():
        hub.loop = asyncio.get_running_loop()
        alerts = AlertCenter(alerts_dir or (APP / "alerts"), hub.broadcast)

        def emit(msg):
            hub.broadcast(msg)
            if msg.get("type") == "status":
                alerts.on_status(msg)
        pipe = LivePipeline(cfg, emit)
        pipe.on_warn = alerts.on_warn
        src = SourceManager(pipe, hub.broadcast)
        state.update(pipe=pipe, src=src, alerts=alerts)
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
                "trigger": getattr(p.trigger, "name", "?"), "vlm": p.vlm_name, "scene": p.scene_name}

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
        if role == "guardian" and not (ws.query_params.get("pin") and hmac.compare_digest(ws.query_params["pin"], str(pin))):
            await ws.send_text(json.dumps({"type": "auth_error"}))
            await ws.close(code=4401)
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        hub.clients[q] = role
        p, s = state["pipe"], state["src"]
        await ws.send_text(json.dumps({"type": "hello", "source": s.mode, "url": s.url, "default_url": source,
                                       "vlm": p.vlm_name, "scene": p.scene_name,
                                       "perception": p.perception.name, "trigger": getattr(p.trigger, "name", "?")},
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
                    s.on_phone_frame(m["bytes"])
                elif m.get("text"):
                    try:
                        cmd = json.loads(m["text"])
                    except json.JSONDecodeError:
                        continue
                    if cmd.get("type") == "set_source":
                        if cmd.get("mode") == "phone":
                            await asyncio.to_thread(s.use_phone)
                        elif cmd.get("mode") == "url" and cmd.get("url"):
                            await asyncio.to_thread(s.use_url, cmd["url"])
                    elif cmd.get("type") == "set_scene":
                        p.set_scene(bool(cmd.get("on", True)))
                    elif cmd.get("type") == "sos" and role != "guardian":
                        state["alerts"].on_sos(p.last_frame)
                    elif cmd.get("type") == "client_event" and role != "guardian":
                        state["alerts"].on_client_event(cmd)
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


def main():
    ap = argparse.ArgumentParser(description="Insight 앱 서버")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=None, help="기본: --https면 8443, 아니면 8000")
    ap.add_argument("--https", action="store_true", help="자체 서명 HTTPS (폰 카메라에 필요)")
    ap.add_argument("--source", default=None, help="서버가 여는 카메라: http://IP:81/stream · rtsp://… · udp://@:8554 · 0")
    ap.add_argument("--perception", default="none", choices=["none", "demo", "yolo"])
    ap.add_argument("--weights", default="E0", help="E0 · F10 · A5 또는 .pt 파일 경로 (hitbox 는 final_10class_best.pt 권장)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trigger", default="b0", help="b0 | hitbox(v5_29 히트박스) | none | 모듈:클래스")
    ap.add_argument("--conf", type=float, default=None, help="YOLO 신뢰도 (기본: hitbox 0.2, 그 밖 0.25)")
    ap.add_argument("--cam-h", type=float, default=None, help="hitbox: 카메라 높이(m). 원본 기본 1.70. 폰을 가슴 앞에 들면 1.3~1.5")
    ap.add_argument("--vlm", default="pt", choices=["none", "pt", "mock", "server"])
    ap.add_argument("--mock-latency", type=float, default=1.0)
    ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default="qwen2_5-vl-7b")
    ap.add_argument("--scene", default="none", choices=["none", "mock", "openai", "anthropic"])
    ap.add_argument("--scene-url", default=None, help="openai 호환 서버 주소 (기본 http://localhost:8000/v1)")
    ap.add_argument("--scene-model", default=None)
    ap.add_argument("--scene-interval", type=float, default=5.0, help="특별 상황을 묻는 최소 간격(초)")
    ap.add_argument("--max-fps", type=float, default=15.0)
    ap.add_argument("--guardian-pin", default=None, help="보호자 화면 확인번호 (기본: 처음 한 번 만들어 certs/guardian_pin.txt 에 저장)")
    a = ap.parse_args()
    if a.cam_h:
        import os
        os.environ["CAM_H"] = str(a.cam_h)         # hitbox_core 가 import 될 때 읽는다

    cfg = Config(perception=a.perception, weights=a.weights, device=a.device, conf=a.conf, trigger=a.trigger, vlm=a.vlm,
                 mock_latency=a.mock_latency, vlm_url=a.vlm_url, vlm_model=a.vlm_model, scene=a.scene,
                 scene_url=a.scene_url, scene_model=a.scene_model, scene_interval=a.scene_interval, max_fps=a.max_fps)
    port = a.port or (8443 if a.https else 8000)
    ips = lan_ips()
    kw = {}
    scheme = "http"
    if a.https:
        crt, key = ensure_cert(ips)
        kw = dict(ssl_certfile=crt, ssl_keyfile=key)
        scheme = "https"
    urls = [f"{scheme}://{ip}:{port}" for ip in ips] or [f"{scheme}://localhost:{port}"]
    pin = load_pin(a.guardian_pin)
    print("\n보행자 폰 브라우저로 접속:")
    for u in urls:
        print("  ", u)
    print(f"\n보호자 화면: {urls[0]}/guardian   확인번호: {pin}")
    if a.https:
        print("   (처음 한 번 '안전하지 않음' 경고가 뜨면 '고급 → 계속'을 누르세요)")
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(urls[0])
        qr.print_ascii(invert=True)
    except Exception:
        pass
    print(f"\n인식={a.perception} · Trigger={a.trigger} · VLM={a.vlm} · 특별 상황={a.scene}"
          + (f" · 서버 카메라={a.source}" if a.source else "") + "\n")
    uvicorn.run(build_app(cfg, a.source, pin), host=a.host, port=port, log_level="warning", **kw)


if __name__ == "__main__":
    main()

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
import json
import socket
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

from pipeline import Config, LivePipeline  # noqa: E402
from sources import SourceManager  # noqa: E402


class Hub:
    """파이프라인 스레드 → 접속한 모든 브라우저로 메시지 전달."""

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def broadcast(self, msg: dict):
        if self.loop is None or not self.clients:
            return
        s = json.dumps(msg, ensure_ascii=False)
        self.loop.call_soon_threadsafe(self._fanout, s)

    def _fanout(self, s: str):
        for q in list(self.clients):
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


def build_app(cfg: Config, source: str | None):
    hub = Hub()
    app = FastAPI(title="Insight")
    state = {}

    @app.on_event("startup")
    async def _start():
        hub.loop = asyncio.get_running_loop()
        pipe = LivePipeline(cfg, hub.broadcast)
        src = SourceManager(pipe, hub.broadcast)
        state.update(pipe=pipe, src=src)
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

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        hub.clients.add(q)
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
        except Exception:
            pass
        finally:
            task.cancel()
            hub.clients.discard(q)

    @app.get("/")
    async def index():
        return FileResponse(APP / "web" / "index.html")

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
    print("\n폰 브라우저로 접속:")
    for u in urls:
        print("  ", u)
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
    uvicorn.run(build_app(cfg, a.source), host=a.host, port=port, log_level="warning", **kw)


if __name__ == "__main__":
    main()

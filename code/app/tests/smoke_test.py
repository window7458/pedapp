"""서버 스모크 테스트 (모델 · 카메라 없이): 가짜 인식 + mock VLM + mock 장면 설명으로 흐름 전체를 확인한다.

    python tests/smoke_test.py            # 폰 카메라 경로 (웹소켓으로 합성 프레임 전송)
    python tests/smoke_test.py --mjpeg    # 서버 카메라 경로 (XIAO와 같은 MJPEG 주소를 합성 프레임으로 흉내)
"""
import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import websockets

APP = Path(__file__).resolve().parents[1]
PORT, MJ_PORT = 8011, 8012
rng = np.random.default_rng(0)
BASE = cv2.GaussianBlur(rng.integers(0, 255, (720, 960), dtype=np.uint8), (0, 0), 2)
BASE = cv2.cvtColor(BASE, cv2.COLOR_GRAY2BGR)


def frame(i):
    dx, dy = int(6 * np.sin(i / 7)), int(4 * np.cos(i / 9))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    im = cv2.warpAffine(BASE, M, (960, 720))
    return cv2.resize(im, (640, 480))


def jpg(i):
    return cv2.imencode(".jpg", frame(i), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


class MJPEG(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        i = 0
        try:
            while True:
                b = jpg(i)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b) + b + b"\r\n")
                i += 1
                time.sleep(1 / 12)
        except (BrokenPipeError, ConnectionResetError):
            pass


async def run(mjpeg: bool, seconds: float):
    seen, first = {}, {}
    t0 = time.time()
    async with websockets.connect(f"ws://localhost:{PORT}/ws", max_size=None) as ws:
        if mjpeg:
            await ws.send(json.dumps({"type": "set_source", "mode": "url", "url": f"http://localhost:{MJ_PORT}/stream"}))
        else:
            await ws.send(json.dumps({"type": "set_source", "mode": "phone"}))

        async def sender():
            i = 0
            while time.time() - t0 < seconds:
                if not mjpeg:
                    await ws.send(jpg(i))
                    i += 1
                await asyncio.sleep(1 / 12)

        task = asyncio.create_task(sender())
        while time.time() - t0 < seconds:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
            except asyncio.TimeoutError:
                continue
            ty = m["type"]
            seen[ty] = seen.get(ty, 0) + 1
            if ty not in first:
                first[ty] = (round(time.time() - t0, 1), m)
        await task
    return seen, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjpeg", action="store_true")
    ap.add_argument("--seconds", type=float, default=16)
    a = ap.parse_args()
    if a.mjpeg:
        srv = ThreadingHTTPServer(("127.0.0.1", MJ_PORT), MJPEG)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    p = subprocess.Popen([sys.executable, str(APP / "server.py"), "--port", str(PORT), "--perception", "demo",
                          "--vlm", "mock", "--mock-latency", "0.5", "--scene", "mock", "--scene-interval", "2"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=APP)
    try:
        time.sleep(4)
        seen, first = asyncio.run(run(a.mjpeg, a.seconds))
    finally:
        p.terminate()
        try:
            out = p.communicate(timeout=5)[0]
        except Exception:
            out = ""
    print("수신 메시지:", seen)
    for k in ("hello", "warn", "desc", "scene", "source", "status"):
        if k in first:
            print(f"  첫 {k} @ {first[k][0]}s:", json.dumps(first[k][1], ensure_ascii=False)[:230])
    ok = all(k in seen for k in ("dets", "warn", "desc", "scene", "status"))
    print("결과:", "통과" if ok else "실패 (빠진 메시지 있음)")
    if not ok:
        print("--- 서버 출력 ---\n", out[-2500:])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

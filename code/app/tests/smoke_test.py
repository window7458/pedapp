"""서버 실제 실행 스모크 테스트: 진짜 YOLO 가중치로 서버를 띄우고
  · 폰 역할로 영상(찍은 시각 헤더 포함)을 보내 → 보행자 화면이 박스(dets)를 받는지
  · 보호자: 위치(gps) 받기, 실시간 영상(/api/live.mjpg), 푸시 키
를 확인한다.  python tests/smoke_test.py --weights /경로/모델.pt   (기본: weights/ 폴더 또는 /tmp/yolo/yolov8n.pt)
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
PORT = 8791
PIN = "135790"


def frames(img_path: Path):
    im = cv2.imread(str(img_path)) if img_path.exists() else None
    if im is None:
        im = np.full((640, 480, 3), 128, np.uint8)
    H, W = im.shape[:2]
    i = 0
    while True:
        s = 1.0 + 0.003 * (i % 50)
        M = np.array([[s, 0, (1 - s) * W / 2], [0, s, (1 - s) * H / 2]], np.float32)
        ok, b = cv2.imencode(".jpg", cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REFLECT), [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield b.tobytes()
        i += 1


async def run(seconds: float, img: Path):
    import websockets
    seen, classes = {}, set()
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ped, \
            websockets.connect(f"ws://127.0.0.1:{PORT}/ws?role=guardian&pin={PIN}", max_size=None) as gd:
        hello = json.loads(await ped.recv())
        print("hello:", {k: hello.get(k) for k in ("perception", "trigger", "weights")})
        await gd.recv()
        t_boot = time.time()
        while True:                                                   # 판단 장치는 뒤에서 준비된다 (모델 올리기)
            m = json.loads(await asyncio.wait_for(ped.recv(), 180))
            if m["type"] == "ready":
                print(f"준비까지 {time.time() - t_boot:.1f}s")
                break
        await ped.send(json.dumps({"type": "set_source", "mode": "phone"}))
        await ped.send(json.dumps({"type": "share_allow"}))      # 보행자가 공유를 허락한 상황 (평소엔 위치 · 영상이 막혀 있음)
        await ped.send(json.dumps({"type": "gps", "lat": 37.4502, "lon": 126.6535, "acc": 8}))
        gen = frames(img)
        t0 = time.time()

        async def pump(ws, who):
            while time.time() - t0 < seconds:
                try:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 0.5))
                except asyncio.TimeoutError:
                    continue
                key = f"{who}:{m['type']}"
                seen[key] = seen.get(key, 0) + 1
                if m["type"] == "dets":
                    classes.update(d[1] for d in m["d"])

        async def send():
            import struct
            while time.time() - t0 < seconds:
                await ped.send(b"SEF1" + struct.pack("!d", time.time() * 1000) + next(gen))
                await asyncio.sleep(0.12)
        await asyncio.gather(pump(ped, "ped"), pump(gd, "guardian"), send())
    return seen, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--image", default="/tmp/yolo/bus.jpg")
    a = ap.parse_args()
    w = a.weights or ("/tmp/yolo/yolov8n.pt" if not any((APP / "weights").glob("*.pt")) else "auto")
    env = dict(os.environ, HB_SURFACE="0")
    p = subprocess.Popen([sys.executable, str(APP / "server.py"), "--port", str(PORT), "--weights", w,
                          "--guardian-pin", PIN], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=APP, env=env)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=1)
                break
            except Exception:
                time.sleep(1)
        seen, classes = asyncio.run(run(a.seconds, Path(a.image)))
        print("받은 메시지:", seen, "· 인식:", sorted(classes), flush=True)
        key = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/push/key").read())
        loc = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/location?pin={PIN}").read())
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/live.mjpg?pin={PIN}", timeout=5)
        chunk = r.fp.read1(200) if hasattr(r.fp, "read1") else r.read(200)
        r.close()
    finally:
        p.terminate()
        try:
            out = p.communicate(timeout=5)[0]
        except Exception:
            out = ""
    print("받은 메시지:", seen)
    print("인식한 물체:", sorted(classes))
    print("푸시 키:", key["ok"], "· 위치:", loc["loc"], "· 실시간 영상 첫 조각:", chunk[:40])
    ok = (seen.get("ped:dets", 0) > 5 and seen.get("guardian:loc", 0) >= 1 and seen.get("guardian:status", 0) > 3
          and key["ok"] and loc["loc"] and b"--frame" in chunk and "ped:loc" not in seen and "ped:alert" not in seen)
    print("결과:", "통과" if ok else "실패")
    if not ok:
        print("--- 서버 출력 ---\n", out[-3000:])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

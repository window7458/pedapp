"""라즈베리파이 카메라 → SayEye 서버로 영상 보내기.

라즈베리파이(제로 2 W · 4 · 5)에서 실행한다. 카메라 모듈(picamera2)이 있으면 그걸, 없으면 USB 웹캠을 쓴다.
서버 주소만 닿으면 되므로 같은 와이파이가 아니어도 된다 (Cloudflare 터널 주소도 OK).

    pip install websockets opencv-python-headless numpy        # (카메라 모듈은 sudo apt install python3-picamera2)
    python pi_camera.py --server https://xxxx.trycloudflare.com --pin 123456

보행자 앱 설정에서 카메라를 '라즈베리파이'로 바꾸면 이 영상으로 안내한다.
"""
from __future__ import annotations

import argparse
import asyncio
import ssl
import time


def open_camera(width: int, height: int, usb: int | None):
    if usb is None:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
            cam.start()
            time.sleep(0.5)
            print("카메라: 라즈베리파이 카메라 모듈")
            return lambda: cam.capture_array()             # RGB888 = OpenCV 가 쓰는 BGR 순서로 나온다
        except Exception as e:
            print(f"카메라 모듈을 못 열었어요 ({e!r}) → USB 웹캠으로 시도")
    import cv2
    cap = cv2.VideoCapture(usb or 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없어요. 카메라 연결을 확인해 주세요.")
    print("카메라: USB 웹캠")

    def read():
        ok, im = cap.read()
        return im if ok else None
    return read


async def run(a):
    import cv2
    import websockets
    read = open_camera(a.width, a.height, a.usb)
    base = a.server.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    url = f"{base}/ws?role=camera&pin={a.pin}"
    ctx = None
    if url.startswith("wss://"):
        ctx = ssl.create_default_context()
        if a.insecure:                                   # 자체 서명 인증서(--https) 서버일 때
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    gap = 1.0 / a.fps
    while True:
        try:
            async with websockets.connect(url, ssl=ctx, max_size=None, ping_interval=20) as ws:
                hello = await ws.recv()
                if '"auth_error"' in hello:
                    raise SystemExit("확인번호가 달라요. 서버 화면에 나온 번호를 --pin 으로 주세요.")
                print("서버에 연결됐어요. 보행자 앱 설정에서 카메라를 '라즈베리파이'로 바꾸세요.")
                sent, t0 = 0, time.time()
                while True:
                    t = time.time()
                    im = read()
                    if im is not None:
                        if a.rotate:
                            im = cv2.rotate(im, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a.rotate])
                        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                        if ok:
                            await ws.send(buf.tobytes())
                            sent += 1
                    if time.time() - t0 >= 10:
                        print(f"  보내는 중 {sent / (time.time() - t0):.1f} fps")
                        sent, t0 = 0, time.time()
                    await asyncio.sleep(max(0.0, gap - (time.time() - t)))
        except SystemExit:
            raise
        except Exception as e:
            print(f"연결이 끊겼어요 ({e!r}). 3초 뒤 다시 연결합니다.")
            await asyncio.sleep(3)


def main():
    ap = argparse.ArgumentParser(description="라즈베리파이 카메라 → SayEye 서버")
    ap.add_argument("--server", required=True, help="서버 주소 (예: https://xxxx.trycloudflare.com)")
    ap.add_argument("--pin", required=True, help="서버를 켤 때 나온 확인번호")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--usb", type=int, default=None, help="카메라 모듈 대신 USB 웹캠 번호")
    ap.add_argument("--insecure", action="store_true", help="자체 서명 인증서 서버에 연결할 때")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()

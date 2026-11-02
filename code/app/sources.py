"""영상 소스 — 폰 카메라 또는 라즈베리파이 카메라 (둘 다 웹소켓으로 JPEG 를 밀어 보낸다).

  phone : 보행자 폰 브라우저가 /ws 로 보냄
  pi    : 라즈베리파이에서 pi_camera.py 가 /ws?role=camera 로 보냄 (같은 와이파이가 아니어도 서버 주소만 닿으면 됨)
  url   : (개발용) 서버가 직접 여는 카메라 주소

서버 카메라로 쓸 수 있는 것
  http://<IP>:81/stream        XIAO ESP32S3 Sense (공식 CameraWebServer 예제의 MJPEG)
  rtsp://<IP>/...              RTSP를 내는 카메라
  udp://@:8554                 GoPro Wi-Fi 미리보기 (카메라 쪽에서 미리보기를 켠 뒤)
  0, 1, ...                    노트북에 꽂힌 USB 웹캠 번호
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np


def parse_source(text: str):
    text = text.strip()
    return int(text) if text.isdigit() else text


class SourceManager:
    def __init__(self, pipeline, emit, log=print):
        self.pipe, self.emit, self.log = pipeline, emit, log
        self.mode = "phone"
        self.url: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # 폰 ───────────────────────────────────────────────
    def on_phone_frame(self, data: bytes):
        if self.mode != "phone":
            return
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            self.pipe.submit(im)

    # 라즈베리파이 ───────────────────────────────────
    def on_camera_frame(self, data: bytes):
        if self.mode != "pi":
            return
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            self.pipe.submit(im)

    def use_pi(self):
        self._halt()
        self.mode, self.url = "pi", None
        self.pipe.preview_enabled = True             # 폰 화면에서 파이 카메라 영상을 보여 주려고
        self.pipe.set_source_label("raspberrypi")
        self.emit({"type": "source", "mode": "pi", "state": "ok"})

    def use_phone(self):
        self._halt()
        self.mode, self.url = "phone", None
        self.pipe.preview_enabled = False
        self.pipe.set_source_label("phone")
        self.emit({"type": "source", "mode": "phone", "state": "ok"})

    # 서버 카메라 ──────────────────────────────────────
    def use_url(self, url: str):
        self._halt()
        self.mode, self.url = "url", url.strip()
        self.pipe.preview_enabled = True
        self.pipe.set_source_label(self.url)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, args=(self.url, self._stop), daemon=True)
        self._thread.start()

    def _halt(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def _read_loop(self, url: str, stop: threading.Event):
        src = parse_source(url)
        if isinstance(src, str) and src.startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        while not stop.is_set():
            self.emit({"type": "source", "mode": "url", "url": url, "state": "connecting"})
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                cap.release()
                self.emit({"type": "source", "mode": "url", "url": url, "state": "error",
                           "error": "카메라에 연결할 수 없어요. 주소와 같은 네트워크인지 확인해 주세요."})
                stop.wait(3.0)
                continue
            self.emit({"type": "source", "mode": "url", "url": url, "state": "ok"})
            misses = 0
            while not stop.is_set():
                ok, im = cap.read()
                if not ok or im is None:
                    misses += 1
                    if misses > 30:
                        break
                    time.sleep(0.05)
                    continue
                misses = 0
                self.pipe.submit(im)
            cap.release()
            if not stop.is_set():
                self.emit({"type": "source", "mode": "url", "url": url, "state": "error",
                           "error": "영상이 끊겼어요. 다시 연결하는 중이에요."})
                stop.wait(1.0)

    def close(self):
        self._halt()

"""영상 소스 — 보행자 폰 카메라. 폰 브라우저가 /ws 로 JPEG 를 밀어 보낸다 (긴 변 960 px, 약 10 fps)."""
from __future__ import annotations

import cv2
import numpy as np


class SourceManager:
    def __init__(self, pipeline, emit, log=print):
        self.pipe, self.emit, self.log = pipeline, emit, log
        self.mode = "phone"
        self.pipe.set_source_label("phone")

    def on_phone_frame(self, data: bytes):
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            self.pipe.submit(im)

    def use_phone(self):
        self.emit({"type": "source", "mode": "phone", "state": "ok"})

    def close(self):
        pass

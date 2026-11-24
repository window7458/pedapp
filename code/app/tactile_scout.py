"""점자블록 자동 알림 (11/24): 평소 안내 중에 카메라에 점자블록이 담기면 폰에 'tactile_seen' 을 보낸다.
폰은 "점자블록이 있어요. 탐색을 시작할게요." 를 띄우고 점자블록 찾기(임태규 V6)를 시작한다 — 설정에서 켜고 끄는 것이 아님.

- 2초마다 최신 한 장만, 판단 고리와 따로(스레드) 돌린다 → 장애물 경고가 늦어지지 않게.
- 점자블록 모델(GuideTWSI YOLO11n-seg)을 한 장 · 640 · 신뢰도 0.5 로 (찾기 모드는 좌우 두 장 · 960 으로 더 엄격).
- 화면 아래 60 %(땅)에서 1 % 넘게 두 번 이어 보이면 알림, 그 뒤 90초는 다시 알리지 않는다.
- TACTILE_SCOUT=0 이면 끈다.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

EVERY_S, NEED, COOL_S, MIN_AREA = 2.0, 2, 90.0, 0.01


class TactileScout:
    def __init__(self, device, emit, log=print):
        self.device, self.emit, self.log = str(device), emit, log
        self.on = os.environ.get("TACTILE_SCOUT", "1") != "0"
        self.busy = False
        self.at = 0.0
        self.hits = 0
        self.cool_until = 0.0
        self.err_at = 0.0

    def offer(self, image):
        """판단 고리가 매 장 부른다. 2초가 지났고 쉬고 있으면 그 장을 따로 본다."""
        now = time.monotonic()
        if not self.on or self.busy or now - self.at < EVERY_S or now < self.cool_until:
            return
        self.busy, self.at = True, now
        threading.Thread(target=self._look, args=(image.copy(),), daemon=True).start()

    def reset(self):
        self.hits = 0

    def _look(self, image):
        try:
            from tactile_feature import runtime
            from PIL import Image
            with runtime._MODEL_LOCK:
                if self.device not in runtime._MODELS:
                    from pathlib import Path
                    from tactile_feature.segment_tactile import TactileSegmenter
                    w = Path(runtime.__file__).resolve().parents[1] / "weights" / "yolo11n_tactile.pt"
                    runtime._MODELS[self.device] = TactileSegmenter(weights=w, device=self.device, threads=2)
                model = runtime._MODELS[self.device]
                r = model.predict(Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])), imgsz=640, conf=0.5, mode="single")
            mask = np.asarray(r.get("mask"), dtype=bool)
            h = mask.shape[0] if mask.ndim == 2 else 0
            area = float(mask[int(h * 0.4):].mean()) if h else 0.0
            self.hits = self.hits + 1 if area >= MIN_AREA else 0
            if self.hits >= NEED:
                self.hits = 0
                self.cool_until = time.monotonic() + COOL_S
                self.emit({"type": "tactile_seen", "area": round(area, 3)})
        except Exception as e:                         # 알림 기능이 안내를 멈추게 하면 안 된다
            if time.monotonic() - self.err_at > 30:
                self.err_at = time.monotonic()
                self.log(f"[tactile_scout] {type(e).__name__}: {e}")
        finally:
            self.busy = False

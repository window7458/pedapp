"""실시간 얼굴 가리기 — 서버 밖으로 나가거나 저장되는 영상에만 쓴다 (판단용 원본 프레임은 그대로).

가리는 곳: 보호자 실시간 영상 · '지금 화면' · 알림 사진 · 위험 구간 녹화 · 장애물 원본 저장 · VLM 입력.
  1) YuNet 얼굴 검출 (OpenCV 내장 FaceDetectorYN, 모델 230 KB) — 긴 변 640 으로 줄여서 돌린다 (CPU 약 5–15 ms)
  2) 이미 나온 사람(person) 박스의 머리 부분 — 추가 계산 없음. 멀리 있어 YuNet 이 놓치는 작은 얼굴 · 뒤통수 대비
환경변수 FACE_BLUR=0 이면 끈다.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2

MODEL = Path(__file__).resolve().parent / "weights" / "face_yunet.onnx"


class FaceBlur:
    def __init__(self, model: Path = MODEL, max_side: int = 640, score: float = 0.6, log=print):
        self.enabled = os.environ.get("FACE_BLUR", "1") != "0"
        self.max_side, self.score = max_side, score
        self.det = None
        self.name = "off"
        if not self.enabled:
            return
        if Path(model).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score, 0.3, 50)
                self.name = "yunet+person"
            except Exception as e:
                log(f"[face] YuNet 을 못 열었어요 ({e!r}) → 사람 박스 머리 부분만 가립니다")
        if self.det is None:
            self.name = "person"

    def faces(self, im) -> list[tuple[int, int, int, int]]:
        if self.det is None:
            return []
        H, W = im.shape[:2]
        s = min(1.0, self.max_side / max(H, W))
        small = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, f = self.det.detect(small)
        out = []
        for r in (f if f is not None else []):
            x, y, w, h = (float(v) / s for v in r[:4])
            m = 0.25 * max(w, h)                               # 머리카락 · 귀까지
            out.append((int(x - m), int(y - m), int(x + w + m), int(y + h + m)))
        return out

    @staticmethod
    def heads(dets) -> list[tuple[int, int, int, int]]:
        out = []
        for d in dets or ():
            if getattr(d, "cls", None) != "person":
                continue
            x1, y1, x2, y2 = d.box
            w, h = x2 - x1, y2 - y1
            if h > 1.4 * w:                                    # 서 있는 사람: 머리 ≈ 키의 1/7 → 위 18 %, 폭은 키의 20 %
                hw, hh = min(w, 0.20 * h), 0.18 * h
            else:                                              # 앉은 · 잘린 · 탈것 탄 박스: 위쪽 정사각형 일부
                hw = hh = 0.45 * min(w, h)
            cx = (x1 + x2) / 2
            out.append((int(cx - 0.5 * hw), int(y1 - 0.02 * h), int(cx + 0.5 * hw), int(y1 + hh)))
        return out

    def apply(self, im, dets=()):
        """가린 사본을 돌려준다 (원본은 그대로). 가릴 곳이 없으면 원본을 그대로 돌려준다."""
        if not self.enabled:
            return im
        boxes = self.faces(im) + self.heads(dets)
        if not boxes:
            return im
        out = im.copy()
        H, W = im.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            roi = out[y1:y2, x1:x2]
            k = max(4, min(roi.shape[:2]) // 6)                # 모자이크: 한 변 6칸 안팎
            small = cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k)), interpolation=cv2.INTER_AREA)
            out[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out

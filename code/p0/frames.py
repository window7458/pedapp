"""트리거 시각까지의 프레임 창을 뽑고, 대상 bbox를 표시해 VLM 입력 이미지로 만든다."""
from __future__ import annotations

import base64
import bisect
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import FRAMES, IMG_LONG_SIDE


def window_times(t_trigger: float, setting: str) -> list[float]:
    """마지막 프레임 = 트리거 시각. 과거로 1/fps 간격. 0초 이전은 잘라낸다 (미래 프레임은 절대 안 씀)."""
    s = FRAMES[setting]
    if s["n"] == 1:
        return [t_trigger]
    ts = [t_trigger - k / s["fps"] for k in range(s["n"] - 1, -1, -1)]
    return [t for t in ts if t >= 0.0]


class Video:
    def __init__(self, path: Path, rotate: int | None = None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.rotate = rotate

    def at(self, t: float) -> np.ndarray:
        idx = min(max(int(round(t * self.fps)), 0), self.n - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = self.cap.read()
        if not ok:
            raise RuntimeError(f"frame read failed t={t:.2f}")
        return cv2.rotate(im, self.rotate) if self.rotate is not None else im

    def close(self):
        self.cap.release()


class Tracks:
    """dump_tracks.py 출력 CSV → track_id별 (t, bbox). 좌표는 회전 보정 후, 긴 변 1920 이하로 줄인 프레임 기준
    (common.iter_frames max_side=1920) → CSV의 W, H로 저장해 두고 원본 해상도로 환산한다."""

    def __init__(self, csv_path: Path):
        self.by = defaultdict(list)
        self.W = self.H = None
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            self.W, self.H = float(r["W"]), float(r["H"])
            self.by[int(r["track_id"])].append(
                (float(r["t_sec"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")), r["cls"]))
        for v in self.by.values():
            v.sort()

    def bbox(self, tid: int, t: float, tol: float = 0.25):
        rows = self.by.get(tid)
        if not rows:
            return None
        i = bisect.bisect_left(rows, (t,))
        cand = [rows[j] for j in (i - 1, i) if 0 <= j < len(rows)]
        best = min(cand, key=lambda r: abs(r[0] - t))
        return best[1] if abs(best[0] - t) <= tol else None

    def visible(self, t: float, tol: float = 0.1) -> list[tuple[int, str]]:
        """시각 t에 보이는 (track_id, cls) — 채점의 혼동·환각 판정용."""
        out = []
        for tid, rows in self.by.items():
            i = bisect.bisect_left(rows, (t,))
            for j in (i - 1, i):
                if 0 <= j < len(rows) and abs(rows[j][0] - t) <= tol:
                    out.append((tid, rows[j][2]))
                    break
        return out


def _resize(im: np.ndarray, long_side: int = IMG_LONG_SIDE) -> np.ndarray:
    s = long_side / max(im.shape[:2])
    return cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else im


def _draw(im, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    th = max(2, round(max(im.shape[:2]) / 200))
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), th)          # 빨간 박스 (BGR)
    return im


def _crop(im, box, scale=1.5):
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * scale, (y2 - y1) * scale
    side = max(w, h, 32)
    xa, ya = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    xb, yb = int(min(im.shape[1], cx + side / 2)), int(min(im.shape[0], cy + side / 2))
    return im[ya:yb, xa:xb]


def build_images(video: Video, tracks: Tracks | None, event: dict, setting: str, bbox_mode: str):
    """→ (images[BGR], times, boxes_in_original_px). bbox_mode는 config.BBOX_MODES.
    crop: 각 시각의 대상 크롭 + 트리거 시각 전체 프레임 1장(저해상도, 위치 맥락)."""
    times = window_times(event["t"], setting)
    tid = event.get("track_id")
    imgs, boxes = [], []
    for t in times:
        im = video.at(t)
        box = tracks.bbox(tid, t) if (tracks is not None and tid is not None) else None
        if box is not None:                                    # 트랙 좌표계 → 이 프레임 해상도
            sx, sy = im.shape[1] / tracks.W, im.shape[0] / tracks.H
            box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        boxes.append(box)
        if bbox_mode in ("draw", "draw+text") and box is not None:
            im = _draw(im.copy(), box)
        if bbox_mode == "crop" and box is not None:
            im = _crop(im, box)
        imgs.append(_resize(im))
    if bbox_mode == "crop":
        full = video.at(times[-1])
        if boxes[-1] is not None:
            full = _draw(full.copy(), boxes[-1])
        imgs.append(_resize(full, IMG_LONG_SIDE // 2))
    return imgs, times, boxes


def to_data_url(im: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

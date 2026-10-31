"""배경 움직임 (카메라 ego-motion의 2D 근사) — 임태규 노트북 BackgroundMotion 이식.

검출 박스를 가린 배경 특징점 → LK 광류(정·역 검사) → RANSAC 유사변환 (회전·확대·이동).
  · B0(1차 규칙)의 박스 중심 보정에 그대로 쓰고
  · 기하에서는 수평 이동 → 머리 yaw 회전, 확대율 → 전진 여부의 보조 신호로 쓴다.

    python motion.py --model E0      → results/motion/<clip>.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np

from common import CLIPS, OUT, iter_frames

GMC_IMAGE_SIZE = 640
GMC_MIN_POINTS = 20
GMC_MIN_INLIER_RATIO = 0.6
GMC_MAX_FB_ERROR = 1.5


class BackgroundMotion:
    def __init__(self):
        self.prev_gray = None
        self.prev_mask = None

    def update(self, frame, boxes):
        h, w = frame.shape[:2]
        scale = min(1.0, GMC_IMAGE_SIZE / max(h, w))
        sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
        gray = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY)
        mask = np.full((sh, sw), 255, dtype=np.uint8)
        sx, sy = sw / w, sh / h
        for x1, y1, x2, y2 in boxes:
            l, t = max(0, int(np.floor(x1 * sx)) - 5), max(0, int(np.floor(y1 * sy)) - 5)
            r, b = min(sw, int(np.ceil(x2 * sx)) + 5), min(sh, int(np.ceil(y2 * sy)) + 5)
            if r > l and b > t:
                mask[t:b, l:r] = 0
        prev_gray, prev_mask = self.prev_gray, self.prev_mask
        self.prev_gray, self.prev_mask = gray, mask
        info = {"ok": 0, "reason": "INIT", "points": 0, "inliers": 0}
        if prev_gray is None or prev_gray.shape != gray.shape:
            return None, info
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=10,
                                     mask=prev_mask, blockSize=7)
        if p0 is None or len(p0) < GMC_MIN_POINTS:
            info["reason"] = "FEW_FEATURES"
            return None, info
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
        if p1 is None:
            info["reason"] = "FLOW_FAILED"
            return None, info
        ok = st1.ravel().astype(bool) & np.isfinite(p1.reshape(-1, 2)).all(1)
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        if len(a) < GMC_MIN_POINTS:
            info["reason"] = "FEW_TRACKS"
            return None, info
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, b.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm(back.reshape(-1, 2) - a, axis=1)
        ok = st2.ravel().astype(bool) & (fb <= GMC_MAX_FB_ERROR) & (b[:, 0] >= 0) & (b[:, 0] < sw) & (b[:, 1] >= 0) & (b[:, 1] < sh)
        a, b = a[ok], b[ok]
        if len(b):
            idx = np.floor(b).astype(int)
            bg = mask[idx[:, 1], idx[:, 0]] > 0
            a, b = a[bg], b[bg]
        info["points"] = len(a)
        if len(a) < GMC_MIN_POINTS:
            info["reason"] = "FEW_BACKGROUND_TRACKS"
            return None, info
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                             maxIters=2000, confidence=0.99, refineIters=10)
        if M is None:
            info["reason"] = "RANSAC_FAILED"
            return None, info
        im_ = inl.ravel().astype(bool)
        info["inliers"] = int(im_.sum())
        if not np.isfinite(M).all() or im_.sum() < GMC_MIN_POINTS or im_.mean() < GMC_MIN_INLIER_RATIO:
            info["reason"] = "LOW_CONFIDENCE"
            return None, info
        spread = np.ptp(a[im_], axis=0)
        if spread[0] < sw * 0.2 or spread[1] < sh * 0.2:
            info["reason"] = "LOCAL_FEATURES"
            return None, info
        s = float(np.hypot(M[0, 0], M[1, 0]))
        if not 0.9 <= s <= 1.1:
            info["reason"] = "EXTREME_SCALE"
            return None, info
        S = np.eye(3)
        S[:2] = M
        R = np.diag([sx, sy, 1.0])
        info["ok"], info["reason"] = 1, "OK"
        return (np.linalg.inv(R) @ S @ R)[:2], info


def read_tracks(path):
    by = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[int(r["frame"])].append(r)
    return by


def run_clip(model_id, key, fps=15.0):
    tracks = read_tracks(OUT / "tracks" / model_id / f"{key}.csv")
    bm = BackgroundMotion()
    rows = []
    for fidx, t, im in iter_frames(key, fps, max_side=1920):
        boxes = [(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])) for r in tracks.get(fidx, [])]
        M, info = bm.update(im, boxes)
        if M is None:
            rows.append([fidx, round(t, 3), 0, info["reason"], 1, 0, 0, 0, 1, 0, im.shape[1], im.shape[0]])
            continue
        rows.append([fidx, round(t, 3), 1, "OK", *[round(float(v), 6) for v in M.ravel()], im.shape[1], im.shape[0]])
    od = OUT / "motion"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_sec", "ok", "reason", "a", "b", "tx", "c", "d", "ty", "W", "H"])
        w.writerows(rows)
    okr = np.mean([r[2] for r in rows])
    print(f"{key}: frames {len(rows)} ok {okr:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E0")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    a = ap.parse_args()
    for k in a.clips:
        run_clip(a.model, k)

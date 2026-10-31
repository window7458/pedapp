"""B0 — 1차 규칙 트리거 (임태규 노트북 'Untitled3', taegyu 브랜치) 를 같은 E0 트랙 위에서 재현.

  접근: 배경 움직임 보정한 박스 중심이 거의 안 움직이고(≤0.03 화면/s) 면적이 ×1.10 이상 커지는 상태가 연속 → 1회 발화
  신호: HSV 빨강/초록 판정이 3회 연속 같으면 안정 색, 안정 색이 바뀌면 발화

원본은 원본 fps(≈30)에서 프레임 단위 파라미터. 여기는 15 fps 샘플이므로 같은 '시간'이 되게 환산:
  window 15→8 프레임, avg 5→3, streak 5→3 (≈0.2 s), 중심속도·면적비는 그대로.
클래스 대응: COCO vehicle(bicycle, car, motorcycle, bus, truck) → seg10 bicycle·scooter·motorcycle·car·bus·other_vehicle
             COCO obstacle(bench, potted plant, chair) → seg10 obstacle  /  conf ≥ 0.5 (원본 CONF_THRESHOLD)
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque

import numpy as np

from common import CLIPS, OUT, clip_info, iter_frames


def hsv_scores(crop_bgr: np.ndarray) -> tuple[float, float]:
    """빨강·초록 램프 픽셀 비율 (임태규 노트북 기준값)."""
    import cv2
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h < 10) | (h > 170)) & (s > 70) & (v > 100)).mean()
    green = ((h > 35) & (h < 90) & (s > 50) & (v > 80)).mean()
    return float(red), float(green)

VEH = {"bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle"}
OBS = {"obstacle"}
WINDOW, AVG, STREAK = 8, 3, 3
GROWTH, CENTER_SPEED_MAX, MAX_GAP_S, CONF = 1.10, 0.03, 0.3, 0.5


def hsv_color(crop):
    red, green = hsv_scores(crop)
    n = crop.shape[0] * crop.shape[1]
    mn = max(3, int(n * 0.005)) / n
    if max(red, green) < mn:
        return "unknown"
    if red > green * 1.2:
        return "red"
    if green > red * 1.2:
        return "green"
    return "unknown"


def run_clip(model_id, key, fps=15.0):
    by = defaultdict(list)
    for r in csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")):
        if float(r["conf"]) >= CONF:
            by[int(r["frame"])].append(r)
    mot = {int(r["frame"]): r for r in csv.DictReader(open(OUT / "motion" / f"{key}.csv", encoding="utf-8"))}
    hist = defaultdict(lambda: deque(maxlen=WINDOW))
    last_seen, streak, fired = {}, defaultdict(int), set()
    col_hist, last_stable, light_fired = defaultdict(lambda: deque(maxlen=3)), {}, set()
    events = []
    need_frames = any(r["cls"] == "traffic_light" for rs in by.values() for r in rs)
    frames = iter_frames(key, fps, max_side=1920) if need_frames else ((f, float(mot[f]["t_sec"]), None) for f in sorted(mot))
    for fidx, t, im in frames:
        m = mot.get(fidx)
        cam_ok = m is not None and int(m["ok"]) == 1
        if not cam_ok:
            hist.clear()
            streak.clear()
        else:
            A = np.array([[float(m["a"]), float(m["b"]), float(m["tx"])], [float(m["c"]), float(m["d"]), float(m["ty"])]])
            for k in list(hist):
                if not hist[k] or t - hist[k][-1]["t"] > MAX_GAP_S:
                    del hist[k]
                    streak.pop(k, None)
                    continue
                for o in hist[k]:
                    o["cc"] = A[:, :2] @ o["cc"] + A[:, 2]
        for r in by.get(fidx, []):
            tid, cls = int(r["track_id"]), r["cls"]
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            W, H = float(r["W"]), float(r["H"])
            if cls in VEH or cls in OBS:
                prev = last_seen.get(tid)
                last_seen[tid] = t
                if prev is not None and t - prev > 1.5 / fps:
                    streak[tid] = 0
                c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                h = hist[tid]
                h.append({"t": t, "area": max(0, x2 - x1) * max(0, y2 - y1), "cc": c.copy()})
                if len(h) < WINDOW:
                    continue
                obs = list(h)
                past, rec = obs[:AVG], obs[-AVG:]
                g = np.mean([o["area"] for o in rec]) / max(np.mean([o["area"] for o in past]), 1.0)
                dt = np.mean([o["t"] for o in rec]) - np.mean([o["t"] for o in past])
                sp = float(np.linalg.norm((np.mean([o["cc"] for o in rec], 0) - np.mean([o["cc"] for o in past], 0))
                                          / np.array([W, H])) / max(dt, 1e-6))
                ok = g >= GROWTH and sp <= CENTER_SPEED_MAX and cam_ok
                streak[tid] = streak[tid] + 1 if ok else 0
                if streak[tid] >= STREAK and tid not in fired:
                    fired.add(tid)
                    events.append(dict(t=round(t, 3), scenario="S1" if cls in VEH else "S2", kind="warn",
                                       track_id=tid, cls=cls, text=f"{cls} 접근", growth=round(float(g), 2)))
            elif cls == "traffic_light" and im is not None:
                xa, ya, xb, yb = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
                col = "unknown" if xb <= xa or yb <= ya else hsv_color(im[ya:yb, xa:xb])
                q = col_hist[tid]
                q.append(col)
                if len(q) == 3 and q[0] == q[1] == q[2] and q[0] in ("red", "green"):
                    prev = last_stable.get(tid)
                    if prev in ("red", "green") and prev != q[0] and (tid, prev, q[0]) not in light_fired:
                        light_fired.add((tid, prev, q[0]))
                        events.append(dict(t=round(t, 3), scenario="S3", kind="warn", track_id=tid, cls=cls,
                                           text=f"신호 {prev}→{q[0]}", state="GO" if q[0] == "green" else "STOP"))
                    last_stable[tid] = q[0]
    return events


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="B0 1차 규칙 → results/events/b0_<model>/<clip>.json")
    ap.add_argument("--model", default="E0")
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    a = ap.parse_args()
    od = OUT / "events" / f"b0_{a.model}"
    od.mkdir(parents=True, exist_ok=True)
    for key in a.clips:
        ev = run_clip(a.model, key)
        json.dump({"clip": key, "system": "b0", "model": a.model, "dur": clip_info(key)["dur"], "events": ev},
                  open(od / f"{key}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print(f"{key:<22} {len(ev):>3} events", flush=True)

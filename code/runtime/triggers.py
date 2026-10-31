"""② 트리거 — 교체 가능한 인터페이스.

    class 아무트리거:
        name = "..."
        def step(self, frame: core.Frame) -> list[core.Event]: ...

frame에는 영상 시각 · 이미지 · 이 프레임의 트랙(Det 목록)이 들어 있다. 트리거는 상태(이력 · 배경 움직임 등)를
스스로 들고 있고, '지금 말해야 할 것'만 Event로 돌려준다. 임태규 최종 규칙 코드는 이 모양으로 감싸서 TRIGGERS에 추가.

B0Trigger     : 1차 규칙(baseline_rule.py)을 프레임 단위로 옮긴 것. 오프라인 결과와 같은지 test_b0_stream.py로 확인
ReplayTrigger : 미리 만든 이벤트 파일(results/events/<이름>/<clip>.json)을 시각에 맞춰 내보냄 (다른 트리거 비교용)
"""
from __future__ import annotations

import json
from collections import defaultdict, deque

import numpy as np

import core  # noqa: F401
from baseline_rule import (AVG, CENTER_SPEED_MAX, CONF, GROWTH, MAX_GAP_S, OBS, STREAK, VEH, WINDOW,
                           hsv_color)
from common import OUT
from core import Event, Frame
from motion import BackgroundMotion


class B0Trigger:
    """배경 움직임 보정한 박스 중심이 거의 안 움직이고(≤0.03 화면/s) 면적이 ×1.10 이상 커지는 상태가 0.2 s 연속 → 1회.
    신호: HSV 빨강/초록이 3회 연속 같으면 안정 색, 안정 색이 바뀌면 1회."""
    name = "b0"

    def __init__(self, fps=15.0):
        self.fps = fps
        self.bm = BackgroundMotion()
        self.hist = defaultdict(lambda: deque(maxlen=WINDOW))
        self.last_seen, self.streak, self.fired = {}, defaultdict(int), set()
        self.col_hist, self.last_stable, self.light_fired = defaultdict(lambda: deque(maxlen=3)), {}, set()

    def step(self, fr: Frame) -> list[Event]:
        t, im = fr.t, fr.image
        A, _ = self.bm.update(im, [d.box for d in fr.dets])
        cam_ok = A is not None
        if not cam_ok:
            self.hist.clear()
            self.streak.clear()
        else:
            for k in list(self.hist):
                if not self.hist[k] or t - self.hist[k][-1]["t"] > MAX_GAP_S:
                    del self.hist[k]
                    self.streak.pop(k, None)
                    continue
                for o in self.hist[k]:
                    o["cc"] = A[:, :2] @ o["cc"] + A[:, 2]
        out = []
        W, H = float(fr.W), float(fr.H)
        for d in fr.dets:
            if d.conf < CONF:
                continue
            tid, cls = d.track_id, d.cls
            x1, y1, x2, y2 = d.box
            if cls in VEH or cls in OBS:
                prev = self.last_seen.get(tid)
                self.last_seen[tid] = t
                if prev is not None and t - prev > 1.5 / self.fps:
                    self.streak[tid] = 0
                c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                h = self.hist[tid]
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
                self.streak[tid] = self.streak[tid] + 1 if ok else 0
                if self.streak[tid] >= STREAK and tid not in self.fired:
                    self.fired.add(tid)
                    out.append(Event(t=round(t, 3), cls=cls, scenario="S1" if cls in VEH else "S2", track_id=tid,
                                     box=d.box, W=fr.W, H=fr.H, extra={"growth": round(float(g), 2)}))
            elif cls == "traffic_light" and im is not None:
                xa, ya, xb, yb = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
                col = "unknown" if xb <= xa or yb <= ya else hsv_color(im[ya:yb, xa:xb])
                q = self.col_hist[tid]
                q.append(col)
                if len(q) == 3 and q[0] == q[1] == q[2] and q[0] in ("red", "green"):
                    prev = self.last_stable.get(tid)
                    if prev in ("red", "green") and prev != q[0] and (tid, prev, q[0]) not in self.light_fired:
                        self.light_fired.add((tid, prev, q[0]))
                        out.append(Event(t=round(t, 3), cls=cls, scenario="S3", track_id=tid, box=d.box, W=fr.W, H=fr.H,
                                         extra={"state": "GO" if q[0] == "green" else "STOP"}))
                    self.last_stable[tid] = q[0]
        return out


class ReplayTrigger:
    """미리 계산한 이벤트를 시각에 맞춰 낸다. 박스는 같은 트랙의 현재 프레임 박스로 채운다."""

    def __init__(self, src: str, key: str):
        self.name = f"replay:{src}"
        ev = json.load(open(OUT / "events" / src / f"{key}.json", encoding="utf-8"))["events"]
        self.pending = sorted((e for e in ev if e.get("kind", "warn") == "warn"), key=lambda e: e["t"])

    def step(self, fr: Frame) -> list[Event]:
        out = []
        while self.pending and self.pending[0]["t"] <= fr.t + 1e-6:
            e = self.pending.pop(0)
            det = next((d for d in fr.dets if d.track_id == e.get("track_id")), None)
            cls = e.get("cls") or (det.cls if det else "obstacle")
            extra = {k: e[k] for k in ("state", "text") if k in e}
            out.append(Event(t=fr.t, cls=cls, scenario=e.get("scenario", "S2"), track_id=e.get("track_id"),
                             box=det.box if det else None, W=fr.W, H=fr.H, extra=extra))
        return out


def make_trigger(spec: str, key: str, fps: float):
    if spec == "b0":
        return B0Trigger(fps)
    if spec.startswith("replay:"):
        return ReplayTrigger(spec.split(":", 1)[1], key)
    raise ValueError(f"unknown trigger {spec} (b0 | replay:<results/events 폴더>)")

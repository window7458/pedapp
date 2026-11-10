"""임태규 v12.1 보행 트리거 (taegyu 브랜치 navigation_trigger) — 웹앱 연결 (`--trigger nav`, 기본값).

규칙 코드는 navigation_trigger/trigger/ 를 그대로 가져왔다 (고치지 않음). 모델은 같은 팀 10클래스 가중치
(weights/final_10class_best.pt — taegyu 의 trigger/models/ 와 해시가 같음).

이 트리거는 자기 Pipeline 안에서 YOLO 추론(전체 + 0.4 s마다 잘라 보기)까지 직접 한다.
그래서 앱의 YoloTracker 를 따로 돌리지 않고, 이 객체가 '인식기'와 '트리거'를 함께 맡는다
(같은 모델을 두 번 돌리면 CPU 에서 실시간이 안 나온다).

  perception(fidx, image) -> [Det]   Pipeline.step 을 돌리고, 추적된 검출을 화면 박스로 돌려준다
  step(fr)                -> [Event] 같은 프레임에서 나온 경고를 앱의 Event 모양으로 바꾼다
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from core import Det, Event

NAV = Path(__file__).resolve().parent / "navigation_trigger"
if str(NAV / "trigger") not in sys.path:
    sys.path.insert(0, str(NAV / "trigger"))

NAME_FIX = {"traffic light": "traffic_light", "e_scooter": "scooter", "kickboard": "scooter"}


def _scenario(e: dict) -> str:
    if e["event"].startswith("light_"):
        return "S3"
    return "S2" if e.get("class_id") in (7, 8) else "S1"


def _prio(e: dict) -> int:
    """폰 화면 단계: 3 빨강(위험) · 2 노랑(주의) · 1 안내만. 트리거의 priority 를 그대로 옮긴다."""
    if e["event"].startswith("light_"):
        return 1
    if e.get("priority") == "urgent":
        return 3
    return 2


class NavTrigger:
    name = "nav_v12.1"
    is_trigger = True           # pipeline: 이 객체가 인식기이자 트리거
    self_deduped = True         # 같은 대상 재경고 · 같은 종류 쿨다운은 HazardPolicy 가 이미 한다
    wants_vlm = False           # 문장이 이미 방향 · 대상 · 움직임을 말한다 (vlm_min_prio 이상만 설명을 붙임)

    def __init__(self, weights, device="cpu", imgsz=None, crop=True, log=print):
        from single_model import Pipeline
        dev = str(device)
        dev = "cpu" if dev == "cpu" else dev.replace("cuda:", "").replace("cuda", "0") or "0"
        if imgsz is None:
            imgsz = 640 if dev == "cpu" else 960            # 원본 기본 960. 노트북 CPU 에서는 640 이어야 실시간에 가깝다
        self.log = log
        self.p = Pipeline(str(weights), device=dev, camera_mode="egocentric", imgsz=imgsz, crop=crop)
        self.imgsz, self.crop = imgsz, crop
        self.path = Path(weights)
        self._origin = time.perf_counter()
        self._last_t = None
        self._pending: list[dict] = []
        self._err_t = -1e9
        self._lock = threading.Lock()
        self.info = {"imgsz": imgsz, "crop": crop}

    def set_cam_h(self, h: float):
        """보행자 처음 설정(키 × 카메라 위치)으로 카메라 높이를 바꾼다. 원본 ego_motion 은 고치지 않고 모듈 값만 바꾼다."""
        if not 0.5 <= h <= 2.2:
            return
        em = sys.modules.get("ego_motion")
        if em is not None:
            em.CAM_H = float(h)                                   # 보행 속도 · 경로 추정
        # 위험 판정의 거리 · 통로 폭은 HazardConfig.camera_height_m 을 따로 쓴다 → 같이 바꾼다 (설정 객체만 교체)
        from dataclasses import replace
        self.p.config = replace(self.p.config, camera_height_m=float(h))
        self.p.core.policy.config = replace(self.p.core.policy.config, camera_height_m=float(h))
        self.info["cam_h"] = round(float(h), 2)

    # ── 인식기 쪽 ──
    def warmup(self, image, n=2):
        with self._lock:
            for i in range(n):
                self.p.step(np.zeros_like(image), i + 1, i * 0.1)
            self.p.reset()
            self._last_t = None

    def __call__(self, fidx, image):
        with self._lock:
            t = time.perf_counter() - self._origin
            if self._last_t is not None and t <= self._last_t:
                t = self._last_t + 1e-3
            self._last_t = t
            try:
                events, debug = self.p.step(image, fidx, t)
            except Exception:
                if time.time() - self._err_t > 5.0:
                    self._err_t = time.time()
                    self.log("[nav] 오류\n" + traceback.format_exc())
                self._pending = []
                return []
            self._pending = events
            self.info = {"imgsz": self.imgsz, "crop": self.crop,
                         "ms": round(debug.get("processing_s", 0) * 1000, 1),
                         "walk": round(float(debug.get("ego", {}).get("walk_mps") or 0), 2),
                         "cam_ok": bool(debug.get("camera", {}).get("camera_ok"))}
            out = []
            for d in debug.get("detections", []):
                if d.get("veto"):
                    continue
                cls = NAME_FIX.get(d.get("class"), d.get("class"))
                b = tuple(round(float(v), 1) for v in d["box"])
                out.append(Det(int(d.get("track_id", -1)), cls, round(float(d["confidence"]), 3), b,
                               ((b[0] + b[2]) / 2, b[3]), 0, 0.0))
            return out

    # ── 트리거 쪽 ──
    def step(self, fr):
        events, self._pending = self._pending, []
        out = []
        for e in events:
            box = tuple(float(v) for v in e["box"]) if e.get("box") is not None else None
            cls = NAME_FIX.get(e.get("class"), e.get("class") or "obstacle")
            out.append(Event(fr.t, cls, _scenario(e), e.get("track_id"), box, fr.W, fr.H,
                             extra=dict(text=e["message"], prio=_prio(e),
                                        level="danger" if e.get("priority") == "urgent" else "warn",
                                        motion=e.get("motion_state"), rule=e.get("rule_group") or e["event"],
                                        state=("GO" if e.get("color") == "green" else "STOP") if e.get("color") else None)))
        return out

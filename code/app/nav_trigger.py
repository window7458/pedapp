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

# ── 폰 센서(자이로 · 가속도) 넣기 ─────────────────────────────────────────
# 태규 코드는 고치지 않는다. 그 코드의 두 함수(EgoMotion.step, SceneContext.step)가 돌려주는 값 위에
# 센서로 구한 값을 덮어쓰는 포장만 씌운다. 이번 프레임의 센서 추정은 _IMU_NOW 에 둔다(없으면 원래 값 그대로).
#   · 지평선: 중력 기울기로 계산 (원래는 영상 확장 중심 → 화면 20~70% 로 제한) → 거리 · 보행 통로가 정확해진다
#   · 카메라 흔들림: 직전 프레임 뒤 12° 넘게 돌았거나 순간 150°/s 넘으면 흔들림으로 (원래 판정에 더하기만)
#   · 보행 속도 · 멈춤: 걸음 빈도 × 보폭으로 (원래는 바닥 광류, 실패하면 1.2 m/s 가정)
_IMU_NOW: dict | None = None
# 여러 보행자가 서버 하나를 같이 쓸 때: 태규 코드의 모듈 값(ego_motion.CAM_H)과 _IMU_NOW 는 하나뿐이라,
# 한 사람의 프레임을 처리하는 동안은 다른 사람이 끼어들지 못하게 한 번에 한 명씩 돌린다 (GPU 추론도 어차피 하나씩).
_STEP_LOCK = threading.Lock()
SHOCK_DEG, SHOCK_DPS = 12.0, 150.0


def _apply_imu(d: dict, w: int, h: int, ctx: bool):
    est = _IMU_NOW
    if not est:
        return d
    from imu_fusion import horizon_from
    hz = horizon_from(est, h, d.get("focal_px") or 26 / 36 * max(w, h))
    if hz is not None:
        d["horizon"] = hz
    if not ctx and (est.get("rot_deg", 0) > SHOCK_DEG or est.get("rate_peak_dps", 0) > SHOCK_DPS):
        d["camera_shock"] = True
    if est.get("walking"):
        d["walk_mps"], d["walk_valid"] = float(est["walk_mps"]), True
        if ctx:
            d["stationary"] = False
    elif est.get("stationary"):
        d["walk_mps"], d["walk_valid"] = 0.0, True
        if ctx:
            d["stationary"] = True
    d["imu"] = True
    return d


def _install_imu_hooks():
    import ego_motion
    import scene_context
    if getattr(ego_motion.EgoMotion, "_imu_hooked", False):
        return
    ego_step, scene_step = ego_motion.EgoMotion.step, scene_context.SceneContext.step

    def ego_with_imu(self, frame, pairs):
        return _apply_imu(ego_step(self, frame, pairs), frame["width"], frame["height"], ctx=False)

    def scene_with_imu(self, f, new=None, backflow=None):
        return _apply_imu(scene_step(self, f, new, backflow), f["width"], f["height"], ctx=True)

    ego_motion.EgoMotion.step = ego_with_imu
    scene_context.SceneContext.step = scene_with_imu
    ego_motion.EgoMotion._imu_hooked = True


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
        _install_imu_hooks()
        self._cap_ms = None          # 이번 프레임을 폰이 찍은 시각(폰 시계 ms). 없으면 서버 시계
        self._cap0 = None
        self.imu = None              # 이 보행자 폰의 센서 (서버가 사람마다 넣어 줌). 없으면 공용
        self._cam_h = None           # 이 보행자의 카메라 높이 (처음 설정)
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
        self._cam_h = float(h)                                    # ego_motion.CAM_H 는 처리할 때마다 이 사람 값으로 (아래 __call__)
        # 위험 판정의 거리 · 통로 폭은 HazardConfig.camera_height_m 을 따로 쓴다 → 같이 바꾼다 (설정 객체만 교체)
        from dataclasses import replace
        self.p.config = replace(self.p.config, camera_height_m=float(h))
        self.p.core.policy.config = replace(self.p.core.policy.config, camera_height_m=float(h))
        self.info["cam_h"] = round(float(h), 2)

    def set_capture(self, cap_ms):
        """pipeline 이 프레임마다 부른다: 폰이 붙여 보낸 찍은 시각."""
        self._cap_ms = float(cap_ms) if cap_ms is not None else None

    # ── 인식기 쪽 ──
    def warmup(self, image, n=2):
        with _STEP_LOCK:
            for i in range(n):
                self.p.step(np.zeros_like(image), i + 1, i * 0.1)
            self.p.reset()
            self._last_t = None

    def __call__(self, fidx, image):
        with _STEP_LOCK:
            global _IMU_NOW
            from imu_fusion import IMU
            em = sys.modules.get("ego_motion")
            if em is not None:
                em.CAM_H = self._cam_h or 1.70                    # 보행 속도 · 경로 추정: 이 사람의 카메라 높이
            cap = self._cap_ms
            self._cap_ms = None
            if cap is not None:
                # 폰이 찍은 시각으로 시간을 잰다 → 네트워크가 출렁여도 물체 크기 증가율 · 속도가 실제 시간 간격으로 계산된다
                if self._cap0 is None or (self._last_t is not None and cap - self._cap0 < self._last_t * 1000 - 5000):
                    self._cap0 = cap - (self._last_t or 0) * 1000
                t = (cap - self._cap0) / 1000
            else:
                t = time.perf_counter() - self._origin
            _IMU_NOW = (self.imu or IMU).estimate(cap)              # 이 사람 폰의 센서
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
            if _IMU_NOW:
                self.info.update(imu=True, pitch=round(np.degrees(_IMU_NOW.get("pitch_rad", 0)), 1),
                                 step_hz=_IMU_NOW.get("step_hz"), walking=_IMU_NOW.get("walking"),
                                 horizon=round(float(debug.get("context", {}).get("horizon") or 0), 1))
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

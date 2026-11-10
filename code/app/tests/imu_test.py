"""폰 센서 → 트리거: 기울기 → 지평선, 자이로 → 흔들림, 걸음 → 속도 · 멈춤, 프레임 시각 헤더. 태규 코드의 EgoMotion · SceneContext 결과가 바뀌는지까지."""
import math, os, struct, sys
from pathlib import Path
import numpy as np
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
import pipeline  # noqa: F401  (core 경로)
from imu_fusion import ImuBuffer, horizon_from
from sources import split_stamp

def rows(t0, dur_s, hz=60, pitch_deg=20, step_hz=0.0, amp=0.0, rot_dps=0.0, noise=0.03, ios=False):
    out = []; rng = np.random.default_rng(1)
    for i in range(int(dur_s * hz)):
        t = t0 + i * 1000 / hz; p = math.radians(pitch_deg)
        g = np.array([0, 9.81 * math.cos(p), 9.81 * math.sin(p)])
        lin = (g / 9.81) * amp * math.sin(2 * math.pi * step_hz * i / hz) + rng.normal(0, noise, 3)   # 걸음은 위아래(중력 방향)로 출렁인다
        gg = (g + lin) * (-1 if ios else 1)
        out.append([t, *lin.tolist(), *gg.tolist(), rot_dps, 0, 0])
    return out

# 1) 기울기 20° 아래 → 지평선이 화면 가운데보다 위
b = ImuBuffer(); b.set_height(170); b.add(rows(0, 3, pitch_deg=20))
e = b.estimate(2900); assert abs(math.degrees(e["pitch_rad"]) - 20) < 1.5 and e["tilt_ok"], e
H, F = 960, 26 / 36 * 960
hz = horizon_from(e, H, F); assert abs(hz - (H / 2 - F * math.tan(math.radians(20)))) < 15, hz
# 아이폰처럼 부호가 반대여도 같은 값
b2 = ImuBuffer(); b2.add(rows(0, 3, pitch_deg=20, ios=True)); assert abs(math.degrees(b2.estimate(2900)["pitch_rad"]) - 20) < 1.5
# 2) 멈춤 · 걸음
assert e["stationary"] and not e["walking"], e
b3 = ImuBuffer(); b3.set_height(170); b3.add(rows(0, 3.5, step_hz=1.8, amp=1.6))
w = b3.estimate(3400); assert w["walking"] and 1.4 <= w["step_hz"] <= 2.2 and 0.9 <= w["walk_mps"] <= 1.6, w
# 3) 크게 돌기 → 흔들림
b4 = ImuBuffer(); b4.add(rows(0, 2, rot_dps=200)); b4.estimate(1500); r = b4.estimate(1900); assert r["rot_deg"] > 12, r
# 4) 오래된 센서만 있으면 안 넣음
assert b.estimate(2900 + 1000) is None
# 5) 프레임 헤더
cap, jpg = split_stamp(b"SEF1" + struct.pack("!d", 1234.5) + b"\xff\xd8xx"); assert cap == 1234.5 and jpg == b"\xff\xd8xx"
assert split_stamp(b"\xff\xd8yy") == (None, b"\xff\xd8yy")

# 6) 태규 코드의 결과 위에 덮어쓰기 (코드는 그대로)
import nav_trigger
nav_trigger._install_imu_hooks()
import ego_motion, scene_context
img_w, img_h = 540, 960
frame = dict(time_s=0.1, frame=1, width=img_w, height=img_h, matrix=None, detections=[])
nav_trigger._IMU_NOW = None
base = ego_motion.EgoMotion().step(dict(frame), np.zeros((0, 4), np.float32))
nav_trigger._IMU_NOW = dict(e, walking=True, walk_mps=1.3, stationary=False, rot_deg=20, rate_peak_dps=30)
got = ego_motion.EgoMotion().step(dict(frame), np.zeros((0, 4), np.float32))
assert abs(got["horizon"] - horizon_from(e, img_h, got["focal_px"])) < 1e-6 and got["camera_shock"] and got["walk_mps"] == 1.3, got
f2 = dict(frame, ego=got, direction=dict(valid=True, heading_rad=0.0, vertical_slope=0.3, stationary=True), image_quality=dict(valid=True))
ctx = scene_context.SceneContext().step(f2, None, None)
assert abs(ctx["horizon"] - got["horizon"]) < 1e-6 and ctx["stationary"] is False and ctx["imu"], ctx   # 영상 추정(0.3 기울기 · 멈춤) 대신 센서 값
nav_trigger._IMU_NOW = None
ctx0 = scene_context.SceneContext().step(dict(f2, ego=base), None, None)
assert "imu" not in ctx0 and ctx0["stationary"] is True                                          # 센서 없으면 원래 그대로
print("센서 연결 테스트 통과", round(base["horizon"]), "→", round(got["horizon"]), "px (20° 아래)")

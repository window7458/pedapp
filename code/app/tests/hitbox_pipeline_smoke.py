"""LivePipeline + HitboxTrigger 를 서버 없이 한 프로세스에서 돌려 본다 (가짜 인식 · 가짜 영상). 모델 · 네트워크 불필요.
    python tests/hitbox_pipeline_smoke.py
오른쪽에서 커지는 가짜 자동차 + 앞으로 걸어가는 것처럼 확대되는 배경 → 'warn' 메시지에 한국어 문장이 나와야 한다.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["HB_LIGHTS"] = "0"
os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from pipeline import Config, LivePipeline  # noqa: E402

msgs = []
cfg = Config(perception="demo", trigger="hitbox", vlm="none", scene="mock", scene_interval=1.0, max_fps=30.0)
pipe = LivePipeline(cfg, msgs.append, log=print)
rng = np.random.default_rng(1)
base = cv2.GaussianBlur(rng.integers(0, 255, (960, 540, 3), dtype=np.uint8), (0, 0), 3)
t0 = time.perf_counter()
for i in range(260):
    s = 1.0 + 0.004 * (i % 60)
    M = np.array([[s, 0, (1 - s) * 270], [0, s, (1 - s) * 480]], np.float32)
    pipe.submit(cv2.warpAffine(base, M, (540, 960), borderMode=cv2.BORDER_REFLECT))
    time.sleep(1 / 15)
time.sleep(1.5)
pipe.stop()
warns = [m for m in msgs if m["type"] == "warn"]
st = [m for m in msgs if m["type"] == "status"][-1]
print(f"{time.perf_counter() - t0:.1f}s · 메시지 {len(msgs)}개 · warn {len(warns)}개 · scene {sum(m['type'] == 'scene' for m in msgs)}개")
for w in warns:
    print(f"  p{w['prio']} {w['level']} [{w['scenario']}] {w['text']}")
print("status:", {k: st[k] for k in ("fps", "perception", "trigger", "hb")})
assert st["trigger"] == "hitbox_v5_29" and st["hb"], "hitbox 트리거가 안 돌고 있음"

"""폰 센서 + 파이 카메라 경로 확인 (포트 없이, 가짜 인식 · 가짜 트리거).
파이 역할: SEF1 프레임 전송 → frame_done 수신. 폰 역할: sensors 묶음 전송 → CSV 저장 · 접속 키."""
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient  # noqa: E402

from pipeline import Config  # noqa: E402
from server import build_app  # noqa: E402

tmp = Path(tempfile.mkdtemp())
from fakes import FakePerception  # noqa: E402
cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
rng = np.random.default_rng(2)
base = cv2.GaussianBlur(rng.integers(0, 255, (640, 360, 3), dtype=np.uint8), (0, 0), 3)
app = build_app(cfg, None, "111111", alerts_dir=tmp / "alerts", session_dir=tmp / "s", key="abc")

with TestClient(app, base_url="https://testserver") as c:
    assert c.get("/").status_code == 401, "키 없이 열리면 안 됨"
    assert c.get("/preview.mjpg").status_code == 401
    r = c.get("/?k=abc")
    assert r.status_code == 200 and "sensors.js" in r.text
    assert c.get("/sensors.js").status_code == 200, "쿠키로 기억"
    with c.websocket_connect("/ws?role=pedestrian&k=abc") as phone:
        assert json.loads(phone.receive_text())["type"] == "hello"
        now = time.time() * 1000
        phone.send_text(json.dumps({"type": "sensors", "now": now, "meta": {"ua": "t"},
                                    "motion": [[now + i * 10, .1, .2, .3, .1, -9.6, .3, 10, 20, 30] for i in range(100)],
                                    "orient": [[now, 10, 85, 2, 123.0]],
                                    "gps": [[now, 37.45, 126.65, 30, 5, 3, 1.1, 90]]}))
        with c.websocket_connect("/ws?role=camera&rotate=90&k=abc&pin=111111") as pi:
            h = json.loads(pi.receive_text()); assert h.get("pi") == 1, h
            phone.send_text(json.dumps({"type": "set_source", "mode": "pi"}))     # 새 화면: 설정에서 파이 카메라를 고른다
            time.sleep(0.3)
            done = []
            for i in range(25):
                ok, b = cv2.imencode(".jpg", np.roll(base, i * 2, 1))
                time.sleep(0.05)      # 파이프라인 속도 제한(max_fps)에 안 걸리게
                cap = time.time() * 1000
                pi.send_bytes(b"SEF1" + struct.pack("!d", cap) + b.tobytes())
                m = json.loads(pi.receive_text())
                assert m["type"] == "frame_done", m
                assert m["c"] == cap, "최신 프레임 짝"
                done.append(time.time() * 1000 - cap)
            print("frame_done", len(done), "median ms", round(sorted(done)[len(done) // 2], 1))
        for _ in range(200):
            t = json.loads(phone.receive_text())
            assert t["type"] != "frame_done", "폰에는 안 감"
            if t["type"] == "camera" and t["connected"] == 0:
                break
        s = c.get("/api/sensors").json()
        print("summary", s["session"], s["latest"], s["total"])
        assert s["latest"]["gps"]["lat"] == 37.45 and s["latest"]["heading"] == 123.0
    c.cookies.clear()
    with c.websocket_connect("/ws?k=wrong") as bad:
        assert json.loads(bad.receive_text())["type"] == "auth_error"
d = next((tmp / "s").iterdir())
for f in ("Gyroscope", "Accelerometer", "Gravity", "Orientation", "Compass", "Location", "Frames", "clock"):
    assert (d / f"{f}.csv").exists(), f
rows = (d / "Gyroscope.csv").read_text().splitlines()
assert len(rows) >= 101 and rows[0] == "time,seconds_elapsed,z,y,x", rows[0]
assert abs(float(rows[1].split(",")[2]) - np.radians(10)) < 1e-6
assert len((d / "Frames.csv").read_text().splitlines()) == 26
print("OK", d.name)

"""폰 센서 + 폰 프레임(찍은 시각 헤더) 경로 확인 (포트 없이, 가짜 인식 · 가짜 트리거).
폰: sensors 묶음 → CSV 저장, 'SEF1'+시각+JPEG 프레임 → 처리 · Frames.csv 에 찍은 시각 기록, 접속 키."""
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
        for _ in range(100):                                         # 판단 장치가 뒤에서 준비될 때까지
            if json.loads(phone.receive_text())["type"] == "ready": break
        now = time.time() * 1000
        phone.send_text(json.dumps({"type": "sensors", "now": now, "meta": {"ua": "t"},
                                    "motion": [[now + i * 10, .1, .2, .3, .1, -9.6, .3, 10, 20, 30] for i in range(100)],
                                    "orient": [[now, 10, 85, 2, 123.0]],
                                    "gps": [[now, 37.45, 126.65, 30, 5, 3, 1.1, 90]]}))
        phone.send_text(json.dumps({"type": "set_source", "mode": "phone"}))
        caps = []
        for i in range(25):
            ok, b = cv2.imencode(".jpg", np.roll(base, i * 2, 1))
            cap = time.time() * 1000
            caps.append(cap)
            phone.send_bytes(b"SEF1" + struct.pack("!d", cap) + b.tobytes())
            time.sleep(0.05)
        got = 0
        for _ in range(400):
            t = json.loads(phone.receive_text())
            assert t["type"] != "frame_done"
            got += t["type"] == "dets"
            if got >= 5:
                break
        assert got >= 5, "폰 프레임이 처리되지 않음"
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
fr = (d / "Frames.csv").read_text().splitlines()
assert len(fr) == 26 and all(r.split(",")[1] == "phone" and r.split(",")[2] for r in fr[1:]), fr[:3]   # 찍은 시각이 기록됨
print("OK", d.name)

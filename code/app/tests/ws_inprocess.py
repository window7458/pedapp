"""서버(/ws) 배선을 포트를 열지 않고 확인한다 (FastAPI TestClient, 가짜 인식 + hitbox + 가짜 VLM/장면)."""
import os
import sys
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

cfg = Config(perception="demo", trigger="hitbox", vlm="none", scene="mock", scene_interval=1.0, max_fps=30.0)
rng = np.random.default_rng(2)
base = cv2.GaussianBlur(rng.integers(0, 255, (640, 360, 3), dtype=np.uint8), (0, 0), 3)
got = {}
with TestClient(build_app(cfg, None)) as c:
    print("health:", c.get("/api/health").json())
    assert "Insight" in c.get("/").text or "<html" in c.get("/").text.lower()
    with c.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        print("hello:", {k: hello[k] for k in ("type", "perception", "trigger", "scene")})
        ws.send_json({"type": "set_source", "mode": "phone"})
        t0 = time.time()
        i = 0
        while time.time() - t0 < 14:
            s = 1.0 + 0.004 * (i % 60)
            M = np.array([[s, 0, (1 - s) * 180], [0, s, (1 - s) * 320]], np.float32)
            ok, buf = cv2.imencode(".jpg", cv2.warpAffine(base, M, (360, 640), borderMode=cv2.BORDER_REFLECT))
            ws.send_bytes(buf.tobytes())
            i += 1
            time.sleep(1 / 12)
        # 쌓인 메시지 읽기
        ws.send_json({"type": "set_scene", "on": True})
        t1 = time.time()
        while time.time() - t1 < 3:
            try:
                m = ws.receive_json()
            except Exception:
                break
            got[m["type"]] = got.get(m["type"], 0) + 1
            if m["type"] == "warn":
                print("warn:", m["text"], "prio", m.get("prio"))
            if sum(got.values()) > 4000:
                break
print("받은 메시지 종류:", got)
assert got.get("dets") and got.get("status"), "dets/status 가 안 옴"

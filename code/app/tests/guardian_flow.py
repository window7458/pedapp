"""보호자 흐름 확인 (포트 없이 TestClient): 확인번호 · 위험 경고→알림(사진) · 묶기 · SOS · 끊김/복구 · 역할별 메시지."""
import os, sys, tempfile, time
from pathlib import Path
import cv2, numpy as np
os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from pipeline import Config
from server import build_app
import alerts as A

# 1) 분류 규칙
assert A.classify({"prio": 3}) == "urgent"
assert A.classify({"prio": 1, "level": "danger", "motion": "approaching"}) == "urgent"
assert A.classify({"prio": 2}) == "caution" and A.classify({"prio": 1, "level": "danger", "motion": "static"}) == "caution"
assert A.classify({"prio": 1, "level": "warn"}) == "info"

cfg = Config(perception="demo", trigger="hitbox", vlm="none", scene="none", max_fps=30.0)
rng = np.random.default_rng(2)
base = cv2.GaussianBlur(rng.integers(0, 255, (640, 360, 3), dtype=np.uint8), (0, 0), 3)
def frame(i):
    s = 1.0 + 0.004 * (i % 60); M = np.array([[s,0,(1-s)*180],[0,s,(1-s)*320]], np.float32)
    ok, b = cv2.imencode(".jpg", cv2.warpAffine(base, M, (360, 640), borderMode=cv2.BORDER_REFLECT)); return b.tobytes()

with tempfile.TemporaryDirectory() as d, TestClient(build_app(cfg, None, pin="424242", alerts_dir=Path(d))) as c:
    assert c.get("/api/alerts").status_code == 401 and c.get("/api/alerts?pin=000000").status_code == 401
    for path in ("/", "/guardian", "/dev", "/manifest-guardian.webmanifest"):
        assert c.get(path).status_code == 200, path
    with c.websocket_connect("/ws?role=guardian&pin=999999") as g:           # 틀린 번호
        assert g.receive_json()["type"] == "auth_error"
    with c.websocket_connect("/ws?role=guardian&pin=424242") as g, c.websocket_connect("/ws") as p:
        p.receive_json(); g.receive_json()
        p.send_json({"type": "set_source", "mode": "phone"})
        t0 = time.time(); i = 0
        while time.time() - t0 < 14:                                       # 보행자: 영상 전송
            p.send_bytes(frame(i)); i += 1; time.sleep(1/12)
        p.send_json({"type": "sos"}); p.send_json({"type": "client_event", "kind": "battery", "pct": 15})
        time.sleep(1)
        got_g, got_p = {}, {}
        for _ in range(3000):                                              # 보호자가 받은 것
            try: m = g.receive_json()
            except Exception: break
            got_g[m["type"]] = got_g.get(m["type"], 0) + 1
            if sum(got_g.values()) > 600 or got_g.get("alert", 0) >= 4 and got_g.get("status", 0) > 3: break
        j = c.get("/api/alerts?pin=424242").json()
        lv = [a["level"] for a in j["alerts"]]
        print("보호자 수신:", got_g); print("알림 내역:", [(a["level"], a["text"], a["count"], a["snap"]) for a in j["alerts"]])
        assert "dets" not in got_g and "warn" not in got_g, "보호자에게 박스/경고 음성 메시지가 가면 안 됨"
        assert "sos" in lv and "battery" in lv and "online" in lv and ("urgent" in lv or "caution" in lv or "info" in lv)
        sosa = next(a for a in j["alerts"] if a["level"] == "sos"); assert sosa["snap"]
        r = c.get(f"/api/alerts/{sosa['id']}.jpg?pin=424242"); assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
        assert c.get("/api/latest.jpg?pin=424242").status_code == 200
        assert c.get(f"/api/alerts/{sosa['id']}.jpg").status_code == 401
        # 영상이 멈춘 뒤 6초 → 끊김 알림
        time.sleep(9)
        lv2 = [a["level"] for a in c.get("/api/alerts?pin=424242").json()["alerts"]]
        print("끊김 후:", lv2); assert "offline" in lv2
print("보호자 흐름 통과")

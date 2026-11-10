"""여러 보행자가 서버 하나를 같이 쓸 때 서로 섞이지 않는지:
판단 장치 · 확인번호 · 처음 설정 · 보호자 연결 · 알림 · 위치 · 메시지가 사람마다 따로."""
import json, os, sys, tempfile, time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
os.environ["DATA_DIR"] = tempfile.mkdtemp(); os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
import cv2, numpy as np
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app

def until(ws, typ, n=400):
    for _ in range(n):
        m = ws.receive_json()
        if m["type"] == typ: return m
    raise AssertionError(typ)

def drain(ws, sec=0.6):
    """sec 동안 받은 종류 모음 (보행자 화면은 status 가 1초마다 와서 receive 가 막히지 않는다)"""
    seen, t0 = [], time.time()
    while time.time() - t0 < sec:
        seen.append(ws.receive_json()["type"])
    return seen

cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
A, B = "userAAAA1111", "userBBBB2222"
app = build_app(cfg, None, pin="424242")
with TestClient(app) as c, c.websocket_connect(f"/ws?u={A}") as pa, c.websocket_connect(f"/ws?u={B}") as pb:
    ha, hb = pa.receive_json(), pb.receive_json()
    until(pa, "ready"); until(pb, "ready")          # 판단 장치는 뒤에서 준비된다
    assert ha["pin"] != hb["pin"], "확인번호가 사람마다 달라야"
    users = app.state.users
    assert users[A].pipe is not None and users[B].pipe is not None and users[A].pipe is not users[B].pipe
    # 처음 설정: A 는 어린이, B 는 어른
    pa.send_json({"type": "profile", "age": 10, "height_cm": 140, "mount": "neck"})
    pb.send_json({"type": "profile", "age": 30, "height_cm": 180, "mount": "eye"})
    time.sleep(0.4)
    assert users[A].profile.child() and not users[B].profile.child()
    assert users[A].alerts.sharing() and not users[B].alerts.sharing()
    # 보호자: A 하고만 연결
    pa.send_json({"type": "pair_new"}); m = until(pa, "pair_link")
    tok = c.post(f"/api/pair?code={parse_qs(urlparse(m['url']).query)['pair'][0]}").json()["token"]
    with c.websocket_connect(f"/ws?role=guardian&pin={tok}") as ga:
        assert ga.receive_json()["profile"]["age"] == 10
        # B 의 안부 · 위치는 A 의 보호자에게 안 감, A 의 것은 감
        pb.send_json({"type": "client_event", "kind": "ping"}); pb.send_json({"type": "gps", "lat": 1.0, "lon": 2.0})
        pa.send_json({"type": "client_event", "kind": "ping"}); pa.send_json({"type": "gps", "lat": 37.5, "lon": 127.0})
        a = until(ga, "alert"); assert a["level"] == "notice"
        loc = c.get(f"/api/location?pin={tok}").json()["loc"]; assert loc["lat"] == 37.5, loc   # A(어린이) 위치만
        assert len(users[B].alerts.list()) == 1 and len(users[A].alerts.list()) >= 1
        # 보호자 메시지는 A 에게만
        ga.send_json({"type": "g_msg", "text": "천천히 와"})
        assert "g_msg" in drain(pa, 1.0)
        assert "g_msg" not in drain(pb, 1.0)
    # 프레임: A 가 보낸 영상의 검출은 A 에게만
    ok, b = cv2.imencode(".jpg", np.full((320, 240, 3), 90, np.uint8))
    pa.send_json({"type": "set_source", "mode": "phone"}); pb.send_json({"type": "set_source", "mode": "phone"})
    for _ in range(8):
        pa.send_bytes(b.tobytes()); time.sleep(0.06)
    seen_a, seen_b = drain(pa, 1.0), drain(pb, 1.0)
    assert "dets" in seen_a and "dets" not in seen_b, (seen_a.count("dets"), seen_b.count("dets"))
    # 확인번호(보조)도 각자
    assert c.get(f"/api/alerts?pin={hb['pin']}").json()["alerts"][0]["cls"] == "ping"
    assert c.get(f"/api/alerts?pin={ha['pin']}").status_code == 200
print("여러 사용자 분리 테스트 통과")

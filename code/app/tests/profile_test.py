"""처음 설정: 키 × 카메라 위치 → 카메라 높이. 만 13세 미만 → 보호자가 항상 봄, 보행자 쪽에선 못 풀고 보호자만 풂. 재시작해도 남음."""
import os, sys, tempfile, time
from pathlib import Path
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d
os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app, Profile

assert Profile(Path(d) / "x.json").cam_h() is None
p = Profile(Path(d) / "x.json"); p.update({"age": 30, "height_cm": 190, "mount": "eye"}); assert p.cam_h() == 1.77
p.update({"height_cm": 155, "mount": "chest"}); assert p.cam_h() == 1.12          # 예전 실측값과 같음

def until(ws, typ, n=300):
    for _ in range(n):
        m = ws.receive_json()
        if m["type"] == typ: return m
    raise AssertionError(typ)

def both(ws, *types):                       # 순서와 상관없이 여러 종류를 받는다
    got = {}
    while len(got) < len(types):
        m = ws.receive_json()
        if m["type"] in types: got.setdefault(m["type"], m)
    return [got[t] for t in types]

cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
P = "pin=424242"
with TestClient(build_app(cfg, None, pin="424242", alerts_dir=Path(d) / "alerts")) as c:
    with c.websocket_connect("/ws?role=guardian&" + P) as g, c.websocket_connect("/ws") as ped:
        assert g.receive_json()["profile"]["set"] is False; ped.receive_json()
        ped.send_json({"type": "profile", "age": 10, "height_cm": 140, "mount": "neck"})
        pr = s = None
        for _ in range(400):                        # 보행자가 연결될 때 설정을 한 번 더 알리므로, 바뀐 값이 올 때까지 읽는다
            m = g.receive_json()
            if m["type"] == "profile" and m.get("child"): pr = m
            if m["type"] == "share" and m.get("on"): s = m
            if pr and s: break
        assert pr["child"] and pr["cam_h"] == 1.15 and s["on"] and s["reason"] == "child"
        ped.send_json({"type": "gps", "lat": 37.1, "lon": 127.1, "acc": 5}); time.sleep(0.3)
        assert c.get("/api/location?" + P).json()["loc"]["lat"] == 37.1          # 항상 위치 보임
        ped.send_json({"type": "share_stop"}); time.sleep(0.2)
        assert c.get("/api/location?" + P).json()["loc"] is not None              # 보행자가 못 끔
        ped.send_json({"type": "profile", "age": 20})
        assert "보호자" in until(ped, "profile_denied")["why"]                     # 보행자 쪽에선 못 풂
with TestClient(build_app(cfg, None, pin="424242", alerts_dir=Path(d) / "alerts")) as c:   # 재시작
    assert c.get("/api/location?" + P).json()["share"]["reason"] == "child"
    with c.websocket_connect("/ws?role=guardian&" + P) as g:
        g.receive_json()
        g.send_json({"type": "profile", "age": 14})
        pr, s = both(g, "profile", "share")
        assert pr["child"] is False and s["on"] is False
    assert c.get("/api/location?" + P).json()["loc"] is None
print("처음 설정 테스트 통과")

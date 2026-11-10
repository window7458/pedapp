"""보호자 연결: 보행자 앱이 QR(1회용 링크)을 만들고 → 보호자가 열면 토큰을 받아 계속 연결 → 다시 쓰면 거절 → 연결 끊기.
확인번호(보조)는 그대로 통하고, 보호자 연결에는 번호를 보내지 않는다. 토큰은 재시작해도 남는다."""
import os, sys, tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d
os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app

cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
def app(): return build_app(cfg, None, pin="424242", alerts_dir=Path(d) / "alerts")

with TestClient(app()) as c:
    assert c.get("/api/alerts?pin=424242").status_code == 200              # 보조 확인번호
    with c.websocket_connect("/ws") as p:
        h = p.receive_json(); assert h["pin"] == "424242" and h["paired"] == 0
        p.send_json({"type": "pair_new"})
        while (m := p.receive_json())["type"] != "pair_link": pass
        assert m["qr"].lstrip().startswith("<?xml") or "<svg" in m["qr"]
        code = parse_qs(urlparse(m["url"]).query)["pair"][0]
        r = c.post(f"/api/pair?code={code}"); assert r.status_code == 200
        tok = r.json()["token"]
        while (m := p.receive_json())["type"] != "paired": pass              # 보행자 앱에 "연결됐어요"
    assert c.post(f"/api/pair?code={code}").status_code == 410             # 1회용
    assert c.post("/api/pair?code=nope").status_code == 410
    assert c.get(f"/api/alerts?pin={tok}").status_code == 200
    with c.websocket_connect(f"/ws?role=guardian&pin={tok}") as g:
        assert "pin" not in g.receive_json()

with TestClient(app()) as c:                                              # 서버 재시작 뒤에도 연결 유지
    assert c.get(f"/api/alerts?pin={tok}").status_code == 200
    with c.websocket_connect(f"/ws?role=guardian&pin={tok}") as g, c.websocket_connect("/ws") as p:
        g.receive_json(); p.receive_json()
        p.send_json({"type": "unpair_all"})
        types = []
        for _ in range(100):
            m = g.receive_json(); types.append(m["type"])
            if m["type"] == "auth_error": assert m.get("reason") == "unpaired"; break
        assert "auth_error" in types
    assert c.get(f"/api/alerts?pin={tok}").status_code == 401
    assert c.get("/api/alerts?pin=424242").status_code == 200
print("보호자 연결 통과")

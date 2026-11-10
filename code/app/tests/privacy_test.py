"""사생활: 평소엔 위치 · 영상이 보호자에게 안 간다 → 요청하면 보행자에게 묻고, 허락하면 10분 열림 → 그만하면 닫힘.
SOS · 넘어짐은 30분 자동 공유. 안부 알림 · 보호자 메시지 · 30일 보관."""
import os, sys, tempfile, time, json
from pathlib import Path
d = tempfile.mkdtemp(); os.environ["DATA_DIR"] = d
os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app
import alerts as A

def until(ws, typ, n=300):
    for _ in range(n):
        m = ws.receive_json()
        if m["type"] == typ: return m
    raise AssertionError(typ)

cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", vlm="none", scene="none", max_fps=30.0)
with TestClient(build_app(cfg, None, pin="424242", alerts_dir=Path(d) / "alerts")) as c:
    P = "pin=424242"
    with c.websocket_connect("/ws?role=guardian&" + P) as g, c.websocket_connect("/ws") as p:
        assert g.receive_json()["share"]["on"] is False; p.receive_json()
        p.send_json({"type": "gps", "lat": 37.45, "lon": 126.65, "acc": 5})
        time.sleep(0.3)
        assert c.get("/api/location?" + P).json()["loc"] is None                  # 평소: 위치 안 줌
        assert c.get("/api/live.mjpg?" + P).status_code == 403
        assert c.get("/api/latest.jpg?" + P).status_code == 403
        assert c.post("/api/share/request?" + P).json()["asked"]
        until(p, "share_req")                                                     # 보행자에게 묻는다
        p.send_json({"type": "share_allow"})
        s = until(g, "share"); assert s["on"] and s["reason"] == "allowed" and s["until"] - time.time() > 500
        assert c.get("/api/location?" + P).json()["loc"]["lat"] == 37.45
        p.send_json({"type": "share_stop"})
        assert until(g, "share")["on"] is False
        assert c.get("/api/location?" + P).json()["loc"] is None
        p.send_json({"type": "client_event", "kind": "ping"})                     # 안부 알림
        a = until(g, "alert"); assert a["level"] == "notice" and "loc" not in a
        g.send_json({"type": "g_msg", "text": "천천히 와  기다릴게"})              # 보호자 메시지
        assert until(p, "g_msg")["text"] == "천천히 와 기다릴게"
        until(g, "g_msg_ok")
        p.send_json({"type": "client_event", "kind": "fall"})                     # 넘어짐 → SOS + 30분 공유
        s = until(g, "share"); assert s["on"] and s["reason"] == "sos" and s["until"] - time.time() > 1700
        a = until(g, "alert"); assert a["level"] == "sos" and a["cls"] == "fall" and a["loc"]["lat"] == 37.45
        print("share/fall ok")

# 30일 보관
al = A.AlertCenter(Path(d) / "a2", lambda m: None, log=lambda *_: None)
x = al.add("info", "오래된 것", "x"); y = al.add("info", "새 것", "y", merge_key=("y",))
x["ts"] = x["last_ts"] = time.time() - 31 * 86400
(Path(d) / "a2" / "log.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in al.items.values()), encoding="utf-8")
al2 = A.AlertCenter(Path(d) / "a2", lambda m: None, log=lambda *_: None)
assert [v["text"] for v in al2.items.values()] == ["새 것"], al2.items
print("사생활 · 보관 테스트 통과")

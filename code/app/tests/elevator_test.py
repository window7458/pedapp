"""역 엘리베이터: 코드표 · 문장 만들기(네트워크 없이) + KRIC_KEY 가 있으면 /api/elevator 실제 호출."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_OFF", "1")
from subway import Subway, sentence, _norm, _place

assert _norm("인하대역[수인분당선]") == "인하대" and _norm("부평역 4번출구") == "부평"
s = Subway(key="x")
assert {r["LN_NM"] for r in s.stations("부평역")} >= {"1호선", "인천1호선"}
print(_place("(1F)4번출입구앞(B1)임대상가앞(B2)고객안내센터옆"), "/", _place("우측게이트옆인천행타는곳"))
els = [dict(exit="1,2", place="", outside=True), dict(exit="6", place="", outside=True), dict(exit=None, place="교대 방면 3-4", outside=False)]
t = sentence("강남", els); print(t)
assert "1번 2번 출구 사이" in t and "6번 출구" in t and "교대 방면 3-4" in t
if os.environ.get("KRIC_KEY"):
    from fastapi.testclient import TestClient
    from fakes import FakePerception
    from pipeline import Config
    from server import build_app
    with tempfile.TemporaryDirectory() as d:
        c = TestClient(build_app(Config(perception=FakePerception(), trigger="fakes:FakeTrigger"), None, pin="1", alerts_dir=Path(d)))
        r = c.get("/api/elevator", params={"q": "인하대역"}); assert r.status_code == 200, r.text; print(r.json()["sentence"])
        assert c.get("/api/elevator", params={"q": "없는역"}).status_code == 404
print("elevator ok")

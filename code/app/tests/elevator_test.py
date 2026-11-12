"""지하철(엘리베이터 · 시간표 · 승강장 가는 길): 코드표 · 문장(네트워크 없이) + KRIC_KEY 가 있으면 실제 호출."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_OFF", "1")
from subway import Subway, sentence, _norm, _place, _step, train_sentence, move_sentence

assert _norm("인하대역[수인분당선]") == "인하대" and _norm("부평역 4번출구") == "부평"
s = Subway(key="x")
assert {r["LN_NM"] for r in s.stations("부평역")} >= {"1호선", "인천1호선"}
print(_place("(1F)4번출입구앞(B1)임대상가앞(B2)고객안내센터옆"), "/", _place("우측게이트옆인천행타는곳"))
els = [dict(exit="1,2", place="", outside=True), dict(exit="6", place="", outside=True), dict(exit=None, place="교대 방면 3-4", outside=False)]
t = sentence("강남", els); print(t)
assert "1번 2번 출구 사이" in t and "6번 출구" in t and "교대 방면 3-4" in t
assert _step("3) (B1) 대합실로 이동") == "지하 1층 대합실로 이동" and _step("1/2번 출입구") == "1번 2번 출입구"
t = train_sentence("인하대", [dict(line="수인분당", next=[dict(min=4, to="송도"), dict(min=8, to="숭의"), dict(min=11, to="송도"), dict(min=3, to="")])])
print(t); assert "송도 쪽은 4분 뒤, 다음 차 11분 뒤" in t and "숭의 쪽은 8분 뒤" in t
t = move_sentence("인하대", [dict(start="1번 2번 출입구 엘리베이터", end="송도 방면", steps=["a"]), dict(start="6번 7번 출입구 엘리베이터", end="숭의 방면", steps=["b"])])
print(t); assert "말해 주세요" in t
if os.environ.get("KRIC_KEY"):
    from fastapi.testclient import TestClient
    from fakes import FakePerception
    from pipeline import Config
    from server import build_app
    with tempfile.TemporaryDirectory() as d:
        c = TestClient(build_app(Config(perception=FakePerception(), trigger="fakes:FakeTrigger"), None, pin="1", alerts_dir=Path(d)))
        r = c.get("/api/elevator", params={"q": "인하대역"}); assert r.status_code == 200, r.text; print(r.json()["sentence"])
        assert c.get("/api/elevator", params={"q": "없는역"}).status_code == 404
        r = c.get("/api/train", params={"q": "인하대"}); assert r.status_code == 200, r.text; print(r.json()["sentence"])
        r = c.get("/api/platform", params={"q": "인하대", "exit": "1", "dir": "송도"}); assert r.status_code == 200; print(r.json()["sentence"])
print("subway ok")

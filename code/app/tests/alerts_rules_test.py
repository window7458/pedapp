"""알림 규칙: 위험 구간 영상은 사람 · 탈것(+호출)만, 위치가 알림에 붙는지, 계단만 말하지 않는지, 푸시 단계 거르기."""
import sys, tempfile
from pathlib import Path
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
import pipeline  # noqa: F401  (sys.path)
import alerts as A
import hitbox_trigger as HT
from push import Push

with tempfile.TemporaryDirectory() as d:
    got = []
    c = A.AlertCenter(Path(d), lambda m: got.append(m))
    trig = []
    c.clips.ok = True
    c.clips.trigger = lambda aid: trig.append(aid)
    c.on_location({"lat": 37.45, "lon": 126.65, "acc": 9})
    a1 = c.add("caution", "정면 장애물이 있어요. 왼쪽으로 피하세요.", "obstacle")
    a2 = c.add("urgent", "정면 자동차가 다가와요. 멈추세요.", "car")
    a3 = c.add("caution", "1시 방향 킥보드가 다가와요. 피하세요.", "scooter")
    a4 = c.add("info", "신호등이 초록불로 바뀌었어요.", "traffic_light")
    assert trig == [a2["id"], a3["id"]], trig                      # 장애물 · 신호는 영상 없음
    assert a2.get("loc", {}).get("lat") == 37.45                    # 알림에 위치가 붙음
    assert any(m["type"] == "loc" for m in got)
assert HT.WARN_EXCLUDE == {"stairs"}, HT.WARN_EXCLUDE
assert Push.wanted("urgent", "urgent") and not Push.wanted("caution", "urgent")
assert Push.wanted("caution", "caution") and not Push.wanted("info", "caution") and Push.wanted("info", "all")
assert not Push.wanted("online", "all")
print("알림 규칙 테스트 통과")

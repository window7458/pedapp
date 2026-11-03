"""알림 규칙: 위험 구간 영상은 사람 · 탈것(+호출)만, 위치가 알림에 붙는지, 계단만 말하지 않는지, 푸시 단계 거르기."""
import sys, tempfile
from pathlib import Path
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
import pipeline  # noqa: F401  (sys.path)
import alerts as A
import hitbox_trigger as HT
from push import Push

def cv2_shape(f):
    import cv2
    return cv2.imread(str(f)).shape[:2]

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
# 장애물 경고만 원본 + 그 프레임의 검출을 남김. 묶인 알림은 첫 장면만. 보호자용 사진(snap)과 별개
import json, numpy as np
from types import SimpleNamespace as NS
with tempfile.TemporaryDirectory() as d:
    c = A.AlertCenter(Path(d), lambda m: None)
    im = np.zeros((960, 540, 3), np.uint8)
    det = NS(track_id=3, cls="motorcycle", conf=0.23, box=(1.0, 2.0, 3.0, 4.0))
    ev = NS(cls="obstacle", box=(1, 2, 3, 4), W=540, H=960, track_id=3)
    m = {"text": "정면 장애물이 있어요. 멈추세요.", "prio": 3}
    c.on_warn(m, ev, im, [det]); c.on_warn(m, ev, im, [det])
    c.on_warn({"text": "정면 자동차가 다가와요. 멈추세요.", "prio": 3}, NS(cls="car", box=None, W=540, H=960, track_id=4), im, [det])
    files = sorted(p.name for p in (Path(d) / "evidence").iterdir())
    assert files == ["1.jpg", "1.json"], files
    assert cv2_shape(Path(d) / "evidence" / "1.jpg") == (960, 540)    # 줄이지 않은 원본
    info = json.loads((Path(d) / "evidence" / "1.json").read_text(encoding="utf-8"))
    assert info["spoken_cls"] == "obstacle" and info["dets"][0]["cls"] == "motorcycle" and info["dets"][0]["conf"] == 0.23
    c.on_warn(dict(m, text="정면 긴 기둥이 있어요."), NS(cls="pole_long", box=None, W=0, H=0, track_id=5), None, None)   # 이미지가 없어도 죽지 않음
assert HT.WARN_EXCLUDE == {"stairs"}, HT.WARN_EXCLUDE
assert Push.wanted("urgent", "urgent") and not Push.wanted("caution", "urgent")
assert Push.wanted("caution", "caution") and not Push.wanted("info", "caution") and Push.wanted("info", "all")
assert not Push.wanted("online", "all")
print("알림 규칙 테스트 통과")

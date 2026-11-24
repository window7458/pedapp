"""실내 모드 오검출 거르기: 의자에 앉은 사람이 '오토바이'로 잡혀도 '사람'으로 바뀌고, 실내에 없을 차량은 버린다."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: F401  (경로 설정)
from nav_trigger import indoor_filter


def det(cls, box, conf=0.6):
    return {"class": cls, "box": box, "confidence": conf, "class_id": 0, "source": "custom"}


seated = [300, 200, 500, 520]
coco_person = [310, 190, 490, 515, 0.8, 0]
coco_chair = [320, 330, 520, 540, 0.7, 56]

# 1) 앉은 사람 → 팀 모델 '오토바이' → 버리고 COCO 사람을 '사람'으로
out = indoor_filter([det("motorcycle", seated)], [coco_person, coco_chair])
assert [d["class"] for d in out] == ["person"], out

# 2) 팀 모델이 이미 사람을 냈으면 사람을 두 번 넣지 않음
out = indoor_filter([det("motorcycle", seated), det("person", [305, 195, 495, 518])], [coco_person])
assert [d["class"] for d in out] == ["person"], out

# 3) 실내 자전거(사람 · 의자와 안 겹침)는 남김, 겹치면 버림
out = indoor_filter([det("bicycle", [20, 300, 200, 500])], [coco_person])
assert [d["class"] for d in out] == ["bicycle"], out
out = indoor_filter([det("scooter", seated)], [coco_chair])
assert out == [], out

# 4) COCO 없이(잘라 보기): 차량 종류만 거르고 나머지는 그대로
out = indoor_filter([det("car", seated), det("person", seated), det("obstacle", seated)], None)
assert [d["class"] for d in out] == ["person", "obstacle"], out
print("indoor filter ok")

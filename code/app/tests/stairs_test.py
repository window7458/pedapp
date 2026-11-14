"""계단 알림: 정면 가까이 계단이 연속으로 보이면 한 번. 지평선 아래 = 내려가는 계단(긴급), 위로 솟음 = 올라가는 계단, 옆은 무시."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: F401  (경로 설정)
from nav_trigger import StairsWatch

W, H, hz = 960, 540, 250
def run(box, n=8):
    sw = StairsWatch()
    return [o for o in (sw.update(i * 0.1, [{"class": "stairs", "confidence": 0.6, "box": box}], W, H, hz) for i in range(n)) if o]
d = run([380, 300, 600, 520]); assert len(d) == 1 and d[0]["event"] == "stairs_down" and d[0]["priority"] == "urgent"
u = run([380, 150, 600, 520]); assert len(u) == 1 and u[0]["event"] == "stairs_up"
assert run([20, 300, 150, 520]) == []                 # 옆쪽 계단은 말하지 않음
assert run([380, 300, 600, 520], n=3) == []           # 잠깐 보인 것은 말하지 않음 (4/6 프레임 필요)
print("stairs ok")

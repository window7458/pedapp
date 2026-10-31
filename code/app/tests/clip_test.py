"""위험 구간 녹화 확인: 평소엔 저장 안 함, 위험 때만 앞 5초+뒤 4초 mp4, 같은 위험이 이어지면 늘어남."""
import sys, tempfile, time, threading
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import alerts as A
A.CLIP_PRE_S, A.CLIP_POST_S = 1.0, 0.6          # 시험용으로 짧게
d = Path(tempfile.mkdtemp()); got = []
ac = A.AlertCenter(d, lambda m: got.append(m))
def feed(n, fill):
    for _ in range(n):
        ac.clips.push(np.full((360, 640, 3), fill, np.uint8)); time.sleep(0.05)
feed(30, 60)                                     # 평소 1.5초: 파일 없어야
assert not list((d / "clips").glob("*.mp4"))
a = ac.add("urgent", "정면 자동차가 다가와요. 멈추세요.", "car", np.zeros((360, 640, 3), np.uint8))
feed(8, 200); ac.add("urgent", "정면 자동차가 다가와요. 멈추세요.", "car")   # 이어짐 → 같은 알림에 묶이고 끝 늘어남
feed(30, 120)
for _ in range(60):
    if (d / "clips" / f"{a['id']}.mp4").exists(): break
    time.sleep(0.1)
p = d / "clips" / f"{a['id']}.mp4"; assert p.exists(), "클립 없음"
c = cv2.VideoCapture(str(p)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); print("프레임", n, "크기", p.stat().st_size)
assert n >= 10
assert any(m.get("clip_only") and m["clip"] for m in got)
assert ac.add("info", "위치", "pole") and not (d / "clips" / "2.mp4").exists()   # info 는 녹화 안 함
print("OK")

"""테스트 전용 가짜 인식기 (서버에서는 쓰지 않음): 오른쪽에서 자동차 하나가 계속 커진다 (5 s 주기)."""
import os; os.environ.setdefault("TTS_OFF", "1")   # 테스트는 클라우드 음성 API 를 부르지 않는다
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import pipeline  # noqa: E402,F401  (sys.path 설정)
from core import Det  # noqa: E402


class FakePerception:
    name = "fake"

    def __init__(self):
        self.k0 = None

    def __call__(self, fidx, image):
        H, W = image.shape[:2]
        if self.k0 is None:
            self.k0 = fidx
        k = (fidx - self.k0) % 75
        w = (0.10 + 0.004 * k) * W
        h = w * 0.7
        cx, cy = 0.72 * W, 0.62 * H
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        tid = 1 + (fidx - self.k0) // 75
        return [Det(int(tid), "car", 0.9, box, (cx, box[3]), 0, w * h)]


class FakeTrigger:
    """테스트 전용: 가짜 자동차 트랙마다 한 번, 화면 하단에 가까워지면 '위험' 경고를 낸다 (nav 트리거와 같은 Event 모양)."""
    name = "fake"
    self_deduped = True
    wants_vlm = True

    def __init__(self):
        self.fired = set()

    def step(self, fr):
        from core import Event
        out = []
        for d in fr.dets:
            if d.track_id not in self.fired and (d.box[3] - d.box[1]) > 0.12 * fr.H:
                self.fired.add(d.track_id)
                out.append(Event(fr.t, d.cls, "S1", d.track_id, d.box, fr.W, fr.H,
                                 extra=dict(text="1시 방향, 이동 중인 차량에 유의하세요.", prio=3, level="danger",
                                            motion="moving")))
        return out

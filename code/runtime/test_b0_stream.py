"""B0 스트리밍판(triggers.B0Trigger)이 오프라인 B0(baseline_rule.py → results/events/b0_E0)와 같은 이벤트를 내는지 확인.

    python test_b0_stream.py            # 10편 (실시간 대기 없이 최대 속도)
    python test_b0_stream.py --proxy    # 실행 시스템이 쓰는 재생용 영상(h264 재압축)으로도 같은지
"""
import json
import sys

import core  # noqa: F401
from common import CLIPS, OUT, iter_frames
from core import Frame
from perception import ReplayTracks, stream_frames
from triggers import B0Trigger

proxy = '--proxy' in sys.argv
clips = [x for x in sys.argv[1:] if x != '--proxy'] or list(CLIPS)
bad = 0
for key in clips:
    tr, trig = ReplayTracks("E0", key), B0Trigger()
    got = []
    src = stream_frames(key, 15.0) if proxy else iter_frames(key, 15.0, max_side=1920)
    for fidx, t, im in src:
        got += trig.step(Frame(fidx, t, im, tr(fidx, im), im.shape[1], im.shape[0]))
    ref = json.load(open(OUT / "events" / "b0_E0" / f"{key}.json", encoding="utf-8"))["events"]
    a = [(round(e.t, 2), e.scenario, e.track_id) for e in got]
    b = [(round(e["t"], 2), e["scenario"], e["track_id"]) for e in ref]
    same = a == b
    bad += not same
    print(f"{key:<22} stream {len(a):>2}  offline {len(b):>2}  {'SAME' if same else 'DIFF'}", flush=True)
    if not same:
        print("   stream :", a, "\n   offline:", b)
print("ALL SAME" if not bad else f"{bad} clips differ")

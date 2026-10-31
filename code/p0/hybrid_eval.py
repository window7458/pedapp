"""하이브리드 사후 채점: 대상·움직임 = VLM 출력, 방향 = bbox 위치(없으면 VLM), 행동 = 규칙표(시나리오 × 방향 × 움직임).
같은 run 출력을 다시 채점만 한다 (GPU 불필요). 결과를 보고 설계한 사후 분석이므로 별도 데이터(VIABench)로 확인 필요.

  python hybrid_eval.py p0_qwen2_5-vl-7b_F8_draw_oracle p0_qwen2_5-vl-3b_F8_draw_oracle
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import OUT                                              # noqa: E402
from ko_template import action_ok, direction_of, scenario_of       # noqa: E402
from vlm_eval import FIELDS, judge, load_labels                    # noqa: E402

lab = load_labels()
for run in sys.argv[1:]:
    n = fcr = 0
    acc = {k: 0 for k in FIELDS}
    for f in sorted((OUT / "vlm" / run).glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            g = lab[(r["clip"], r["event"]["gt_id"])]
            o = dict(r["out"] or {})
            if not o:
                continue
            if r["boxes"][-1] is not None:
                o["direction"] = direction_of(r["boxes"][-1], r["W"])
            cls = r["event"]["cls"]
            sc = scenario_of("other" if cls == "obstacle" else cls)
            o["action"] = (action_ok(sc, o["direction"], o["motion"]) or [o["action"]])[0]
            j = judge(o, g, r.get("visible", []))
            n += 1
            fcr += all(j[k] for k in FIELDS)
            for k in FIELDS:
                acc[k] += j[k]
    print(f"{run:40s} n={n} 완전정답={fcr / n:.2f} " + " ".join(f"{k}={acc[k] / n:.2f}" for k in FIELDS))

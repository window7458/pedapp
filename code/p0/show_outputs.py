"""경고별로 정답과 VLM 출력을 나란히 출력 (눈으로 오류 패턴 확인용).

  python show_outputs.py p0_qwen2_5-vl-7b_F8_draw_oracle
열: 클립 · 시나리오 · 박스 유무(B/-) · GT[대상 방향 움직임 허용행동] · OUT[대상 방향 움직임 행동 위험여부]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import OUT                  # noqa: E402
from vlm_eval import load_labels        # noqa: E402

lab = load_labels()
for run in sys.argv[1:]:
    print("==", run)
    for f in sorted((OUT / "vlm" / run).glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            g = lab[(r["clip"], r["event"]["gt_id"])]
            o = r["out"] or {}
            box = "B" if r["event"].get("track_id") is not None else "-"
            gt = f'{g["target_fine"]:>14} {g["direction"]:>5} {g["motion"]:>9} {"/".join(g["action_ok"]):<24}'
            out = (f'{str(o.get("target")):>14} {str(o.get("direction")):>5} {str(o.get("motion")):>11} '
                   f'{str(o.get("action")):>10} hz={o.get("hazard")}')
            print(f'{r["clip"][:14]:14} {g["scenario"]} {box} GT[{gt}] OUT[{out}]')

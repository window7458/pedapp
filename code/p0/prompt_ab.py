"""G2 지시문 비교: 같은 입력에 지시문 판만 바꿔 VLM을 다시 부르고, 위험 판단 · 움직임 · 대상을 채점 (docs/16).

입력 묶음
  oracle : 정답 경고 23개 (실험 A와 같은 입력: 정답 시각 · F8 · 빨간 박스, results/tracks/E0 필요) → 전부 "말해야 함"
  g1     : G1 E2E에서 트리거가 낸 33개 이벤트 — VLM에 실제로 들어간 이미지(vlm_inputs/) 그대로.
           판정(정탐 · 오탐 · 중립)은 score_e2e.py의 _events.csv
채점
  위험 판단: 말해야 함(oracle 23 + g1 정탐) 중 hazard=true 비율 = 재현율, 말하면 안 됨(g1 오탐) 중 hazard=false 비율 = 걸러냄
  움직임 · 대상: oracle 23개를 labels/v3 정답과 비교

  python prompt_ab.py --prompts v0 vA vAB vAB_ko --g1-run g1_e2e_10clips   → results/vlm/g2_prompt_ab/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CLIPS, OUT, clip_path               # noqa: E402
from config import MODELS                              # noqa: E402
from describe import describe                          # noqa: E402
from frames import Tracks, Video, build_images         # noqa: E402
from run_p0 import load_events                          # noqa: E402

LAB3 = HERE.parents[1] / "labels" / "v3"


def oracle_items():
    for key in CLIPS:
        evs = [e for e in load_events("oracle", key)]
        lab = {e["id"]: e for e in json.load(open(LAB3 / f"{key}.json", encoding="utf-8"))["events"]}
        evs = [e for e in evs if lab[e["gt_id"]]["kind"] == "warn"]
        if not evs:
            continue
        tracks = Tracks(OUT / "tracks" / "E0" / f"{key}.csv")
        video = Video(clip_path(key), CLIPS[key][2])
        for ev in evs:
            imgs, times, boxes = build_images(video, tracks, ev, "F8", "draw")
            H, W = video.at(ev["t"]).shape[:2]
            yield dict(src="oracle", clip=key, t=ev["t"], cls=ev["cls"], scenario=ev["scenario"], should=True,
                       gt=lab[ev["gt_id"]]), ev, imgs, times, boxes, (W, H)
        video.close()


def g1_items(run):
    base = OUT / "runtime" / run
    judge = {}
    with open(base / "_events.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            judge[(r["clip"], round(float(r["t"]), 2))] = r["judge"]
    for key in CLIPS:
        d = base / key / "vlm_inputs"
        if not d.exists():
            continue
        for ed in sorted(d.iterdir()):
            ts, rest = ed.name[2:].split("_", 1)             # ev011.33_traffic_light_897 → 11.33, traffic_light
            t, cls = float(ts), rest.rsplit("_", 1)[0]
            j = judge.get((key, round(t, 2)))
            if j is None:
                continue
            files = sorted(ed.glob("*.jpg"))
            imgs = [cv2.imread(str(f)) for f in files]
            times = [float(re.search(r"_t(\d+\.\d+)", f.name).group(1)) for f in files]
            box = [0, 0, 1, 1]                           # 이미지에 빨간 박스가 이미 그려져 있음 → "red box" 문구만 필요
            yield dict(src="g1", clip=key, t=t, cls=cls, judge=j, should={"정탐": True, "오탐": False}.get(j)), \
                dict(cls=cls), imgs, times, [box] * len(imgs), (imgs[-1].shape[1], imgs[-1].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=["v0", "vA", "vAB", "vAB_ko"])
    ap.add_argument("--model", default="qwen2_5-vl-7b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--g1-run", default="g1_e2e_10clips")
    ap.add_argument("--run", default="g2_prompt_ab")
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="local")
    served = MODELS[a.model]["hf"]
    od = OUT / "vlm" / a.run
    od.mkdir(parents=True, exist_ok=True)
    items = list(oracle_items()) + list(g1_items(a.g1_run))
    print(f"입력 {len(items)}개 (oracle {sum(i[0]['src'] == 'oracle' for i in items)} · g1 {sum(i[0]['src'] == 'g1' for i in items)})", flush=True)
    for p in a.prompts:
        with open(od / f"{p}.jsonl", "w", encoding="utf-8") as f:
            for meta, ev, imgs, times, boxes, wh in items:
                r = describe(client, served, ev, imgs, times, boxes, "draw", wh, prompt=p)
                rec = {k: v for k, v in meta.items() if k != "gt"}
                rec.update(out=r["out"], lat=r["lat"]["total_s"], tokens=r["usage"].get("prompt_tokens"))
                if "gt" in meta:
                    g = meta["gt"]
                    rec.update(gt_target_ok=g["target_ok"], gt_motion=g["motion"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{p} 완료", flush=True)
    report(od, a.prompts)


def report(od: Path, prompts):
    L = ["| 지시문 | 말해야 함 → 위험=true (재현) | 말하면 안 됨 → 위험=false (걸러냄) | 정답 23 움직임 | 정답 23 대상 | 정답 23 신호 바뀜 | 형식 오류 | 호출 중앙 |",
         "|---|---|---|---|---|---|---|---|"]
    for p in prompts:
        R = [json.loads(x) for x in open(od / f"{p}.jsonl", encoding="utf-8")]
        pos = [r for r in R if r.get("should") is True]
        neg = [r for r in R if r.get("should") is False]
        orc = [r for r in R if r["src"] == "oracle"]
        sig = [r for r in orc if r["gt_motion"] in ("to_green", "to_red")]
        ok = lambda r, k: (r["out"] or {}).get(k)                                         # noqa: E731
        lat = sorted(r["lat"] for r in R)
        L.append(f"| {p} | {sum(ok(r, 'hazard') is True for r in pos)}/{len(pos)} | {sum(ok(r, 'hazard') is False for r in neg)}/{len(neg)} | "
                 f"{sum(ok(r, 'motion') == r['gt_motion'] for r in orc)}/{len(orc)} | "
                 f"{sum(ok(r, 'target') in r['gt_target_ok'] for r in orc)}/{len(orc)} | "
                 f"{sum(ok(r, 'motion') == r['gt_motion'] for r in sig)}/{len(sig)} | {sum(r['out'] is None for r in R)} | {lat[len(lat) // 2]:.2f}s |")
    (od / "_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(OUT / "vlm" / "g2_prompt_ab", sys.argv[2:] or ["v0", "vA", "vAB", "vAB_ko"])
    else:
        main()

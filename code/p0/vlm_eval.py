"""채점: results/vlm/<run>/*.jsonl  ×  labels/v3  →  표 · 경고별 CSV · 짝 비교 검정. 판정 모델 없음 (docs/09, 10, 11).

  python vlm_eval.py pt_oracle p0_qwen2_5-vl-7b_F8_draw_oracle            # 여러 run을 같은 표에
  python vlm_eval.py --ref p0_qwen2_5-vl-7b_F8_draw_oracle <다른 run들>      # ref 대비 McNemar · Wilcoxon (Holm 보정)

정답 시각 모드(oracle): 경고 1개 = 예측 1개. 예측 시각 = t_start + 설명 지연 (VLM 전체 + TTS 첫 소리, 없으면 VLM 전체).
ESTP-F1(칸) = ESTP-F1 원식 (NeurIPS'25 부록 B.3), 내용 점수만 4칸 채점: S_answer = 1 + 맞은 칸 수 (1–5).
  유효: t ∈ [t_start − 1, t_end + 2],  t_opt = t_start,  S_time = 5 − 5·min(1, |t − t_opt| / scale),
  scale = max(1, (t_end − t_start) + 3),  S = (S_answer + S_time) / 10,
  F1 = 2ΣS / (N + M − 2·#{S>0} + 2ΣS)   (= 2ΣS/(2ΣS+FP+FN), TP를 ΣS로 대체)
주의: 정답 시각 모드는 경고마다 예측 1개 · 오탐 0이라, 모든 예측이 창 안이면 내용과 무관하게 F1 = 1이 된다 (원식의 성질).
→ 정답 시각 비교의 주 지표는 ESTP의 경고별 점수 평균 S̄ = mean S(g) (원문 식 8·10, 내용 + 시점). F1은 실제 트리거(S6)용.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT                                   # noqa: E402
from config import COARSE                                # noqa: E402

LAB3 = HERE.parents[1] / "labels" / "v3"
ANT, LAT = 1.0, 2.0
FIELDS = ["target", "direction", "motion", "action"]


def load_labels():
    lab = {}
    for f in LAB3.glob("*.json"):
        d = json.load(open(f, encoding="utf-8"))
        for e in d["events"]:
            lab[(d["clip"], e["id"])] = e
    return lab


def judge(out: dict | None, g: dict, visible: list) -> dict:
    """4칸 정/오 + 혼동/환각. out None(형식 오류) = 전부 오답."""
    if not out:
        return dict(target=0, direction=0, motion=0, action=0, coarse=0, fmt_err=1, confusion=0, halluc=0)
    t_ok = out.get("target") in g["target_ok"]
    c_ok = t_ok or COARSE.get(out.get("target")) == g["target_coarse"]
    r = dict(target=int(t_ok), coarse=int(c_ok), direction=int(out.get("direction") == g["direction"]),
             motion=int(out.get("motion") == g["motion"]), action=int(out.get("action") in g["action_ok"]),
             fmt_err=0, confusion=0, halluc=0)
    if not c_ok:
        tid = (g.get("target_track") or {}).get("track_id")
        others = {COARSE.get(c, c) for t, c in visible if t != tid}
        pc = COARSE.get(out.get("target"), out.get("target"))
        r["confusion"] = int(pc in others)            # 화면 속 다른 물체를 설명
        r["halluc"] = int(pc not in others)           # 화면에 검출된 적 없는 물체 (CHAIR식, 검출기 기준 근사)
    return r


def estp_score(s_answer: float, t_pred: float, g: dict) -> float:
    ts, te = g["t_start"], g["t_end"]
    if not (ts - ANT <= t_pred <= te + LAT):
        return 0.0
    scale = max(1.0, (te - ts) + ANT + LAT)
    s_time = 5 - 5 * min(1.0, abs(t_pred - ts) / scale)
    return (s_answer + s_time) / 10


def score_run(run: str, lab: dict) -> list[dict]:
    rows = []
    for f in sorted((OUT / "vlm" / run).glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            ev = r["event"]
            if ev.get("src") != "oracle":
                raise SystemExit(f"{run}: 실제 트리거 run 채점(S6)은 아직 미구현 — oracle run만 넣으세요")
            g = lab[(r["clip"], ev["gt_id"])]
            j = judge(r.get("out"), g, r.get("visible", []))
            n_ok = sum(j[k] for k in FIELDS)
            lat = r.get("lat", {})
            delay = r.get("desc_latency_s", lat.get("total_s", 0.0))
            s = estp_score(1 + n_ok, g["t_start"] + delay, g)
            rows.append(dict(run=run, clip=r["clip"], gt_id=ev["gt_id"], scenario=g["scenario"], tier=g.get("tier"),
                             **j, fcr=int(n_ok == 4), n_ok=n_ok, estp=s, delay=delay,
                             ttft=lat.get("ttft_s"), total=lat.get("total_s"),
                             prompt_tokens=(r.get("usage") or {}).get("prompt_tokens"),
                             out=json.dumps(r.get("out"), ensure_ascii=False)))
    return rows


def estp_f1(rows) -> float:
    S = np.array([r["estp"] for r in rows])
    N = M = len(S)                                    # oracle: 경고마다 예측 1개 → FP 0
    return float(2 * S.sum() / (N + M - 2 * (S > 0).sum() + 2 * S.sum())) if M else float("nan")


def boot_ci(rows, fn, n=1000, seed=0):
    """영상 단위 부트스트랩 95 % (docs/09 §5-7)."""
    by = defaultdict(list)
    for r in rows:
        by[r["clip"]].append(r)
    clips = list(by)
    rng = np.random.default_rng(seed)
    vals = [fn([r for c in rng.choice(clips, len(clips)) for r in by[c]]) for _ in range(n)]
    return np.nanpercentile(vals, [2.5, 97.5])


def estp_mean(rows) -> float:
    return float(np.mean([r["estp"] for r in rows])) if rows else float("nan")


def summarize(rows) -> dict:
    m = lambda k: float(np.mean([r[k] for r in rows]))
    lat = [r["delay"] for r in rows if r["delay"] is not None]
    lo, hi = boot_ci(rows, estp_mean)
    return dict(n=len(rows), estp_f1=estp_mean(rows), ci=(lo, hi), fcr=m("fcr"), target=m("target"), coarse=m("coarse"),
                direction=m("direction"), motion=m("motion"), action=m("action"), confusion=m("confusion"),
                halluc=m("halluc"), fmt_err=m("fmt_err"),
                delay_med=float(np.median(lat)) if lat else None, delay_p90=float(np.percentile(lat, 90)) if lat else None,
                tok=float(np.median([r["prompt_tokens"] for r in rows if r["prompt_tokens"]])) if any(r["prompt_tokens"] for r in rows) else None)


def mcnemar(a: list[int], b: list[int]) -> float:
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, prev = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None, help="짝 비교 기준 run")
    ap.add_argument("--out", default="eval")
    a = ap.parse_args()
    lab = load_labels()
    runs = ([a.ref] if a.ref and a.ref not in a.runs else []) + a.runs
    all_rows = {run: score_run(run, lab) for run in runs}

    od = OUT / "vlm" / "_eval"
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"{a.out}_events.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(all_rows.values()))[0]))
        w.writeheader()
        for rows in all_rows.values():
            w.writerows(rows)

    hdr = "| run | n | ESTP S̄(칸) [95% CI] | 완전정답 | 대상 | 대상(coarse) | 방향 | 움직임 | 행동 | 혼동 | 환각 | 형식오류 | 설명지연 중앙/p90 (s) | 입력 토큰 |"
    lines = [hdr, "|" + "---|" * 14]
    for run, rows in all_rows.items():
        s = summarize(rows)
        f = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(f"| {run} | {s['n']} | {s['estp_f1']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}] | {s['fcr']:.2f} | "
                     f"{s['target']:.2f} | {s['coarse']:.2f} | {s['direction']:.2f} | {s['motion']:.2f} | {s['action']:.2f} | "
                     f"{s['confusion']:.2f} | {s['halluc']:.2f} | {s['fmt_err']:.2f} | {f(s['delay_med'])} / {f(s['delay_p90'])} | "
                     f"{'-' if s['tok'] is None else int(s['tok'])} |")
    # 시나리오별 완전정답률
    lines += ["", "| run | " + " | ".join(f"S{i} 완전정답 (n)" for i in range(1, 5)) + " |", "|" + "---|" * 5]
    for run, rows in all_rows.items():
        cells = []
        for sc in ["S1", "S2", "S3", "S4"]:
            rr = [r for r in rows if r["scenario"] == sc]
            cells.append(f"{np.mean([r['fcr'] for r in rr]):.2f} ({len(rr)})" if rr else "- (0)")
        lines.append(f"| {run} | " + " | ".join(cells) + " |")

    if a.ref:
        from scipy.stats import wilcoxon
        ref = {(r["clip"], r["gt_id"]): r for r in all_rows[a.ref]}
        others = [r for r in all_rows if r != a.ref]
        p_f, p_d = [], []
        for run in others:
            pairs = [(ref[(r["clip"], r["gt_id"])], r) for r in all_rows[run] if (r["clip"], r["gt_id"]) in ref]
            p_f.append(mcnemar([x["fcr"] for x, _ in pairs], [y["fcr"] for _, y in pairs]))
            d = [(x["delay"] or 0) - (y["delay"] or 0) for x, y in pairs]
            p_d.append(1.0 if not any(d) else wilcoxon(d).pvalue)
        lines += ["", f"짝 비교 (기준 {a.ref}, Holm 보정)", "| run | 완전정답 McNemar p | 설명지연 Wilcoxon p |", "|---|---|---|"]
        for run, pf, pd_ in zip(others, holm(p_f), holm(p_d)):
            lines.append(f"| {run} | {pf:.3f} | {pd_:.3f} |")

    txt = "\n".join(lines)
    (od / f"{a.out}.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

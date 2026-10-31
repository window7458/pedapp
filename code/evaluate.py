"""이벤트 ↔ 라벨(labels/<clip>.json) 매칭.

발화 e (scenario, t) 분류
  TP       : 같은 시나리오 warn 구간 [t_start − pre, t_end] 안
  neutral  : warn은 아니지만 neutral 구간 안 (같은 시나리오 또는 범위 밖 '-' 구간과 겹침) → FP로 세지 않음
  FP       : 그 밖, 또는 같은 시나리오 suppress 구간 안
warn 라벨
  hit (strict)  : 구간 [t_start, t_end] 안에 같은 시나리오 발화  ← VIABench PDR과 같은 기준
  hit (lenient) : [t_start − pre, t_end]  (앞서 말한 경고도 인정, pre = 2 s)
"""
from __future__ import annotations

import json

from common import CLIPS, ROOT

SCEN = ["S1", "S2", "S3", "S4"]


def load_labels(key):
    return json.load(open(ROOT.parent / "labels" / "v2" / f"{key}.json", encoding="utf-8"))["events"]


def same(e, l):
    """같은 시나리오 + (S3이면) 알린 색이 정답 목표 색과 같음"""
    if e["scenario"] != l["scenario"]:
        return False
    if l["scenario"] == "S3" and l.get("state") and e.get("state"):
        return e["state"] == l["state"]
    return True


def match(events, labels, dur, pre=2.0):
    warn = [l for l in labels if l["kind"] == "warn"]
    res = {"tp": 0, "fp": 0, "neutral": 0, "fp_list": [], "warn": len(warn),
           "hit_strict": 0, "hit_lenient": 0, "lead": [], "dur": dur, "n_utt": len(events),
           "by_scen": {s: {"warn": 0, "hit_strict": 0, "hit_lenient": 0, "tp": 0, "fp": 0, "neutral": 0} for s in SCEN}}
    for l in warn:
        bs = res["by_scen"][l["scenario"]]
        bs["warn"] += 1
        es = [e["t"] for e in events if same(e, l)]
        s = any(l["t_start"] <= t <= l["t_end"] for t in es)
        le = [t for t in es if l["t_start"] - pre <= t <= l["t_end"]]
        res["hit_strict"] += s
        bs["hit_strict"] += s
        res["hit_lenient"] += bool(le)
        bs["hit_lenient"] += bool(le)
        if le:
            res["lead"].append(l["t_start"] - min(le))
    for e in events:
        t, sc = e["t"], e["scenario"]
        bs = res["by_scen"].setdefault(sc, {"warn": 0, "hit_strict": 0, "hit_lenient": 0, "tp": 0, "fp": 0, "neutral": 0})
        sup = any(l["kind"] == "suppress" and l["scenario"] == sc and l["t_start"] <= t <= l["t_end"] for l in labels)
        tp = any(l["kind"] == "warn" and same(e, l) and l["t_start"] - pre <= t <= l["t_end"] for l in labels)
        neu = any(l["kind"] == "neutral" and (l["scenario"] == "-" or same(e, l)) and l["t_start"] - pre <= t <= l["t_end"] + 0.5
                  for l in labels)
        if sup:
            k = "fp"
        elif tp:
            k = "tp"
        elif neu:
            k = "neutral"
        else:
            k = "fp"
        res[k] += 1
        bs[k] += 1
        if k == "fp":
            res["fp_list"].append({"t": t, "scenario": sc, "text": e.get("text", ""), "suppressed_zone": sup})
    return res


def summarize(results: dict) -> dict:
    tot = {"warn": 0, "hit_strict": 0, "hit_lenient": 0, "tp": 0, "fp": 0, "neutral": 0, "dur": 0.0, "n_utt": 0, "lead": []}
    by = {s: {"warn": 0, "hit_strict": 0, "hit_lenient": 0, "tp": 0, "fp": 0, "neutral": 0} for s in SCEN}
    for r in results.values():
        for k in ("warn", "hit_strict", "hit_lenient", "tp", "fp", "neutral", "dur", "n_utt"):
            tot[k] += r[k]
        tot["lead"] += r["lead"]
        for s in SCEN:
            for k in by[s]:
                by[s][k] += r["by_scen"].get(s, {}).get(k, 0)
    mins = tot["dur"] / 60
    return {
        "PDR_strict": round(tot["hit_strict"] / max(1, tot["warn"]), 3),
        "PDR_lenient": round(tot["hit_lenient"] / max(1, tot["warn"]), 3),
        "precision": round(tot["tp"] / max(1, tot["tp"] + tot["fp"]), 3),
        "fp_per_min": round(tot["fp"] / max(mins, 1e-6), 2),
        "utt_per_min": round(tot["n_utt"] / max(mins, 1e-6), 2),
        "lead_mean_s": round(sum(tot["lead"]) / max(1, len(tot["lead"])), 2),
        "warn": tot["warn"], "tp": tot["tp"], "fp": tot["fp"], "neutral": tot["neutral"],
        "by_scenario": {s: {**v, "PDR_lenient": round(v["hit_lenient"] / v["warn"], 2) if v["warn"] else None} for s, v in by.items()},
    }


def eval_system(events_by_clip: dict, durs: dict, pre=2.0):
    res = {k: match(events_by_clip.get(k, []), load_labels(k), durs[k], pre) for k in CLIPS}
    return res, summarize(res)

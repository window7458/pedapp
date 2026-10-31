"""G1 · G2 채점: run_demo 결과(log.jsonl) × 정답 라벨 → 이벤트별 표 + 요약 (docs/16).

  python score_e2e.py g1_e2e_10clips          → results/runtime/<run>/_score.md, _events.csv

이벤트 하나(분배기를 통과해 경고가 나간 것)마다:
  판정   : 정탐(같은 시나리오 정답 구간 [시작 − 2 s, 끝] 안) · 오탐 · 중립 — code/evaluate.py와 같은 기준 (labels/v2)
  경고   : 재생 여부 · 지연 (소리 시작 − 트리거 시각)
  VLM    : 위험 여부 · 대상 · 움직임 · 호출 시간
  설명   : 재생 / 위험 아님이라 안 함 / 밀려서 버림 / 늦어서 버림 · 지연
  내용   : 정탐이면 labels/v3 4칸 정답과 비교 — 대상(target_ok 안) · 움직임
G2: 정탐 경고 중 VLM이 "위험 아님"으로 버린 비율(놓침) vs 오탐 중 VLM이 버린 비율(걸러냄).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import core  # noqa: F401
from common import CLIPS, OUT, clip_info
from evaluate import load_labels, same

ROOT = Path(__file__).resolve().parents[2]
PRE = 2.0


def v3_labels(clip):
    p = ROOT / "labels" / "v3" / f"{clip}.json"
    return json.load(open(p, encoding="utf-8"))["events"] if p.exists() else []


def judge_event(e, labels):
    t, sc = e["t"], e["scenario"]
    if any(l["kind"] == "suppress" and l["scenario"] == sc and l["t_start"] <= t <= l["t_end"] for l in labels):
        return "오탐", None
    for l in labels:
        if l["kind"] == "warn" and same(e, l) and l["t_start"] - PRE <= t <= l["t_end"]:
            return "정탐", l
    if any(l["kind"] == "neutral" and (l["scenario"] == "-" or same(e, l)) and l["t_start"] - PRE <= t <= l["t_end"] + 0.5
           for l in labels):
        return "중립", None
    return "오탐", None


def score_clip(od: Path, clip: str):
    rows = [json.loads(x) for x in open(od / "log.jsonl", encoding="utf-8")]
    labels, l3 = load_labels(clip), v3_labels(clip)
    evs = {}
    for r in rows:
        st = r["stage"]
        if st == "event":
            e = r["ev"]
            evs[e["t"]] = dict(clip=clip, t=e["t"], scenario=e["scenario"], cls=e["cls"], track=e.get("track_id"),
                               state=e.get("state"), warn_text=r.get("warn_text"), dispatched=True,
                               warn="-", warn_lat=None, hazard=None, vlm_target=None, vlm_motion=None, vlm_s=None,
                               desc="-", desc_lat=None, desc_text=None)
        elif st == "suppress" and r.get("ev_t") in evs:
            evs[r["ev_t"]]["dispatched"] = False
        elif st == "vlm_end" and r["ev_t"] in evs:
            o = r.get("out") or {}
            evs[r["ev_t"]].update(hazard=o.get("hazard"), vlm_target=o.get("target"), vlm_motion=o.get("motion"),
                                  vlm_s=(r.get("lat") or {}).get("total_s"))
        elif st == "desc_skip" and r["ev_t"] in evs:
            evs[r["ev_t"]]["desc"] = "위험 아님→안 함" if r["why"] == "no_hazard" else r["why"]
        elif st == "desc_text" and r["ev_t"] in evs:
            evs[r["ev_t"]]["desc_text"] = r["text"]
        elif st == "audio_start" and r["t_event"] in evs:
            k = "warn" if r["kind"] == "warn" else "desc"
            evs[r["t_event"]][k] = "재생"
            evs[r["t_event"]][f"{k}_lat"] = round(r["now"] - r["t_event"], 3)
        elif st == "audio_drop" and r["t_event"] in evs:
            k = "warn" if r["kind"] == "warn" else "desc"
            evs[r["t_event"]][k] = {"superseded": "밀려서 버림", "desc_late": "늦어서 버림", "warn_late": "늦어서 버림"}.get(r["why"], r["why"])
    out = []
    for e in evs.values():
        if not e["dispatched"]:
            continue
        j, l = judge_event(e, labels)
        e["judge"] = j
        e["target_ok"] = e["motion_ok"] = None
        if l is not None and e["hazard"] is not None:
            g = next((x for x in l3 if x.get("id") == l.get("id")), None)
            if g:
                e["target_ok"] = e["vlm_target"] in g.get("target_ok", [])
                e["motion_ok"] = e["vlm_motion"] == g.get("motion")
        out.append(e)
    warn_lbl = [l for l in labels if l["kind"] == "warn"]
    hit = sum(any(same(e, l) and l["t_start"] - PRE <= e["t"] <= l["t_end"] for e in out) for l in warn_lbl)
    return out, len(warn_lbl), hit


def main():
    run = sys.argv[1]
    base = OUT / "runtime" / run
    allev, nwarn, nhit, dur, summ = [], 0, 0, 0.0, {}
    for clip in CLIPS:
        od = base / clip
        if not (od / "log.jsonl").exists():
            continue
        ev, w, h = score_clip(od, clip)
        allev += ev
        nwarn += w
        nhit += h
        dur += clip_info(clip)["dur"]
        summ[clip] = json.load(open(od / "summary.json", encoding="utf-8"))
    c = Counter(e["judge"] for e in allev)
    tp = [e for e in allev if e["judge"] == "정탐"]
    fp = [e for e in allev if e["judge"] == "오탐"]

    def frac(xs, f):
        return f"{sum(map(f, xs))}/{len(xs)}" if xs else "-"

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return f"{xs[len(xs) // 2]:.2f} s" if xs else "-"

    perc = [s.get("perception_ms", {}).get("median") for s in summ.values()]
    lag = [s.get("frame_lag_s", {}).get("max") for s in summ.values()]
    L = [f"# {run} — E2E 채점 (영상 {len(summ)}편, {dur / 60:.1f}분)", "",
         "## 요약", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 정답 경고 | {nwarn}개 → 맞힌 것(PDR, 2 s 앞 허용) **{nhit}/{nwarn}** |",
         f"| 트리거 이벤트 (분배기 통과) | {len(allev)}개 = 정탐 {c['정탐']} · 오탐 {c['오탐']} · 중립 {c['중립']} · 오탐/분 {c['오탐'] / (dur / 60):.2f} |",
         f"| 즉시 경고 재생 | {frac(allev, lambda e: e['warn'] == '재생')} · 지연 중앙 {med([e['warn_lat'] for e in allev])} |",
         f"| VLM 호출 | {sum(e['vlm_s'] is not None for e in allev)}회 · 호출 시간 중앙 {med([e['vlm_s'] for e in allev])} |",
         f"| VLM \"위험\" 답 | 전체 {frac(allev, lambda e: e['hazard'] is True)} |",
         f"| 설명 재생 | {frac(allev, lambda e: e['desc'] == '재생')} · 지연 중앙 {med([e['desc_lat'] for e in allev])} |",
         f"| 인식 프레임당 (편별 중앙값) | {min(perc):.1f}–{max(perc):.1f} ms |",
         f"| 프레임 지연 최대 (편별) | {min(lag):.2f}–{max(lag):.2f} s |", "",
         "## G2 — VLM 위험 걸러내기", "",
         "| 트리거 판정 | 이벤트 | VLM \"위험\" (설명 감) | VLM \"위험 아님\" (설명 안 함) |", "|---|---|---|---|",
         f"| 정탐 (말해야 함) | {len(tp)} | {sum(e['hazard'] is True for e in tp)} | **{sum(e['hazard'] is False for e in tp)}** ← 놓침 |",
         f"| 오탐 (말하면 안 됨) | {len(fp)} | **{sum(e['hazard'] is True for e in fp)}** ← 못 거름 | {sum(e['hazard'] is False for e in fp)} ← 걸러냄 |",
         f"| 중립 | {c['중립']} | {sum(e['hazard'] is True for e in allev if e['judge'] == '중립')} | {sum(e['hazard'] is False for e in allev if e['judge'] == '중립')} |", "",
         f"정탐에서 VLM 내용: 대상 맞음 {frac([e for e in tp if e['target_ok'] is not None], lambda e: e['target_ok'])} · "
         f"움직임 맞음 {frac([e for e in tp if e['motion_ok'] is not None], lambda e: e['motion_ok'])}", "",
         "## 이벤트별", "",
         "| 영상 | 시각 | 트리거 | 판정 | 경고 | VLM (위험 · 대상 · 움직임 · 시간) | 설명 |", "|---|---|---|---|---|---|---|"]
    for e in allev:
        wl = f"{e['warn']} {e['warn_lat']:.2f}s" if e["warn_lat"] is not None else e["warn"]
        hz = "" if e["hazard"] is None else ("위험 · " if e["hazard"] else "위험 아님 · ")    # 2칸 출력엔 위험 칸 없음
        vl = "-" if e["vlm_s"] is None else f"{hz}{e['vlm_target']} · {e['vlm_motion']} · {e['vlm_s']:.2f}s"
        dl = f"{e['desc']} {e['desc_lat']:.2f}s" if e["desc_lat"] is not None else e["desc"]
        L.append(f"| {e['clip'][:3]} | {e['t']:.2f} | {e['scenario']} {e['cls']} #{e['track']} | {e['judge']} | {wl} | {vl} | {dl} |")
    (base / "_score.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(base / "_events.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(allev[0].keys()))
        w.writeheader()
        w.writerows(allev)
    print("\n".join(L[:22]))
    print(f"→ {base / '_score.md'}")


if __name__ == "__main__":
    main()

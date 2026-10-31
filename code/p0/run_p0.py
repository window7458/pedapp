"""P0 실행: 트리거 이벤트(또는 정답 시각) → 프레임 창 → VLM(JSON 4칸) → 한국어 문장 → TTS. 녹화 영상 오프라인 처리.

예)
  # 배관 점검 (서버 없이 입력 이미지·프롬프트만 저장)
  python run_p0.py --clips c09_bike_on_tactile --events b0_E0 --dry
  # S0: 규칙 트리거 이벤트, Qwen2.5-VL-7B, F8, 박스 그림
  python run_p0.py --events b0_E0 --model qwen2_5-vl-7b --frames F8 --bbox draw
  # S-M / S-F: 정답 시각 (labels/<clip>.json 의 warn 시작 시각)
  python run_p0.py --events oracle --model internvl3_5-8b --frames F3
  # PT (VLM 없음)
  python run_p0.py --events b0_E0 --arm pt

출력: results/vlm/<run>/<clip>.jsonl  (이벤트 1줄 = 입력·출력·지연 전부, 채점은 vlm_eval.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                        # code/common.py

from common import CLIPS, OUT, clip_path                     # noqa: E402
from config import BBOX_MODES, FRAMES, MODELS                # noqa: E402
from frames import Tracks, Video, build_images               # noqa: E402
from ko_template import direction_of, pt_output, sentence, warn_text  # noqa: E402

LABELS = HERE.parents[1] / "labels" / "v3"


def load_events(src: str, key: str) -> list[dict]:
    """b0_E0 등 = results/events/<src>/<clip>.json (트리거 출력)
       oracle  = labels/v3/<clip>.json 의 경고 정답, 시각 = t_start (docs/09 §5-1).
                 cls = 정답 coarse 클래스(검출기가 줄 수 있는 정보), track_id = 라벨의 대상 트랙 (없으면 박스 없이)."""
    if src == "oracle":
        lab = json.load(open(LABELS / f"{key}.json", encoding="utf-8"))
        return [dict(t=e["t_start"], gt_id=e["id"], scenario=e["scenario"], cls=e["target_coarse"],
                     track_id=(e.get("target_track") or {}).get("track_id"), src="oracle")
                for e in lab["events"]]
    ev = json.load(open(OUT / "events" / src / f"{key}.json", encoding="utf-8"))["events"]
    return [dict(e, src=src) for e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=list(CLIPS))
    ap.add_argument("--events", default="b0_E0", help="results/events/<이름> 또는 oracle")
    ap.add_argument("--tracks", default="E0", help="results/tracks/<이름>")
    ap.add_argument("--arm", default="p0", choices=["p0", "pt"])
    ap.add_argument("--model", default="qwen2_5-vl-7b", choices=list(MODELS))
    ap.add_argument("--frames", default="F8", choices=list(FRAMES))
    ap.add_argument("--bbox", default="draw", choices=BBOX_MODES)
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM 서버")
    ap.add_argument("--repeat", type=int, default=1, help="지연 측정 반복 (docs/09 §5-7: 3)")
    ap.add_argument("--tts-url", default=None, help="로컬 TTS 서버 (예: supertonic serve). 없으면 TTS 생략")
    ap.add_argument("--tts-model", default="supertonic")
    ap.add_argument("--tts-voice", default="F3")
    ap.add_argument("--prompt", default="v0", help="지시문 판 (code/p0/prompts.py). h2 · h2g = 대상 · 움직임 2칸 (G3)")
    ap.add_argument("--dry", action="store_true", help="서버 호출 없이 입력 이미지·프롬프트만 저장")
    ap.add_argument("--run", default=None, help="결과 폴더 이름")
    a = ap.parse_args()

    run = a.run or (f"pt_{a.events}" if a.arm == "pt" else f"{a.arm}_{a.model}_{a.frames}_{a.bbox}_{a.events}")
    od = OUT / "vlm" / run
    od.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(od / "args.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    client = tts = None
    if a.arm == "p0" and not a.dry:
        from openai import OpenAI
        client = OpenAI(base_url=a.base_url, api_key="local")          # 로컬 vLLM, 키는 형식상 필요
    if a.tts_url and not a.dry:
        from openai import OpenAI
        from tts import TTS
        tts = TTS(OpenAI(base_url=a.tts_url, api_key="local"), a.tts_model, a.tts_voice, OUT / "vlm" / "tts_cache")
    served = MODELS[a.model]["hf"]

    for key in a.clips:
        evs = load_events(a.events, key)
        tr_path = OUT / "tracks" / a.tracks / f"{key}.csv"
        tracks = Tracks(tr_path) if tr_path.exists() else None
        video = Video(clip_path(key), CLIPS[key][2])
        fout = open(od / f"{key}.jsonl", "w", encoding="utf-8")
        for i, ev in enumerate(evs):
            imgs, times, boxes = build_images(video, tracks, ev, a.frames if a.arm == "p0" else "F1", a.bbox)
            H, W = video.at(ev["t"]).shape[:2]
            box_now = boxes[-1]
            rec = dict(clip=key, idx=i, event=ev, arm=a.arm, model=a.model if a.arm == "p0" else None,
                       frames=a.frames, bbox=a.bbox, times=[round(t, 3) for t in times],
                       boxes=[None if b is None else [round(v, 1) for v in b] for b in boxes], W=W, H=H,
                       visible=tracks.visible(ev["t"]) if tracks else [])
            d0 = direction_of(box_now, W) if box_now is not None else "front"
            rec["warn_text"] = warn_text(ev.get("cls", "obstacle"), d0)

            if a.arm == "pt":
                rec["out"], rec["lat"] = pt_output(ev, box_now, W), {"total_s": 0.0}
            elif a.dry:
                import cv2
                from describe import build_prompt
                for j, im in enumerate(imgs):                     # imwrite는 한글 경로에서 조용히 실패 → imencode
                    (od / f"{key}_{i:03d}_{j:02d}.jpg").write_bytes(cv2.imencode(".jpg", im)[1].tobytes())
                rec["prompt"] = build_prompt(ev, times, boxes, a.bbox, (W, H))
                rec["out"] = None
            else:
                from describe import describe
                runs = [describe(client, served, ev, imgs, times, boxes, a.bbox, (W, H), prompt=a.prompt) for _ in range(a.repeat)]
                rec.update(out=runs[0]["out"], raw=runs[0]["raw"], usage=runs[0]["usage"],
                           lat_runs=[r["lat"] for r in runs],
                           lat={k: sorted(r["lat"][k] for r in runs)[len(runs) // 2] for k in runs[0]["lat"]})

            if rec.get("out") and "direction" not in rec["out"]:     # 2칸 출력(G3) → 방향 · 행동은 코드가 채움
                from ko_template import action_ok, scenario_of
                o = dict(rec["out"], hazard=True)
                o["direction"] = direction_of(box_now, W) if box_now is not None else "front"
                sc = ev.get("scenario") or scenario_of("other" if ev.get("cls") == "obstacle" else ev.get("cls"))
                o["action"] = (action_ok(sc, o["direction"], o["motion"]) or ["caution"])[0]
                rec["out_vlm"], rec["out"] = rec["out"], o
            rec["sentence"] = sentence(rec["out"]) if rec.get("out") else None
            if tts:
                _, rec["tts_warn"] = tts.synth(rec["warn_text"])
                if rec["sentence"]:
                    _, rec["tts_desc"] = tts.synth(rec["sentence"])
                # 설명 지연 = VLM 전체 + TTS 첫 소리 (docs/09 §5-2). 경고 지연 = 경고 wav (미리 합성 시 0)
                rec["warn_latency_s"] = rec["tts_warn"]["first_audio_s"]
                rec["desc_latency_s"] = rec["lat"]["total_s"] + rec.get("tts_desc", {}).get("first_audio_s", 0.0)
            fout.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
            fout.flush()
            print(f"{key} #{i:<3} t={ev['t']:6.2f} {ev.get('cls', ''):<14} "
                  f"{rec['sentence'] or '-'}  ({rec.get('lat', {}).get('total_s', 0):.2f}s)", flush=True)
        fout.close()
        video.close()


if __name__ == "__main__":
    main()

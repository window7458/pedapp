"""실행 시스템: 녹화 영상을 실제 속도로 재생하며 ① 인식 → ② 트리거 → ③ 분배 → ④ 경고 / ⑤ VLM → ⑥ TTS → ⑦ 재생.

예)
  # 노트북 (서버 없음): 저장된 E0 트랙 + B0 + VLM 지연 1 s 흉내 + Windows 한국어 음성, 스피커로 재생
  python run_demo.py --clip c09_bike_on_tactile --vlm mock --mock-latency 1.0 --play
  # 시연 영상 만들기 (소리 + 박스 + 자막)
  python run_demo.py --clip c09_bike_on_tactile --vlm mock --render
  # 서버 (A5000): YOLO 실시간 + vLLM + TTS 서버
  python run_demo.py --clip c09_bike_on_tactile --perception yolo --device 0 \
                     --vlm server --base-url http://localhost:8000/v1 --tts server --tts-url http://localhost:8880/v1
  # 다른 트리거 결과로 같은 실험 (예: 임태규 규칙을 오프라인으로 돌린 결과)
  python run_demo.py --clip c09_bike_on_tactile --trigger replay:b0_E0

출력 results/runtime/<run>/<clip>/ : log.jsonl (모든 단계 시각) · timeline.json (재생 · 폐기) · summary.json
         (--render) demo.mp4 · audio.wav
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import core
from common import CLIPS, OUT, clip_info
from core import Clock, Frame, Log
from audio import AudioQueue
from dispatcher import DispatchConfig, Dispatcher, all_warn_texts
from perception import FrameBuffer, ReplayTracks, YoloTracker, build_proxy, needs_proxy, stream_frames
from speech import make_engine
from triggers import make_trigger
from vlm_worker import MockBackend, PTBackend, ServerBackend, VLMWorker


def summarize(timeline, disp_stats, lag):
    def lat(kind):
        xs = [it["t_start"] - it["t_event"] for it in timeline if it["kind"] == kind and "t_start" in it]
        return dict(n=len(xs), median_s=round(statistics.median(xs), 3) if xs else None,
                    max_s=round(max(xs), 3) if xs else None)
    cnt = lambda kind, st: sum(1 for it in timeline if it["kind"] == kind and it["status"] == st)  # noqa: E731
    return {"dispatch": disp_stats,
            "warn": {**lat("warn"), "played": cnt("warn", "played"), "dropped": cnt("warn", "dropped")},
            "desc": {**lat("desc"), "played": cnt("desc", "played"), "cut": cnt("desc", "cut"),
                     "dropped": cnt("desc", "dropped")},
            "frame_lag_s": {"median": round(statistics.median(lag), 3) if lag else None,
                            "max": round(max(lag), 3) if lag else None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, choices=list(CLIPS))
    ap.add_argument("--perception", default="replay", choices=["replay", "yolo"])
    ap.add_argument("--tracks", default="E0", help="replay: results/tracks/<이름> · yolo: 가중치 키")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--trigger", default="b0", help="b0 | replay:<results/events 폴더>")
    ap.add_argument("--vlm", default="mock", choices=["none", "pt", "mock", "server"])
    ap.add_argument("--mock-latency", type=float, default=1.0)
    ap.add_argument("--model", default="qwen2_5-vl-7b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--frames", default="F8")
    ap.add_argument("--bbox", default="draw")
    ap.add_argument("--prompt", default="h2", help="VLM 지시문 판 (code/p0/prompts.py). 기본 h2 = 대상 · 움직임 2칸 (G3, 10/9). v0 = 예전 5칸")
    ap.add_argument("--no-hybrid", action="store_true")
    ap.add_argument("--vlm-filter", action="store_true", help="VLM이 위험 아님이라 답하면 설명 생략 (10/9 전 기본값, 비교용)")
    ap.add_argument("--tts", default="sapi", choices=["sapi", "server"])
    ap.add_argument("--tts-url", default=None)
    ap.add_argument("--tts-model", default="supertonic")
    ap.add_argument("--tts-voice", default="F3")
    ap.add_argument("--play", action="store_true", help="스피커로 실제 재생")
    ap.add_argument("--render", action="store_true", help="끝난 뒤 demo.mp4 (소리 · 박스 · 자막) 생성")
    ap.add_argument("--run", default=None)
    ap.add_argument("--dump-vlm-inputs", action="store_true", help="VLM에 넣은 이미지를 <결과>/vlm_inputs/에 저장")
    a = ap.parse_args()

    vlm_tag = f"mock{a.mock_latency:g}s" if a.vlm == "mock" else a.model if a.vlm == "server" else a.vlm
    run = a.run or f"{a.trigger.replace(':', '-')}_{vlm_tag}_{a.tts}"
    od = OUT / "runtime" / run / a.clip
    od.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(od / "args.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    clock = Clock()
    log = Log(od / "log.jsonl", clock)
    clock.start()                                      # 준비 단계 시각도 기록되게 (아래에서 다시 0으로)

    # ⑥ TTS + 경고 문장 미리 합성
    tts = make_engine(a.tts, OUT / "runtime" / "tts_cache", base_url=a.tts_url, model=a.tts_model, voice=a.tts_voice)
    t0 = time.perf_counter()
    for f in [tts.synth(x) for x in all_warn_texts()]:
        f.result()
    print(f"경고 문장 {len(all_warn_texts())}개 준비 {time.perf_counter() - t0:.1f}s", flush=True)

    # ① ② ③ ⑤ ⑦
    perception = ReplayTracks(a.tracks, a.clip) if a.perception == "replay" else YoloTracker(a.tracks, a.device)
    trigger = make_trigger(a.trigger, a.clip, a.fps)
    buffer = FrameBuffer()
    audio = AudioQueue(clock, log, play=a.play)
    cfg = DispatchConfig(frames=a.frames, bbox=a.bbox, use_hybrid=not a.no_hybrid,
                         skip_no_hazard=a.vlm_filter, use_vlm=a.vlm != "none",
                         dump_dir=str(od / "vlm_inputs") if a.dump_vlm_inputs else None)
    disp = Dispatcher(cfg, log, buffer, audio, tts)
    vlm = None
    if a.vlm != "none":
        backend = {"pt": lambda: PTBackend(), "mock": lambda: MockBackend(a.mock_latency),
                   "server": lambda: ServerBackend(a.base_url, a.model, a.bbox, a.prompt)}[a.vlm]()
        vlm = VLMWorker(backend, disp.on_vlm, log)
        disp.vlm = vlm
        vlm.start()
    audio.start()

    if needs_proxy(a.clip):                            # 4K 원본: 무손실 재생용 영상 (처음 한 번만 만듦)
        t0 = time.perf_counter()
        build_proxy(a.clip, a.fps)
        print(f"재생용 영상 준비 {time.perf_counter() - t0:.1f}s", flush=True)
    info = clip_info(a.clip)
    print(f"{a.clip}  {info['dur']:.1f}s  perception={perception.name} trigger={trigger.name} "
          f"vlm={a.vlm} tts={tts.name}", flush=True)
    if hasattr(perception, "warmup"):                  # 첫 프레임 GPU 초기화 시간을 실시간 밖으로
        t0 = time.perf_counter()
        _, _, im0 = next(iter(stream_frames(a.clip, a.fps)))
        perception.warmup(im0)
        print(f"인식 모델 예열 {time.perf_counter() - t0:.1f}s", flush=True)
    dets_log, lag, perc = open(od / "dets.jsonl", "w", encoding="utf-8"), [], []
    clock.start()
    log("start", clip=a.clip)
    for fidx, t, im in stream_frames(a.clip, a.fps):
        clock.sleep_until(t)                           # 실제 속도 (늦으면 기다리지 않고 바로 처리)
        tp = time.perf_counter()
        dets = perception(fidx, im)
        perc.append(time.perf_counter() - tp)
        fr = Frame(fidx, t, im, dets, im.shape[1], im.shape[0])
        buffer.push(fr)
        for ev in trigger.step(fr):
            disp.on_event(fr, ev)
        lag.append(clock.now() - t)
        dets_log.write(json.dumps({"f": fidx, "t": round(t, 3), "perc_s": round(perc[-1], 4),
                                   "d": [[d.track_id, d.cls, *[round(v) for v in d.box]] for d in dets]},
                                  ensure_ascii=False) + "\n")
    dets_log.close()
    log("video_end", lag_median=round(statistics.median(lag), 3))
    perc_ms = sorted(x * 1000 for x in perc[3:] or perc)               # 처음 몇 프레임은 예열
    print(f"인식(perception) 프레임당 중앙 {statistics.median(perc_ms):.1f} ms · p90 {perc_ms[int(len(perc_ms) * .9)]:.1f} ms"
          f" → 최대 {1000 / statistics.median(perc_ms):.0f} FPS (입력 {a.fps:g} fps)", flush=True)

    deadline = time.perf_counter() + 15                # 남은 설명 · 재생이 끝나길 기다림
    while time.perf_counter() < deadline and not (audio.idle() and (vlm is None or vlm.idle())):
        time.sleep(0.05)
    time.sleep(0.2)
    audio.stop()
    if vlm:
        vlm.stop()
    tts.close()

    timeline = sorted(audio.timeline, key=lambda it: it.get("t_start", it.get("t_drop", 0)))
    json.dump(timeline, open(od / "timeline.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    summ = summarize(timeline, disp.stats, lag)
    summ["perception_ms"] = {"median": round(statistics.median(perc_ms), 1), "p90": round(perc_ms[int(len(perc_ms) * .9)], 1)}
    vl = [r["lat"]["total_s"] for r in log.rows if r.get("stage") == "vlm_end" and (r.get("lat") or {}).get("total_s")]
    if vl:
        summ["vlm_call_s"] = {"n": len(vl), "median": round(statistics.median(vl), 3), "max": round(max(vl), 3)}
    json.dump(summ, open(od / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.close()

    print("\n재생 기록")
    for it in timeline:
        if "t_start" in it:
            print(f"  {it['t_start']:6.2f}s  [{'경고' if it['kind'] == 'warn' else '설명'}] {it['text']}"
                  f"   (트리거 {it['t_event']:.2f}s + {it['t_start'] - it['t_event']:.2f}s, {it['status']})")
        else:
            print(f"  {it['t_drop']:6.2f}s  [버림] {it['text']}  ({it['why']})")
    print("\n요약", json.dumps(summ, ensure_ascii=False))
    print(f"→ {od}")

    if a.render:
        from render import render
        render(a.clip, od, a.fps)


if __name__ == "__main__":
    main()

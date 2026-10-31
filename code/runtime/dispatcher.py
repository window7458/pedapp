"""③ 분배기 — 트리거 이벤트 하나를 '즉시 경고'와 'VLM 설명' 두 갈래로 나눈다.

즉시 경고  : 클래스 × 방향 템플릿 ("왼쪽 오토바이"). 미리 합성해 둔 wav → VLM을 기다리지 않음
VLM 설명   : 트리거 순간의 프레임 창을 VLM 작업자에 넘김 → JSON → (하이브리드) → 문장 → TTS → 재생 큐
중복 방지  : 같은 트랙 cooldown_s 안 재경고 금지, 같은 경고 문장 same_text_s 안 반복 금지

VLM ↔ 트리거 정합성 규칙 (docs/13 '정해야 할 규칙' — 기본값, 실험으로 바꿀 수 있게 설정으로 둠)
  · 위험 판단은 트리거 몫 (10/9 결정, G2: VLM hazard는 모델별 버릇) → 기본은 VLM이 위험 아님이라 해도 설명을 말함.
    skip_no_hazard=True(`--vlm-filter`)면 예전처럼 생략 (비교용). hazard 값은 항상 로그에 남김
  · VLM 대상 ≠ YOLO 클래스 (coarse) → VLM 쪽을 말함 (여러 프레임을 보고 판단한 쪽). 불일치는 로그로 남겨 채점
"""
from __future__ import annotations

from dataclasses import dataclass

import core  # noqa: F401
from common import CLASSES
from ko_template import KO_DIR, direction_of, sentence, warn_text
from vlm_worker import build_inputs, coarse_match, hybrid

S3_WARN = {"GO": "초록불로 바뀜", "STOP": "빨간불로 바뀜"}


@dataclass
class DispatchConfig:
    cooldown_s: float = 4.0
    same_text_s: float = 3.0
    warn_ttl_s: float = 2.0          # 경고를 이 안에 못 틀면 버림
    desc_deadline_s: float = 6.0     # 설명은 트리거 후 이 안에 시작 못 하면 버림
    frames: str = "F8"
    bbox: str = "draw"
    use_hybrid: bool = True
    skip_no_hazard: bool = False     # 10/9: 위험 판단은 트리거가 함 (G2)
    use_vlm: bool = True
    dump_dir: str | None = None      # 지정하면 VLM에 넣은 이미지를 그대로 저장 (연결 확인용)


def warn_text_of(ev) -> str:
    if ev.scenario == "S3" and ev.extra.get("state") in S3_WARN:
        return S3_WARN[ev.extra["state"]]
    d = direction_of(ev.box, ev.W) if ev.box is not None else "front"
    return warn_text(ev.cls, d)


def all_warn_texts() -> list[str]:
    """미리 합성할 경고 문장 전부 (클래스 × 방향 + 신호 2개)."""
    return [warn_text(c, d) for c in CLASSES for d in KO_DIR] + list(S3_WARN.values())


class Dispatcher:
    def __init__(self, cfg: DispatchConfig, log, buffer, audio, tts, vlm=None):
        self.cfg, self.log, self.buffer, self.audio, self.tts, self.vlm = cfg, log, buffer, audio, tts, vlm
        self.last_track, self.last_text = {}, {}
        self.stats = dict(events=0, suppressed=0, vlm_jobs=0, desc_skip=0, mismatch=0, bad_json=0)

    def on_event(self, frame, ev):
        c = self.cfg
        self.stats["events"] += 1
        text = warn_text_of(ev)
        self.log("event", ev=ev.as_dict(), warn_text=text)
        if ev.track_id is not None and ev.t - self.last_track.get((ev.scenario, ev.track_id), -1e9) < c.cooldown_s:
            self.stats["suppressed"] += 1
            self.log("suppress", why="track_cooldown", ev_t=ev.t, track_id=ev.track_id)
            return
        if ev.t - self.last_text.get(text, -1e9) < c.same_text_s:
            self.stats["suppressed"] += 1
            self.log("suppress", why="same_text", ev_t=ev.t, text=text)
            return
        if ev.track_id is not None:
            self.last_track[(ev.scenario, ev.track_id)] = ev.t
        self.last_text[text] = ev.t

        fut = self.tts.synth(text)                       # 미리 합성했으면 즉시 완료
        fut.add_done_callback(lambda f: self._push(f, "warn", text, ev, ev.t + c.warn_ttl_s))

        if c.use_vlm and self.vlm is not None:
            imgs, times, boxes = build_inputs(self.buffer, frame, ev, c.frames, c.bbox)
            self.stats["vlm_jobs"] += 1
            self.log("vlm_input", ev_t=ev.t, scenario=ev.scenario, cls=ev.cls, track_id=ev.track_id,
                     ev_box=None if ev.box is None else [round(v) for v in ev.box], warn_text=text,
                     img_times=[round(t, 2) for t in times], n_img=len(imgs),
                     boxes=[None if b is None else [round(v) for v in b] for b in boxes],
                     img_shapes=[list(im.shape[:2]) for im in imgs])
            if c.dump_dir:
                import cv2
                from pathlib import Path
                d = Path(c.dump_dir) / f"ev{ev.t:06.2f}_{ev.cls}_{ev.track_id}"
                d.mkdir(parents=True, exist_ok=True)
                for k, (im, t) in enumerate(zip(imgs, times)):
                    cv2.imwrite(str(d / f"{k}_t{t:06.2f}.jpg"), im)
            self.vlm.submit(dict(ev=ev, imgs=imgs, times=times, boxes=boxes))

    def _push(self, fut, kind, text, ev, deadline):
        try:
            path, dur, st = fut.result()
        except Exception as e:
            self.log("tts_error", kind=kind, text=text, error=repr(e))
            return
        self.log("tts_done", kind=kind, text=text, ev_t=ev.t, **st)
        self.audio.push(kind, text, path, dur, ev.t, deadline, scenario=ev.scenario, track_id=ev.track_id)

    def on_vlm(self, job, res):
        ev, c = job["ev"], self.cfg
        out = res.get("out")
        if not out:
            self.stats["bad_json"] += 1
            self.log("desc_skip", why="no_output", ev_t=ev.t, raw=res.get("raw"), error=res.get("error"))
            return
        o = hybrid(out, ev) if c.use_hybrid else out
        if not coarse_match(o, ev):
            self.stats["mismatch"] += 1
            self.log("target_mismatch", ev_t=ev.t, yolo=ev.cls, vlm=o.get("target"))
        if c.skip_no_hazard and not o.get("hazard", True):
            self.stats["desc_skip"] += 1
            self.log("desc_skip", why="no_hazard", ev_t=ev.t, out=o)
            return
        text = sentence(o)
        self.log("desc_text", ev_t=ev.t, out=o, text=text)
        fut = self.tts.synth(text)
        fut.add_done_callback(lambda f: self._push(f, "desc", text, ev, ev.t + c.desc_deadline_s))

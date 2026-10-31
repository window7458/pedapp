"""실시간 파이프라인 — 프레임 → 인식 → Trigger → 즉시 경고 / VLM 설명, 그리고 특별 상황 묘사.

팀 실행 시스템(code/runtime)의 부품(YoloTracker · Trigger · FrameBuffer · VLM 작업자 · 한국어 문장 규칙)을
그대로 쓰고, 재생 큐 · 스피커 대신 emit(dict)로 웹앱에 내보낸다. 음성은 폰 브라우저가 낸다.

내보내는 메시지 (type)
  dets   : 이 프레임의 박스 (0–1로 정규화)         warn  : 즉시 경고 문장
  desc   : VLM이 만든 설명 문장 (경고 뒤에 온다)    scene : 특별 상황 묘사 (불꽃놀이 · 벚꽃길 …)
  status : 1 s마다 속도 · 지연                     source / scene_error : 상태 알림
"""
from __future__ import annotations

import importlib
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parent
CODE = APP.parent
for p in (CODE, CODE / "p0", CODE / "runtime"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import core  # noqa: E402,F401  (sys.path 설정)
from core import Clock, Det, Frame  # noqa: E402
from ko_template import direction_of, sentence  # noqa: E402
from perception import FrameBuffer  # noqa: E402
from dispatcher import DispatchConfig, warn_text_of  # noqa: E402
from vlm_worker import (MockBackend, PTBackend, ServerBackend, VLMWorker, build_inputs, coarse_match,  # noqa: E402
                        hybrid)

from scene import SceneDescriber, make_backend  # noqa: E402


@dataclass
class Config:
    perception: str = "none"            # none | demo | yolo
    weights: str = "E0"                 # 가중치 키(E0 · F10 · A5) 또는 .pt 파일 경로
    device: str = "cpu"
    conf: float | None = None           # YOLO 신뢰도. 비우면 트리거가 정함 (hitbox 는 0.2, 그 밖은 0.25)
    trigger: str = "b0"                 # b0 | hitbox | none | 모듈경로:클래스  (step(frame) -> [Event] 를 가진 것)
    vlm: str = "pt"                     # none | pt | mock | server
    mock_latency: float = 1.0
    vlm_url: str = "http://localhost:8000/v1"
    vlm_model: str = "qwen2_5-vl-7b"
    vlm_prompt: str = "h2"
    scene: str = "none"                 # none | mock | openai | anthropic
    scene_url: str | None = None
    scene_model: str | None = None
    scene_interval: float = 5.0
    max_fps: float = 15.0
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)


class DemoPerception:
    """모델 없이 UI · 경고 흐름을 시험하는 가짜 인식: 오른쪽에서 자동차 하나가 계속 커진다 (5 s 주기)."""
    name = "demo"

    def __init__(self):
        self.k0 = None

    def __call__(self, fidx, image):
        H, W = image.shape[:2]
        if self.k0 is None:
            self.k0 = fidx
        k = (fidx - self.k0) % 75
        w = (0.10 + 0.004 * k) * W
        h = w * 0.7
        cx, cy = 0.72 * W, 0.62 * H
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        tid = 1 + (fidx - self.k0) // 75
        return [Det(int(tid), "car", 0.9, box, (cx, box[3]), 0, w * h)]


class NoPerception:
    name = "none"

    def __call__(self, fidx, image):
        return []


class NoTrigger:
    name = "none"

    def step(self, fr):
        return []


def make_perception(cfg: Config):
    if cfg.perception == "none":
        return NoPerception()
    if cfg.perception == "demo":
        return DemoPerception()
    if cfg.perception == "yolo":
        from pathlib import Path as _P
        from common import WEIGHTS
        if cfg.weights not in WEIGHTS and _P(cfg.weights).exists():       # --weights /경로/final_10class_best.pt
            WEIGHTS[cfg.weights] = _P(cfg.weights)
        from perception import YoloTracker
        conf = cfg.conf if cfg.conf is not None else (0.2 if cfg.trigger.startswith("hitbox") else 0.25)
        return YoloTracker(cfg.weights, cfg.device, conf=conf)
    raise ValueError(f"unknown perception {cfg.perception} (none | demo | yolo)")


def make_trigger(cfg: Config):
    if cfg.trigger == "none":
        return NoTrigger()
    if cfg.trigger == "b0":
        from triggers import B0Trigger
        return B0Trigger(cfg.max_fps)
    spec = {"hitbox": "hitbox_trigger:HitboxTrigger"}.get(cfg.trigger, cfg.trigger)     # hitbox = 사용자 v5_29
    if ":" in spec:                         # 예: my_trigger:MyTrigger  (code/app 또는 PYTHONPATH에 있는 모듈)
        mod, cls = spec.split(":", 1)
        C = getattr(importlib.import_module(mod), cls)
        import inspect
        return C(cfg) if "cfg" in inspect.signature(C).parameters else C()
    raise ValueError(f"unknown trigger {cfg.trigger} (b0 | hitbox | none | 모듈:클래스)")


class LivePipeline:
    def __init__(self, cfg: Config, emit, log=print):
        self.cfg, self.emit, self.log = cfg, emit, log
        self.clock = Clock()
        self.clock.start()
        self.buffer = FrameBuffer()
        self.perception = make_perception(cfg)
        self.trigger = make_trigger(cfg)
        self.events = 0
        self.last_track, self.last_text = {}, {}
        self.event_seq = 0

        self.vlm = None
        if cfg.vlm != "none":
            backend = {"pt": lambda: PTBackend(), "mock": lambda: MockBackend(cfg.mock_latency),
                       "server": lambda: ServerBackend(cfg.vlm_url, cfg.vlm_model, cfg.dispatch.bbox, cfg.vlm_prompt)
                       }[cfg.vlm]()
            self.vlm = VLMWorker(backend, self._on_vlm, self._vlm_log)
            self.vlm.start()
        self.vlm_name = getattr(self.vlm.backend, "name", "none") if self.vlm else "none"

        sb = make_backend(cfg.scene, cfg.scene_url, cfg.scene_model)
        self.scene = SceneDescriber(sb, emit, min_interval=cfg.scene_interval,
                                    max_interval=max(cfg.scene_interval * 2.4, 12.0), log=log)
        if sb is not None:
            self.scene.start()
        self.scene_name = self.scene.name

        self.preview_enabled = False
        self.preview_jpeg: bytes | None = None
        self._latest, self._evt = None, threading.Event()
        self._stop = threading.Event()
        self._fidx = 0
        self._last_proc = -1e9
        self._perc, self._lag, self._stamps = deque(maxlen=60), deque(maxlen=60), deque(maxlen=60)
        self.source_label = "phone"
        self.last_frame = None              # 보호자 '지금 화면' · 알림 사진용
        self.on_warn = None                 # (msg, ev, image) — 보호자 알림 센터가 붙는다
        self._warmed = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    # ── 입력 ──────────────────────────────────────────────
    def submit(self, image: np.ndarray):
        """소스(폰 · 카메라 URL)가 호출. 항상 최신 한 장만 남긴다 — 밀리면 옛 프레임은 버린다."""
        self._latest = image
        self._evt.set()

    def set_scene(self, on: bool):
        self.scene.set_enabled(on)

    def set_source_label(self, label: str):
        self.source_label = label

    # ── 처리 루프 ─────────────────────────────────────────
    def _loop(self):
        while not self._stop.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            im, self._latest = self._latest, None
            if im is None:
                continue
            now = time.perf_counter()
            if now - self._last_proc < 1.0 / self.cfg.max_fps - 1e-3:
                continue
            self._last_proc = now
            try:
                self._process(im)
            except Exception as e:           # 한 프레임 오류로 전체가 멈추면 안 된다
                self.log(f"[pipeline] frame error: {e!r}")

    def _process(self, im: np.ndarray):
        if max(im.shape[:2]) > 1920:
            s = 1920 / max(im.shape[:2])
            im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if not self._warmed and hasattr(self.perception, "warmup"):
            self.perception.warmup(im)       # 첫 프레임 모델 올리는 시간을 시계 밖으로
            self._warmed = True
        self.last_frame = im
        t = self.clock.now()
        self._fidx += 1
        tp = time.perf_counter()
        dets = self.perception(self._fidx, im)
        self._perc.append(time.perf_counter() - tp)
        H, W = im.shape[:2]
        fr = Frame(self._fidx, t, im, dets, W, H)
        self.buffer.push(fr)
        self.scene.push(t, im)
        for ev in self.trigger.step(fr):
            self._on_event(fr, ev)
        self._lag.append(self.clock.now() - t)
        self._stamps.append(time.perf_counter())
        if self.preview_enabled and self._fidx % 2 == 0:
            self._make_preview(im)
        self.emit({"type": "dets", "t": round(t, 2), "W": W, "H": H,
                   "d": [[d.track_id, d.cls, round(d.box[0] / W, 4), round(d.box[1] / H, 4),
                          round(d.box[2] / W, 4), round(d.box[3] / H, 4)] for d in dets]})

    def _make_preview(self, im):
        s = min(1.0, 640 / max(im.shape[:2]))
        small = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s))) if s < 1 else im
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.preview_jpeg = buf.tobytes()

    # ── 경고 / 설명 (runtime/dispatcher.py와 같은 규칙) ───
    def _on_event(self, fr, ev):
        c = self.cfg.dispatch
        self.events += 1
        text = ev.extra.get("text") or warn_text_of(ev)           # 트리거가 문장을 직접 줄 수 있다 (hitbox)
        if not getattr(self.trigger, "self_deduped", False):       # 트리거가 이미 중복을 거른다면 여기서 또 거르지 않음
            if ev.track_id is not None and ev.t - self.last_track.get((ev.scenario, ev.track_id), -1e9) < c.cooldown_s:
                return
            if ev.t - self.last_text.get(text, -1e9) < c.same_text_s:
                return
            if ev.track_id is not None:
                self.last_track[(ev.scenario, ev.track_id)] = ev.t
            self.last_text[text] = ev.t
        self.event_seq += 1
        eid = self.event_seq
        direction = direction_of(ev.box, ev.W) if ev.box is not None else "front"
        msg = {"type": "warn", "id": eid, "text": text, "cls": ev.cls, "scenario": ev.scenario,
               "direction": direction, "track_id": ev.track_id, "t": round(ev.t, 2),
               "prio": ev.extra.get("prio", 1), "level": ev.extra.get("level"), "motion": ev.extra.get("motion")}
        self.emit(msg)
        if self.on_warn is not None:
            try:
                self.on_warn(msg, ev, fr.image)
            except Exception as e:
                self.log(f"[alerts] {e!r}")
        if self.vlm is not None and c.use_vlm and getattr(self.trigger, "wants_vlm", True) and ev.box is not None:
            imgs, times, boxes = build_inputs(self.buffer, fr, ev, c.frames, c.bbox)
            self.vlm.submit(dict(ev=ev, imgs=imgs, times=times, boxes=boxes, eid=eid))

    def _on_vlm(self, job, res):
        ev, c = job["ev"], self.cfg.dispatch
        out = res.get("out")
        if not out:
            self.log(f"[vlm] no output: {res.get('error') or res.get('raw')}")
            return
        o = hybrid(out, ev) if c.use_hybrid else out
        if c.skip_no_hazard and not o.get("hazard", True):
            return
        self.emit({"type": "desc", "id": job.get("eid"), "text": sentence(o), "cls": ev.cls,
                   "t_event": round(ev.t, 2), "t": round(self.clock.now(), 2),
                   "mismatch": not coarse_match(o, ev)})

    def _vlm_log(self, stage, **kw):
        if stage in ("vlm_end",) and kw.get("error"):
            self.log(f"[vlm] error: {kw['error']}")

    # ── 상태 ──────────────────────────────────────────────
    def _status_loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)
            st = self._stamps
            fps = 0.0
            if len(st) >= 2 and st[-1] - st[0] > 0:
                fps = (len(st) - 1) / (st[-1] - st[0])
            if time.perf_counter() - (st[-1] if st else 0) > 2.0:
                fps = 0.0
            self.emit({"type": "status", "fps": round(fps, 1),
                       "perc_ms": round(statistics.median(self._perc) * 1000, 1) if self._perc else None,
                       "lag_s": round(statistics.median(self._lag), 3) if self._lag else None,
                       "source": self.source_label, "perception": self.perception.name,
                       "trigger": getattr(self.trigger, "name", "?"), "vlm": self.vlm_name,
                       "scene": self.scene_name, "events": self.events,
                       "hb": getattr(self.trigger, "info", None) or None})

    def stop(self):
        self._stop.set()
        self.scene.stop()
        if self.vlm:
            self.vlm.stop()

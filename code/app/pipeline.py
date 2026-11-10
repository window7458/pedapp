"""실시간 파이프라인 — 프레임 → 인식 → Trigger → 즉시 경고.
(경고에 VLM 설명을 붙이던 기능과 5초마다 특별 장면을 묻던 기능은 뺐다. VLM은 사용자가 "주변 설명해줘"라고 할 때만 따로 쓴다.)

팀 실행 시스템(code/runtime)의 부품(YoloTracker · Trigger · FrameBuffer · VLM 작업자 · 한국어 문장 규칙)을
그대로 쓰고, 재생 큐 · 스피커 대신 emit(dict)로 웹앱에 내보낸다. 음성은 폰 브라우저가 낸다.

내보내는 메시지 (type)
  dets   : 이 프레임의 박스 (0–1로 정규화)         warn  : 즉시 경고 문장
  status : 1 s마다 속도 · 지연                     source : 상태 알림
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
from ko_template import direction_of  # noqa: E402
from perception import FrameBuffer  # noqa: E402
from dispatcher import DispatchConfig, warn_text_of  # noqa: E402


@dataclass
class Config:
    perception: object = "yolo"         # yolo | none (테스트는 가짜 인식기 객체를 직접 넣는다)
    weights: str = "auto"               # auto(weights/ 폴더에서 찾기) 또는 .pt 파일 경로
    device: str = "cpu"
    conf: float | None = None           # YOLO 신뢰도 (b0 등 앱 YoloTracker 를 쓰는 트리거만. nav 는 자체 기준)
    trigger: str = "nav"                # nav(임태규 v12.1) | b0 | none | 모듈경로:클래스  (step(frame) -> [Event] 를 가진 것)
    nav_imgsz: int | None = None        # nav YOLO 입력 크기. 비우면 CPU 640 · GPU 960
    nav_crop: bool = True               # nav 의 0.4 s 마다 잘라 보기 (작은 신호등 · 먼 물체)
    max_fps: float = 15.0
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)


WEIGHT_DIRS = [APP / "weights", CODE / "weights", CODE.parent / "weights"]
WEIGHT_PREF = ["final_10class_best.pt", "best_v2.pt", "best.pt"]
NAME_FIX = {"traffic light": "traffic_light", "truck": "other_vehicle", "e_scooter": "scooter", "kickboard": "scooter"}


def find_weights(spec: str | None) -> Path | None:
    """--weights 가 파일이면 그대로, 'auto'(기본)면 weights/ 폴더에서 찾는다 (final_10class_best.pt 우선)."""
    if spec and spec != "auto":
        p = Path(spec)
        if p.exists():
            return p
        try:
            from common import WEIGHTS
            if spec in WEIGHTS and Path(WEIGHTS[spec]).exists():
                return Path(WEIGHTS[spec])
        except Exception:
            pass
        import re
        if re.fullmatch(r"yolo(v8|11|12)?[nsmlx](-seg)?\.pt", spec):    # 공식 일반 모델 이름이면 ultralytics 가 받아 온다 (시험용)
            return Path(spec)
        return None
    for d in WEIGHT_DIRS:
        for n in WEIGHT_PREF:
            if (d / n).exists():
                return d / n
    for d in WEIGHT_DIRS:
        if d.is_dir():
            pts = sorted(d.glob("*.pt"), key=lambda p: ("seg" not in p.name, p.name))
            pts = [p for p in pts if not p.name.startswith(("yolo11n-cls", "signal"))]
            if pts:
                return pts[0]
    return None


def foot_point(poly, box) -> tuple[float, float]:
    """마스크 최하단 접지점 (최하단 2 % 높이 띠의 x 평균). 마스크 없으면 bbox 하단 중앙. (SJ code/dump_tracks.py 와 같음)"""
    if poly is None or len(poly) < 3:
        return (box[0] + box[2]) / 2, box[3]
    ys = poly[:, 1]
    vmax = ys.max()
    band = poly[ys >= vmax - 0.02 * max(1.0, box[3] - box[1])]
    return float(band[:, 0].mean()), float(vmax)


def cv2_area(poly) -> float:
    return float(cv2.contourArea(np.asarray(poly, np.float32)))


class AppYoloTracker:
    """팀 YOLO11s-seg(10클래스) + ByteTrack. 클래스 이름은 가중치 안의 이름을 그대로 쓴다
    (runtime.perception.YoloTracker 는 클래스 순서를 고정해 둬서, 순서가 다른 가중치면 이름이 틀어진다)."""
    name = "yolo"

    def __init__(self, path: Path, device="cpu", imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.m = YOLO(str(path))
        self.path, self.device, self.imgsz, self.conf = Path(path), device, imgsz, conf
        self.names = {int(k): NAME_FIX.get(v, v) for k, v in self.m.names.items()}

    def warmup(self, image, n=3):
        for _ in range(n):
            self.m.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)

    def __call__(self, fidx, image):
        H, W = image.shape[:2]
        r = self.m.track(image, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz, conf=self.conf,
                         device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy if r.masks is not None else [None] * len(ids)
        out = []
        for b, tid, c, p, poly in zip(xyxy, ids, cls, conf, polys):
            fu, fv = foot_point(poly, b)
            edge = int(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            area = float(cv2_area(poly)) if poly is not None and len(poly) >= 3 else 0.0
            out.append(Det(int(tid), self.names.get(int(c), str(c)), round(float(p), 3),
                           tuple(round(float(v), 1) for v in b), (round(fu, 1), round(fv, 1)), edge, round(area, 1)))
        return out


class NoPerception:
    name = "none"

    def __call__(self, fidx, image):
        return []


class NoTrigger:
    name = "none"

    def step(self, fr):
        return []


def make_perception(cfg: Config):
    if not isinstance(cfg.perception, str):          # 테스트가 가짜 인식기를 직접 넣을 때
        return cfg.perception
    if cfg.perception == "none":
        return NoPerception()
    if cfg.perception == "yolo":
        w = find_weights(cfg.weights)
        if w is None:
            raise FileNotFoundError(
                "가중치(.pt)를 찾지 못했어요. code/app/weights/ 폴더에 final_10class_best.pt 를 넣거나 --weights 로 경로를 주세요.")
        cfg.weights = str(w)
        if cfg.trigger == "nav":                     # 트리거가 같은 모델로 직접 추론한다 → 인식기를 겸한다
            from nav_trigger import NavTrigger
            return NavTrigger(w, cfg.device, cfg.nav_imgsz, cfg.nav_crop)
        conf = cfg.conf if cfg.conf is not None else 0.25
        return AppYoloTracker(w, cfg.device, conf=conf)
    raise ValueError(f"unknown perception {cfg.perception} (yolo | none)")


def make_trigger(cfg: Config):
    if cfg.trigger == "none":
        return NoTrigger()
    if cfg.trigger == "b0":
        from triggers import B0Trigger
        return B0Trigger(cfg.max_fps)
    if cfg.trigger == "nav":
        raise ValueError("nav 트리거는 perception='yolo' 와 함께만 쓴다 (트리거가 직접 추론함)")
    spec = cfg.trigger
    if ":" in spec:                         # 예: my_trigger:MyTrigger  (code/app 또는 PYTHONPATH에 있는 모듈)
        mod, cls = spec.split(":", 1)
        C = getattr(importlib.import_module(mod), cls)
        import inspect
        return C(cfg) if "cfg" in inspect.signature(C).parameters else C()
    raise ValueError(f"unknown trigger {cfg.trigger} (nav | b0 | none | 모듈:클래스)")


class LivePipeline:
    def __init__(self, cfg: Config, emit, log=print):
        self.cfg, self.emit, self.log = cfg, emit, log
        self.clock = Clock()
        self.clock.start()
        self.buffer = FrameBuffer()
        self.perception = make_perception(cfg)
        self.trigger = self.perception if getattr(self.perception, "is_trigger", False) else make_trigger(cfg)
        from face_blur import FaceBlur
        self.blur = FaceBlur(log=log)             # 밖으로 나가는 영상의 얼굴 가리기 (판단은 원본으로)
        self.events = 0
        self.last_track, self.last_text = {}, {}
        self.event_seq = 0

        self.preview_enabled = False
        self.live_viewers = 0                # 보호자 실시간 영상을 보는 사람 수
        self.preview_jpeg: bytes | None = None
        self._latest, self._evt = None, threading.Event()
        self._stop = threading.Event()
        self._fidx = 0
        self._last_proc = -1e9
        self._perc, self._lag, self._stamps = deque(maxlen=60), deque(maxlen=60), deque(maxlen=60)
        self._blur = deque(maxlen=60)
        self._rx = deque(maxlen=60)          # 받은 프레임 시각 (처리한 프레임 _stamps 와 구분: 전달 문제 ↔ 처리 지연)
        self._black = False
        self.source_label = "phone"
        self.last_frame = None              # 보호자 '지금 화면' · 알림 사진용
        self.on_frame = None                # (image) — 위험 구간 녹화기가 붙는다
        self.on_warn = None                 # (msg, ev, image) — 보호자 알림 센터가 붙는다
        self._warmed = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    # ── 입력 ──────────────────────────────────────────────
    def submit(self, image: np.ndarray, meta: dict | None = None):
        """소스(폰 · 카메라 URL)가 호출. 항상 최신 한 장만 남긴다 — 밀리면 옛 프레임은 버린다."""
        self._rx.append(time.perf_counter())
        self._latest = (image, meta)
        self._evt.set()


    def set_source_label(self, label: str):
        self.source_label = label

    # ── 처리 루프 ─────────────────────────────────────────
    def _loop(self):
        while not self._stop.is_set():
            if not self._evt.wait(0.5):
                continue
            self._evt.clear()
            item, self._latest = self._latest, None
            if item is None:
                continue
            im, meta = item
            now = time.perf_counter()
            if now - self._last_proc < 1.0 / self.cfg.max_fps - 1e-3:
                continue
            self._last_proc = now
            try:
                self._process(im, meta)
            except Exception as e:           # 한 프레임 오류로 전체가 멈추면 안 된다
                self.log(f"[pipeline] frame error: {e!r}")

    def _process(self, im: np.ndarray, meta: dict | None = None):
        t_in = time.perf_counter()
        if max(im.shape[:2]) > 1920:
            s = 1920 / max(im.shape[:2])
            im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if not self._warmed and hasattr(self.perception, "warmup"):
            self.perception.warmup(im)       # 첫 프레임 모델 올리는 시간을 시계 밖으로
            self._warmed = True
        self._black = float(im[::16, ::16].mean()) < 3.0     # 카메라가 멈추면 폰이 까만 화면을 보낸다
        t = self.clock.now()
        self._fidx += 1
        tp = time.perf_counter()
        if hasattr(self.perception, "set_capture"):
            self.perception.set_capture((meta or {}).get("c"))   # 폰이 찍은 시각 → 센서와 시간 맞추기
        dets = self.perception(self._fidx, im)              # 인식 · 판단은 원본으로
        self._perc.append(time.perf_counter() - tp)
        tb = time.perf_counter()
        safe = self.blur.apply(im, dets)                   # 여기부터 저장 · 전송되는 영상은 얼굴을 가린 사본
        self._blur.append(time.perf_counter() - tb)
        self.last_frame = safe
        if self.on_frame is not None:
            self.on_frame(safe)
        H, W = im.shape[:2]
        fr = Frame(self._fidx, t, safe, dets, W, H)
        self.buffer.push(fr)
        for ev in self.trigger.step(fr):
            self._on_event(fr, ev)
        self._lag.append(self.clock.now() - t)
        self._stamps.append(time.perf_counter())
        if (self.preview_enabled or self.live_viewers > 0) and self._fidx % 2 == 0:
            self._make_preview(safe, dets)
        self.emit({"type": "dets", "t": round(t, 2), "W": W, "H": H,
                   "d": [[d.track_id, d.cls, round(d.box[0] / W, 4), round(d.box[1] / H, 4),
                          round(d.box[2] / W, 4), round(d.box[3] / H, 4)] for d in dets]})
        if meta and meta.get("c") is not None:     # 파이 카메라: 찍은 순간 → 결과 시간을 카메라 쪽에서 잰다
            self.emit({"type": "frame_done", "c": meta["c"], "p_ms": round((time.perf_counter() - t_in) * 1000, 1)})

    def _make_preview(self, im, dets=()):
        s = min(1.0, 640 / max(im.shape[:2]))
        small = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s))) if s < 1 else im.copy()
        for d in dets:                                   # 보호자 실시간 영상에도 인식 박스를 그린다
            x1, y1, x2, y2 = (int(v * s) for v in d.box)
            cv2.rectangle(small, (x1, y1), (x2, y2), (255, 120, 40), 2)
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
                self.on_warn(msg, ev, fr.image, fr.dets)
            except Exception as e:
                self.log(f"[alerts] {e!r}")

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
            now_ = time.perf_counter()
            rx = sum(1 for x in list(self._rx) if now_ - x <= 3.0) / 3.0
            self.emit({"type": "status", "fps": round(fps, 1), "rx_fps": round(rx, 1), "black": self._black,
                       "perc_ms": round(statistics.median(self._perc) * 1000, 1) if self._perc else None,
                       "blur_ms": round(statistics.median(self._blur) * 1000, 1) if self._blur else None,
                       "face": self.blur.name,
                       "lag_s": round(statistics.median(self._lag), 3) if self._lag else None,
                       "source": self.source_label, "perception": self.perception.name,
                       "trigger": getattr(self.trigger, "name", "?"), "events": self.events,
                       "hb": getattr(self.trigger, "info", None) or None})

    def stop(self):
        self._stop.set()

"""상시 서버(hub)의 인식 엔진 연결 — GPU 작업 서버가 붙어 있으면 그쪽으로, 없으면 CPU 느린 모드(또는 꺼짐).

  폰 ──영상──▶ 상시 서버(이 파일) ──영상──▶ GPU 작업 서버(worker.py: YOLO + 히트박스 + VLM)
                     ▲                              │
                     └──── dets · warn · desc ───────┘

상시 서버가 하는 일
  · 받은 프레임에 번호(seq)를 붙여 최근 몇 초를 들고 있다가, GPU 가 '몇 번 프레임에서 경고'라고 알려 주면
    그 장면으로 보호자 알림 사진 · 위험 구간 영상을 만든다 (판단은 GPU 쪽이 하고, 기록은 여기서 한다).
  · GPU 가 없을 때: SAYEYE_FALLBACK=cpu 면 이 서버 CPU 로 같은 파이프라인을 느리게 돌리고, none 이면 인식을 쉰다.
    보호자 알림 · 위치 · 호출은 GPU 와 상관없이 항상 된다.
  · 서버(server.py)에는 LivePipeline 과 같은 모양(submit · set_scene · info · fps · last_frame …)으로 보인다.
"""
from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

KEEP_FRAMES = 60          # 경고 장면을 찾기 위해 들고 있는 최근 프레임 수 (10 fps 면 6초)


class Router:
    wants_jpeg = True

    def __init__(self, emit, fallback_cfg=None, log=print):
        self.emit, self.log = emit, log
        self.fallback_cfg = fallback_cfg
        self.local = None                    # CPU 느린 모드 파이프라인 (처음 필요할 때 만든다)
        self._local_lock = threading.Lock()
        self.worker = None                   # {"q": asyncio.Queue, "info": dict, "since": float}
        self.worker_status: dict = {}
        self.local_status: dict = {}
        self.frames: deque = deque(maxlen=KEEP_FRAMES)     # (seq, image)
        self.seq = 0
        self.scene_on = True
        self.last_frame = None
        self.last_dets: list = []            # 정규화 박스 [id, cls, x1, y1, x2, y2]
        self.preview_jpeg: bytes | None = None
        self.live_viewers = 0
        self.on_frame = None                 # 위험 구간 녹화기
        self.on_warn = None                  # 보호자 알림 센터
        self._rx = deque(maxlen=60)
        self._done = deque(maxlen=60)
        self._black = False
        self._stop = threading.Event()
        self._last_mode = None
        threading.Thread(target=self._status_loop, daemon=True).start()
        if fallback_cfg is not None:         # 모델을 미리 올려 둔다 (첫 보행 때 기다리지 않게)
            threading.Thread(target=self._ensure_local, daemon=True).start()

    # ── 지금 어느 엔진이 일하나 ─────────────────────────
    def mode(self) -> str:
        if self.worker is not None:
            return "gpu"
        if self.local is not None:
            return "cpu"
        return "none"

    def _ensure_local(self):
        with self._local_lock:
            if self.local is None and self.fallback_cfg is not None:
                from pipeline import LivePipeline
                try:
                    p = LivePipeline(self.fallback_cfg, self._local_emit, self.log)
                    p.on_warn = self._local_warn
                    p.set_scene(self.scene_on)
                    self.local = p
                    self.log("[engine] CPU 느린 모드 준비됨")
                    self._announce()
                except Exception as e:
                    self.log(f"[engine] CPU 모드를 켤 수 없어요: {e!r}")
                    self.fallback_cfg = None

    def _announce(self):
        m = self.mode()
        if m != self._last_mode:
            self._last_mode = m
            i = self.info()
            self.emit({"type": "engine", **self.engine_state(), "weights": i.get("weights"), "trigger": i.get("trigger"),
                       "vlm": i.get("vlm")})

    def engine_state(self) -> dict:
        m = self.mode()
        return {"mode": m, "gpu": m == "gpu",
                "label": {"gpu": "GPU 인식", "cpu": "느린 인식(CPU)", "none": "인식 꺼짐"}[m]}

    # ── 입력 (이벤트 루프 스레드에서 불린다) ─────────────
    def set_source_label(self, label: str):
        pass

    def submit(self, image: np.ndarray, jpeg: bytes | None = None):
        self._rx.append(time.perf_counter())
        self.seq += 1
        seq = self.seq
        self.frames.append((seq, image))
        self.last_frame = image
        self._black = float(image[::16, ::16].mean()) < 3.0
        if self.on_frame is not None:
            self.on_frame(image)
        w = self.worker
        if w is not None:
            if jpeg is None:
                jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            q = w["q"]
            if q.qsize() >= 2:                   # GPU 가 밀리면 옛 프레임은 버린다 (항상 최신 우선). 설정 메시지는 남긴다
                keep = []
                while not q.empty():
                    x = q.get_nowait()
                    if isinstance(x, str):
                        keep.append(x)
                for x in keep:
                    q.put_nowait(x)
            q.put_nowait(struct.pack("!I", seq) + jpeg)
        elif self.local is not None:
            self.local.submit(image, tag=seq)
        if self.live_viewers > 0 and seq % 2 == 0:
            self._make_preview(image)

    def set_scene(self, on: bool):
        self.scene_on = bool(on)
        if self.local is not None:
            self.local.set_scene(on)
        if self.worker is not None:
            self.worker["q"].put_nowait(json.dumps({"type": "set_scene", "on": self.scene_on}))

    # ── GPU 작업 서버 연결 ──────────────────────────────
    def attach_worker(self, info: dict) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.worker = {"q": q, "info": info, "since": time.time()}
        self.worker_status = {}
        q.put_nowait(json.dumps({"type": "set_scene", "on": self.scene_on}))
        self.log(f"[engine] GPU 작업 서버 연결됨: {info.get('weights')} · VLM {info.get('vlm')}")
        self._announce()
        return q

    def detach_worker(self, q):
        if self.worker is not None and self.worker["q"] is q:
            self.worker = None
            self.log("[engine] GPU 작업 서버 연결 끊김")
            self._announce()

    def on_worker_msg(self, m: dict):
        t = m.get("type")
        if t == "status":
            self.worker_status = m
        elif t == "worker_info" and self.worker is not None:
            self.worker["info"].update(m.get("info") or {})
        elif t == "warn_ctx":
            self._warn_ctx(m)
        elif t in ("dets", "warn", "desc", "scene", "scene_error"):
            if t == "dets":
                self.last_dets = m.get("d") or []
                self._done.append(time.perf_counter())
            self.emit(m)

    def _warn_ctx(self, m: dict):
        """GPU 가 낸 경고에 그 순간의 장면을 붙여 보호자 알림을 만든다."""
        if self.on_warn is None:
            return
        tag = m.get("tag")
        image = next((im for s, im in reversed(self.frames) if s == tag), self.last_frame)
        e = m.get("ev") or {}
        ev = SimpleNamespace(cls=e.get("cls"), box=e.get("box"), W=e.get("W"), H=e.get("H"), track_id=e.get("track_id"))
        dets = [SimpleNamespace(track_id=d[0], cls=d[1], conf=d[2], box=d[3:7]) for d in m.get("dets") or []]
        try:
            self.on_warn(m.get("msg") or {}, ev, image, dets)
        except Exception as ex:
            self.log(f"[alerts] {ex!r}")

    # ── CPU 느린 모드 ───────────────────────────────────
    def _local_emit(self, m: dict):
        if self.worker is not None:          # GPU 가 붙은 뒤에는 CPU 결과를 내보내지 않는다
            return
        if m.get("type") == "status":
            self.local_status = m
            return
        if m.get("type") == "dets":
            self.last_dets = m.get("d") or []
            self._done.append(time.perf_counter())
        self.emit(m)

    def _local_warn(self, msg, ev, image, dets):
        if self.worker is None and self.on_warn is not None:
            self.on_warn(msg, ev, image, dets)

    # ── 보호자 실시간 영상 ──────────────────────────────
    def _make_preview(self, im):
        s = min(1.0, 640 / max(im.shape[:2]))
        small = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s))) if s < 1 else im.copy()
        H, W = small.shape[:2]
        for d in self.last_dets:
            x1, y1, x2, y2 = int(d[2] * W), int(d[3] * H), int(d[4] * W), int(d[5] * H)
            cv2.rectangle(small, (x1, y1), (x2, y2), (255, 120, 40), 2)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.preview_jpeg = buf.tobytes()

    # ── 상태 ────────────────────────────────────────────
    def info(self) -> dict:
        m = self.mode()
        if m == "gpu":
            base = dict(self.worker["info"])
        elif m == "cpu":
            base = self.local.info()
        else:
            base = {"perception": "none", "trigger": "none", "vlm": "none", "vlm_up": False, "scene": "none", "weights": ""}
        return {**base, "engine": self.engine_state()}

    def fps(self) -> float:
        st = list(self._done)
        if len(st) < 2 or st[-1] - st[0] <= 0 or time.perf_counter() - st[-1] > 2.0:
            return 0.0
        return round((len(st) - 1) / (st[-1] - st[0]), 1)

    def _status_loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)
            self._announce()
            src = self.worker_status if self.worker is not None else self.local_status if self.local is not None else {}
            now = time.perf_counter()
            rx = sum(1 for x in list(self._rx) if now - x <= 3.0) / 3.0
            msg = {k: v for k, v in src.items() if k not in ("type",)}
            msg.update(type="status", fps=self.fps(), rx_fps=round(rx, 1), black=self._black,
                       engine=self.engine_state())
            if not src:
                msg.update(perc_ms=None, lag_s=None, source="phone", events=0, vlm="none", vlm_up=False)
            self.emit(msg)

    def stop(self):
        self._stop.set()
        if self.local is not None:
            self.local.stop()

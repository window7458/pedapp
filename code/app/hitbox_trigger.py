"""v5_29 히트박스 트리거 — 실시간 버전 (`--trigger hitbox`).

판정 본체는 hitbox_core.py (= hitbox_original_v5_29.py 를 hitbox_build.py 로 자동 변환, 줄 단위로 같음).
여기서는 원본이 '미리 저장한 파일'에서 읽던 입력을 프레임이 들어올 때마다 만들어 주고, 나온 사건을 말로 바꾼다.

  원본 입력 (pass1)                      실시간 공급
  ───────────────────────────────────    ────────────────────────────────────────────────
  dets_low.json  (YOLO conf 0.2 추적)    pipeline 의 YoloTracker (conf 0.2, 이 트리거면 자동) — fr.dets
  flows.npz      (배경 LK 광류 쌍)        FlowTracker  — pass1_detect.RecordingMotion 과 같은 계산, 매 프레임
  lights_hsv.json (신호등+상자 안 색)     LightWorker  — 별도 스레드, 3번째 프레임마다 (바쁘면 건너뜀)
  surface2.npz   (Mask2Former 바닥)       SurfaceWorker — 별도 스레드, 5번째 프레임마다 (GPU 권장)
  events_runtime.json (group_speech)     OnlineSpeech — 같은 규칙을 한 사건씩 처리 (0.5 s 모아 말하기 포함)

원본과 다른 점 (정직하게):
  · 바닥 분할 · 신호등은 비동기라 결과가 몇 프레임 늦게 반영된다 (원본은 같은 프레임 번호의 결과를 씀).
  · 사람 경고의 '붐빔' 걸러내기(v5_13)는 앞뒤 ±2 s 를 보던 것을 지난 2 s 만 본다.
  · 프레임 간격이 일정하지 않을 수 있다 (폰 ~10 fps). 판정은 원본과 같이 실제 시각(dt)으로 계산한다.
  · 신호등/바닥 모델이 없으면(torch 미설치 · GPU 없음) 그 입력만 빠진 채 돈다 — status 에 표시된다.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from core import Event
from hitbox_core import (CROWD_EV_N, CROWD_EV_S, GROUP_S, ROW_DX_M, ROW_GAP_S, V529Core, row_msg)

# 보행자에게 경고할 대상: 사람과 탈것(자동차·버스·오토바이·자전거·킥보드)만 (환경변수 HB_WARN_CLASSES 로 바꿀 수 있음, 쉼표 구분)
WARN_CLASSES = set(os.environ.get("HB_WARN_CLASSES", "person,car,bus,motorcycle,other_vehicle,bicycle,scooter").split(","))

# ───────────────────────── 말하기 (SJ 실행 시스템 dispatcher.clock_pieces 와 같은 문장) ─────────────────────────
KO = {"person": "사람", "bicycle": "자전거", "scooter": "킥보드", "motorcycle": "오토바이", "car": "자동차", "bus": "버스",
      "other_vehicle": "차량", "obstacle": "장애물", "stairs": "계단", "traffic_light": "신호등", "pole_long": "긴 기둥",
      "pole_short": "짧은 기둥", "pole": "기둥", "wall": "벽", "bollard": "볼라드"}
MOTION = {"static": "있어요", "approaching": "다가와요", "crossing": "지나가요", "receding": "멀어져요",
          "slow": "천천히 가요", "stopping": "멈추고 있어요", "standing": "서 있어요"}
ACTION = {"stop": "멈추세요.", "veer_left": "왼쪽으로 피하세요.", "veer_right": "오른쪽으로 피하세요.",
          "keep": "경로를 유지하세요."}


def josa(word: str) -> str:
    c = word[-1]
    return "이" if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else "가"


def compose(r: dict) -> str:
    """사건 하나(또는 묶음) → 한 문장. 예: '1시 방향 기둥, 11시 방향 자전거가 있어요.' / '정면 자동차가 다가와요. 멈추세요.'"""
    if r.get("text"):
        return r["text"]
    if r.get("scenario") == "S3":
        return ("신호등이 초록불로 바뀌었어요. 건너가세요." if r.get("state") == "GO"
                else "신호등이 빨간불로 바뀌었어요. 기다리세요.")
    items = r.get("items") or [dict(clock=r["clock"], cls=r["cls"], motion=r.get("motion"))]
    parts = []
    for i, it in enumerate(items):
        tgt = KO.get(it["cls"], "장애물")
        c = it["clock"]
        place = "정면" if c == 12 else "경로상에" if c == 0 else f"{c}시 방향"
        parts.append(place)
        parts.append(f"{tgt}{josa(tgt)} {MOTION.get(it.get('motion'), '있어요')}." if i == len(items) - 1 else f"{tgt},")
    if r.get("action") in ACTION:
        parts.append(ACTION[r["action"]])
    return " ".join(parts)


def prio_of(text: str) -> int:
    """멈추세요 3 > 피하세요 2 > 위치 안내 1 (SJ runtime/audio.py _prio 와 같음)."""
    return 3 if "멈추세요" in text else 2 if "피하세요" in text else 1


class OnlineSpeech:
    """hitbox_core.group_speech(merge_rows → calm_veer → 묶기) 를 사건이 오는 대로 처리하는 버전.
    같은 사건열을 넣으면 group_speech 와 같은 결과가 나온다 (tests/test_hitbox_port.py 가 확인)."""

    def __init__(self):
        self.rows, self.last_veer, self.pend, self.spoken = {}, None, [], {}

    # merge_rows 한 사건 분
    def _merge(self, r):
        side = ("right" if r.get("clock") in (1, 2, 3) else "left" if r.get("clock") in (9, 10, 11) else None)
        if (side is None or r["scenario"] != "S2" or r.get("motion") != "static"
                or r.get("action", "none") not in ("none", "keep") or r.get("lat") is None):
            return [r]
        row = self.rows.get(side)
        if row and r["t"] - row["t"] < ROW_GAP_S and abs(r["lat"] - row["lat"]) < ROW_DX_M:
            row.update(t=r["t"], lat=0.5 * (row["lat"] + r["lat"]), n=row["n"] + 1)
            if not row["said"]:
                row["said"] = True
                return [row_msg(r, side)]
            return []
        many = r.get("row_n", 0) >= 2
        self.rows[side] = dict(t=r["t"], lat=r["lat"], n=1, said=many)
        return [row_msg(r, side) if many else r]

    # calm_veer 한 사건 분
    def _calm(self, r):
        a = r.get("action", "none")
        if a in ("veer_left", "veer_right"):
            if self.last_veer and r["t"] - self.last_veer[0] < 1.5 and a != self.last_veer[1]:
                r["action"] = "none"
            else:
                self.last_veer = (r["t"], a)
        return r

    def _flush(self, out):
        if not self.pend:
            return
        items, seen_ = [], set()
        for r in sorted(self.pend, key=lambda r: r.get("dist") or 99.0):
            if (r["clock"], r["cls"]) not in seen_:
                seen_.add((r["clock"], r["cls"]))
                items.append(r)
        items = items[:3]
        out.append(dict(items[0], t=round(self.pend[0]["t"] + GROUP_S, 3), n_grouped=len(self.pend),
                        items=[dict(clock=r["clock"], cls=r["cls"], motion=r["motion"]) for r in items]))
        self.pend.clear()

    def feed(self, r0) -> list:
        out = []
        for r in self._merge(r0):
            r = self._calm(r)
            if self.pend and r["t"] - self.pend[0]["t"] > GROUP_S:
                self._flush(out)
            urgent = (r["scenario"] in ("S3", "S5") or r.get("action", "none") != "none"
                      or r.get("motion") in ("approaching", "crossing") or r.get("level") == "danger")
            if urgent:
                out.append(r)
                self.spoken[r["track_id"]] = r["t"]
                self.pend[:] = [p for p in self.pend if p["track_id"] != r["track_id"]]
            elif r["t"] - self.spoken.get(r["track_id"], -1e9) > 4.0 and all(p["track_id"] != r["track_id"] for p in self.pend):
                self.pend.append(r)
                self.spoken[r["track_id"]] = r["t"]
        return out

    def tick(self, t) -> list:
        """사건이 없어도 모으기 시간(GROUP_S)이 지나면 내보냄. 원본은 영상 끝에서 flush."""
        out = []
        if self.pend and t - self.pend[0]["t"] >= GROUP_S:
            self._flush(out)
        return out

    def finish(self) -> list:
        out = []
        self._flush(out)
        return out


# ───────────────────────── 입력 공급 ─────────────────────────
class FlowTracker:
    """pass1_detect.RecordingMotion 과 같은 배경 특징점 쌍 (직전 프레임 → 지금, 원본 좌표). 박스 안쪽은 마스크로 뺀다."""
    GMC_IMAGE_SIZE = 640

    def __init__(self):
        self.prev_gray = self.prev_mask = None

    def update(self, frame, boxes):
        h, w = frame.shape[:2]
        scale = min(1.0, self.GMC_IMAGE_SIZE / max(h, w))
        sw, sh = max(2, round(w * scale)), max(2, round(h * scale))
        gray = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2GRAY)
        mask = np.full((sh, sw), 255, np.uint8)
        sx, sy = sw / w, sh / h
        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            mask[max(0, int(y1 * sy) - 5):min(sh, int(np.ceil(y2 * sy)) + 5),
                 max(0, int(x1 * sx) - 5):min(sw, int(np.ceil(x2 * sx)) + 5)] = 0
        pairs = None
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, 500, 0.01, 10, mask=self.prev_mask, blockSize=7)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
                if p1 is not None:
                    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, winSize=(21, 21), maxLevel=3)
                    a, b = p0.reshape(-1, 2), p1.reshape(-1, 2)
                    fb = np.linalg.norm(back.reshape(-1, 2) - a, axis=1)
                    ok = st.ravel().astype(bool) & st2.ravel().astype(bool) & (fb <= 1.5)
                    pairs = np.hstack([a[ok] / [sx, sy], b[ok] / [sx, sy]]).astype(np.float32)
        self.prev_gray, self.prev_mask = gray, mask
        return pairs


class _Worker(threading.Thread):
    """가장 최근 한 장만 처리하는 배경 스레드. 바쁘면 새 작업은 버린다 (실시간이 밀리지 않게)."""

    def __init__(self, name, log):
        super().__init__(daemon=True, name=name)
        self.log = log
        self.cv = threading.Condition()
        self.job, self.busy, self.ready, self.dead = None, False, False, False
        self.result, self.seq, self.seen = None, 0, 0
        self.last_ms = None

    def offer(self, job) -> bool:
        if self.dead:
            return False
        with self.cv:
            if self.busy or self.job is not None:
                return False
            self.job = job
            self.cv.notify()
        return True

    def run(self):
        while True:
            with self.cv:
                while self.job is None:
                    self.cv.wait()
                job, self.job, self.busy = self.job, None, True
            t0 = time.perf_counter()
            try:
                res = self.work(job)
                with self.cv:
                    self.result, self.seq = res, self.seq + 1
                self.last_ms = round((time.perf_counter() - t0) * 1000)
            except Exception as e:
                if not self.ready:                      # 모델을 못 올림 → 이 입력은 포기 (나머지는 계속)
                    self.dead = True
                    self.log(f"[hitbox] {self.name} 사용 불가: {e!r}")
                    return
                self.log(f"[hitbox] {self.name} 오류: {e!r}")
            finally:
                self.busy = False

    def work(self, job):
        raise NotImplementedError

    def poll(self):
        if self.seq != self.seen:
            self.seen = self.seq
            return self.result
        return None


LIT_V, LIT_S, MIN_FRAC, DOM = 140, 70, 0.015, 2.0       # pass1_lightcolor.py 와 같은 값


def color_of(crop):
    """신호등 상자 안 '밝고 진한' 빨강·초록 픽셀 비율로 색을 정함 (pass1_lightcolor.color_of 그대로)."""
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (v >= LIT_V) & (s >= LIT_S)
    red = (lit & ((h <= 10) | (h >= 165))).mean()
    green = (lit & (h >= 40) & (h <= 95)).mean()
    if red >= MIN_FRAC and red >= DOM * green:
        return "red"
    if green >= MIN_FRAC and green >= DOM * red:
        return "green"
    return None


class LightWorker(_Worker):
    """pass1_lights + pass1_lightcolor: traffic_light 만 큰 해상도로 추적하고 상자 안 색으로 red/green/off."""

    def __init__(self, weights, device, log):
        super().__init__("lights", log)
        self.weights, self.device = weights, device
        self.imgsz = int(os.environ.get("LIGHT_IMGSZ", 1280))
        self.conf = float(os.environ.get("LIGHT_CONF", 0.25))
        self.model = self.light_id = None

    def work(self, job):
        _, image = job
        if self.model is None:
            from ultralytics import YOLO
            m = YOLO(str(self.weights))
            ids = [k for k, v in m.names.items() if v == "traffic_light"]
            if not ids:
                raise RuntimeError("이 가중치에 traffic_light 클래스가 없음")
            self.model, self.light_id, self.ready = m, ids[0], True
        r = self.model.track(image, classes=[self.light_id], imgsz=self.imgsz, conf=self.conf, persist=True,
                             verbose=False, tracker="bytetrack.yaml", device=self.device)[0]
        H, W = image.shape[:2]
        lights = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = (round(v, 1) for v in b.xyxy[0].tolist())
                crop = image[max(0, int(round(y1))):min(H, int(round(y2))), max(0, int(round(x1))):min(W, int(round(x2)))]
                st = color_of(crop)
                lights.append(dict(box=[x1, y1, x2, y2], conf=round(float(b.conf[0]), 3),
                                   tid=int(b.id[0]) if b.id is not None else None,
                                   state=st or "off", p=1.0 if st else 0.0))
        return lights


class SurfaceWorker(_Worker):
    """pass1_surface: Mask2Former(Mapillary) → 0 기타 · 1 인도 · 2 차도 · 3 횡단보도 · 4 연석 · 5 고정 장애물 · 6 벽/건물."""
    MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    GROUPS = {1: ("Sidewalk", "Pedestrian Area", "Curb Cut"),
              2: ("Road", "Parking", "Bike Lane", "Service Lane", "Rail Track"),
              3: ("Crosswalk - Plain", "Lane Marking - Crosswalk"),
              4: ("Curb",),
              5: ("Pole", "Utility Pole", "Street Light", "Traffic Sign Frame", "Bench", "Bike Rack", "Fire Hydrant",
                  "Junction Box", "Mailbox", "Phone Booth", "Trash Can", "Barrier", "Fence", "Guard Rail", "Pothole",
                  "Vegetation", "Terrain"),
              6: ("Wall", "Building")}

    def __init__(self, device, log):
        super().__init__("surface", log)
        self.device = device
        self.net = self.proc = self.lut = None
        self.t_res = -1e9

    def work(self, job):
        t, image = job
        import torch
        if self.net is None:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            dev = "cpu" if str(self.device) == "cpu" else (f"cuda:{self.device}" if str(self.device).isdigit() else str(self.device))
            self.proc = AutoImageProcessor.from_pretrained(self.MODEL)
            net = Mask2FormerForUniversalSegmentation.from_pretrained(self.MODEL).to(dev).eval()
            if dev != "cpu":
                net = net.half()
            self.lut = np.zeros(len(net.config.id2label), np.uint8)
            for g, names in self.GROUPS.items():
                for k, v in net.config.id2label.items():
                    if v in names:
                        self.lut[k] = g
            self.net, self.dev, self.ready = net, dev, True
        H, W = image.shape[:2]
        portrait = H >= W
        small = cv2.resize(image, (540, 960) if portrait else (960, 540))
        out_hw = (240, 135) if portrait else (135, 240)
        x = self.proc(images=np.ascontiguousarray(small[:, :, ::-1]), return_tensors="pt").to(self.dev)
        if self.dev != "cpu":
            x["pixel_values"] = x["pixel_values"].half()
        with torch.no_grad():
            o = self.net(**x)
        lab = self.proc.post_process_semantic_segmentation(o, target_sizes=[out_hw])[0].cpu().numpy()
        self.t_res = t
        return self.lut[lab]

    def latest(self, now, stale_s):
        return self.result if (self.result is not None and now - self.t_res <= stale_s) else None


# ───────────────────────── 트리거 ─────────────────────────
class HitboxTrigger:
    name = "hitbox_v5_29"
    self_deduped = True         # 트랙 쿨다운 · 같은 문장 억제는 이 트리거가 이미 함 (pipeline 이 또 하지 않음)
    wants_vlm = False           # 문장이 이미 위치 · 움직임 · 행동까지 말함 → 경고마다 VLM 설명을 덧붙이지 않음
    needs_conf = 0.2            # 저신뢰(0.2) 검출 필요 — pipeline 이 YoloTracker conf 를 이 값으로 맞춤

    def __init__(self, cfg=None, log=print):
        self.log = log
        self.core: V529Core | None = None
        self.flow = FlowTracker()
        self.speech = OnlineSpeech()
        self.person_t = deque(maxlen=16)
        self.dt_hist = deque(maxlen=30)
        self.prev_t = None
        self.info = {}
        self._err_t = -1e9
        self.surf_stale = float(os.environ.get("HB_SURF_STALE", 6.0))
        self.lights = self.surface = None
        weights = None
        if cfg is not None:
            from common import WEIGHTS
            w = WEIGHTS.get(cfg.weights) or Path(cfg.weights)
            weights, device = w, cfg.device
            if os.environ.get("HB_LIGHTS", "1") != "0" and Path(weights).exists():
                self.lights = LightWorker(weights, device, log)
                self.lights.start()
            if os.environ.get("HB_SURFACE", "1") != "0":
                self.surface = SurfaceWorker(device, log)
                self.surface.start()
        self.n = 0

    # fr: core.Frame (idx, t, image, dets[Det], W, H)
    def step(self, fr):
        try:
            return self._step(fr)
        except Exception:
            if time.time() - self._err_t > 5.0:
                self._err_t = time.time()
                self.log("[hitbox] 오류\n" + traceback.format_exc())
            return []

    def _step(self, fr):
        W, H, t = fr.W, fr.H, fr.t
        if self.core is None or (self.core.W, self.core.H) != (W, H):
            self.core = V529Core(W, H)
            self.flow = FlowTracker()
        if self.prev_t is not None and t > self.prev_t:
            self.dt_hist.append(t - self.prev_t)
        self.prev_t = t
        fps = float(np.clip(1.0 / np.median(self.dt_hist), 3.0, 30.0)) if len(self.dt_hist) >= 5 else 10.0
        dets = [dict(box=[float(v) for v in d.box], cls=d.cls, conf=float(d.conf), tid=d.track_id)
                for d in fr.dets if d.cls != "traffic_light"]
        pairs = self.flow.update(fr.image, [d["box"] for d in dets if d["conf"] >= 0.5])
        self.n += 1
        if self.lights is not None and self.n % 3 == 1:
            self.lights.offer((t, fr.image))
        if self.surface is not None and self.n % 5 == 1:
            self.surface.offer((t, fr.image))
        new_lights = self.lights.poll() if self.lights is not None else None
        surf = self.surface.latest(t, self.surf_stale) if self.surface is not None else None

        rt, pf = self.core.step_frame(t, fr.idx, fps, dets, pairs, new_lights, surf)

        out = []
        for r in sorted(rt, key=lambda r: r["t"]):
            if r.get("cls") not in WARN_CLASSES:                 # 사람·차량 외(기둥·신호등 등)는 보행자에게 말하지 않음
                continue
            if r.get("cls") == "person":                         # 붐빌 때 사람 경고 빼기 (v5_13, 지난 2 s 만 봄)
                self.person_t.append(r["t"])
                if sum(r["t"] - u <= CROWD_EV_S for u in self.person_t) >= CROWD_EV_N:
                    continue
            out += self.speech.feed(r)
        out += self.speech.tick(t)

        self.info = dict(walk=round(pf["walk"], 2), road=pf["road"] or "-", heading=round(pf["heading"], 1),
                         surf=pf["surf"], zone_z=round(pf["zone_z"], 1), tracks=len(self.core.tracks),
                         lights=("on" if self.lights is not None and not self.lights.dead else "off"),
                         surface=("on" if self.surface is not None and not self.surface.dead else "off"),
                         light_ms=getattr(self.lights, "last_ms", None), surface_ms=getattr(self.surface, "last_ms", None),
                         fps_est=round(fps, 1))
        events = []
        for r in out:
            text = compose(r)
            events.append(Event(t=t, cls=r["cls"], scenario=r["scenario"], track_id=r.get("track_id"),
                                box=tuple(r["box"]) if r.get("box") is not None else None, W=W, H=H,
                                extra=dict(text=text, prio=prio_of(text), motion=r.get("motion"), action=r.get("action"),
                                           clock=r.get("clock"), level=r.get("level"), items=r.get("items"),
                                           state=r.get("state"))))
        return events

"""점자블록 따라 안내 (11/24 밤 개정): 따로 화면을 열지 않고, 평소 안내 화면 그대로 이어서 추론한다.

- 찾기(scout): 평소 안내 중 2초마다 최신 한 장을 점자블록 모델(GuideTWSI YOLO11n-seg · 640 · 신뢰도 0.5)로 본다.
  화면 아래 60 %(땅)에서 1 % 넘게 두 번 이어 보이면 → 따라가기 시작.
- 따라가기(follow): 0.6초마다 한 장씩 계속 추론해 점자블록이 왼쪽 · 정면 · 오른쪽 어디 있는지 알린다.
  정면이면 점자블록 모양도 본다: 곧게 이어지면 "앞으로 쭉 가세요", 먼 쪽이 꺾이면 "앞에서 왼쪽으로 꺾여요. 왼쪽으로 트세요".
  방향이 바뀔 때만 말한다(같은 방향이 두 번 이어져야 바뀐 것으로 봄). 장애물 경고 · 판단 고리는 그대로 돈다 (스레드라 늦어지지 않음).
  4초 동안 안 보이면 "점자블록이 안 보여요" 한 번 말하고 찾기로 돌아간다.
- 폰 명령: start(점자블록 찾아줘 — 바로 따라가기, 15초 못 찾으면 알림) · stop(그만 — 3분 동안 자동 시작 안 함).
- 말은 모두 SENTENCES 에 있는 고정 문장 → 서버가 켜질 때 구글 음성으로 미리 만들어 둔다 (app_tts).
- TACTILE_SCOUT=0 이면 끈다.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

SCOUT_EVERY_S, FOLLOW_EVERY_S = 2.0, 0.6
NEED, MIN_AREA = 2, 0.01
LOST_S, MANUAL_GIVEUP_S = 4.0, 15.0
COOL_LOST_S, COOL_STOP_S = 20.0, 180.0
REPEAT_S = 12.0                     # 같은 방향을 오래 유지하면 가끔 다시 확인해 준다 (정면은 말하지 않음)

SAY = {
    "found":   "점자블록이 보여요. 따라가며 안내할게요.",
    "center":  "점자블록 위예요. 그대로 가세요.",
    "straight": "앞으로 쭉 가세요.",
    "curve_left":  "앞에서 점자블록이 왼쪽으로 꺾여요. 천천히 왼쪽으로 트세요.",
    "curve_right": "앞에서 점자블록이 오른쪽으로 꺾여요. 천천히 오른쪽으로 트세요.",
    "left":    "점자블록이 왼쪽이에요. 조금 왼쪽으로.",
    "right":   "점자블록이 오른쪽이에요. 조금 오른쪽으로.",
    "lost":    "점자블록이 안 보여요. 평소 안내를 계속할게요.",
    "search":  "점자블록을 찾고 있어요. 앞쪽 바닥을 비춰 주세요.",
    "giveup":  "점자블록을 찾지 못했어요. 평소 안내를 계속할게요.",
    "stop":    "점자블록 안내를 껐어요.",
    "already": "이미 점자블록을 따라 안내하고 있어요.",
}
SENTENCES = list(SAY.values())


def shape_of(comp):
    """점자블록 한 덩어리가 앞으로 어떻게 이어지는지 (카메라는 몸 정면): straight · curve_left · curve_right, 모르면 None.
    덩어리를 아래(가까운 쪽) → 위(먼 쪽)로 세 토막 내어 토막마다 가운데 x 를 본다.
    먼 토막이 가까운 토막보다 화면 폭 15 % 넘게 옆으로 가 있으면 그쪽으로 꺾임, 10 % 안이면 곧게 이어짐."""
    m = comp["mask"]
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    H, W = m.shape
    top, bot = ys.min(), ys.max()
    if (bot - top) / H < .22:               # 앞으로 길게 안 보이면 (가로로 놓인 조각 등) 모양을 말하지 않는다
        return None
    cut1, cut2 = top + (bot - top) / 3, top + 2 * (bot - top) / 3
    far, near = xs[ys <= cut1], xs[ys >= cut2]
    if len(far) < 15 or len(near) < 15:
        return None
    dx = (np.median(far) - np.median(near)) / W
    if dx < -.15: return "curve_left"
    if dx > .15: return "curve_right"
    if abs(dx) < .10: return "straight"
    return None


def direction_of(mask):
    """점자블록이 지금 어디 있는지 + 앞으로 어떻게 이어지는지.
    반환 (안내 키, target): 옆으로 치우쳤으면 left · right, 정면이면 모양(straight · curve_left · curve_right, 모르면 center).
    덩어리 고르기는 임태규 guide.components 그대로."""
    from tactile_feature.guide import components
    choices = components(mask)
    if not choices:
        return None, None
    c = max(choices, key=lambda c: c["bottom"] - .25 * abs(c["target"][0] - .5))
    x = c["target"][0]
    side = "left" if x < .38 else "right" if x > .62 else "center"
    if side == "center":
        side = shape_of(c) or "center"
    return side, c["target"]


class TactileScout:
    def __init__(self, device, emit, log=print):
        self.device, self.emit, self.log = str(device), emit, log
        self.on = os.environ.get("TACTILE_SCOUT", "1") != "0"
        self.lock = threading.Lock()
        self.busy = False
        self.at = 0.0
        self.hits = 0
        self.cool_until = 0.0
        self.err_at = 0.0
        self._off()

    def _off(self):
        self.following = False
        self.manual_at = 0.0          # 0 = 자동, 아니면 "찾아줘" 누른 시각
        self.dir = None               # 지금 알린 방향
        self.cand, self.cand_n = None, 0
        self.seen_at = 0.0
        self.said_at = 0.0

    # ── 폰 명령 ──
    def command(self, action):
        now = time.monotonic()
        with self.lock:
            if action == "start":
                if self.following:
                    return self._say("already", state="on")
                self._off()
                self.following, self.manual_at, self.seen_at, self.cool_until = True, now, now, 0.0
                return self._say("search", state="on")
            if action == "stop":
                was = self.following
                self._off(); self.hits = 0
                self.cool_until = now + COOL_STOP_S
                return self._say("stop", state="off") if was else self.emit({"type": "tactile_follow", "state": "off"})

    def reset(self):
        """보행자 연결이 끊기면 조용히 끈다 (다시 붙으면 찾기부터)."""
        with self.lock:
            self._off(); self.hits = 0

    def _say(self, key, **kw):
        self.said_at = time.monotonic()
        self.emit({"type": "tactile_follow", "say": SAY[key], "key": key, **kw})

    # ── 판단 고리가 매 장 부른다 ──
    def offer(self, image, capture_ms=None):
        now = time.monotonic()
        every = FOLLOW_EVERY_S if self.following else SCOUT_EVERY_S
        if not self.on or self.busy or now - self.at < every:
            return
        if not self.following and now < self.cool_until:
            return
        self.busy, self.at = True, now
        threading.Thread(target=self._look, args=(image.copy(), capture_ms), daemon=True).start()

    def _infer(self, image):
        from tactile_feature import runtime
        from PIL import Image
        with runtime._MODEL_LOCK:
            if self.device not in runtime._MODELS:
                from pathlib import Path
                from tactile_feature.segment_tactile import TactileSegmenter
                w = Path(runtime.__file__).resolve().parents[1] / "weights" / "yolo11n_tactile.pt"
                runtime._MODELS[self.device] = TactileSegmenter(weights=w, device=self.device, threads=2)
            model = runtime._MODELS[self.device]
            r = model.predict(Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])), imgsz=640, conf=0.5, mode="single")
        return np.asarray(r.get("mask"), dtype=bool)

    def _look(self, image, capture_ms):
        try:
            mask = self._infer(image)
            h = mask.shape[0] if mask.ndim == 2 else 0
            with self.lock:
                if self.following:
                    self._follow(mask, image.shape, capture_ms)
                else:
                    area = float(mask[int(h * 0.4):].mean()) if h else 0.0
                    self.hits = self.hits + 1 if area >= MIN_AREA else 0
                    if self.hits >= NEED:
                        self.hits = 0
                        self._off()
                        now = time.monotonic()
                        self.following, self.seen_at = True, now
                        d, _ = direction_of(mask)
                        self.dir = d
                        self._say("found", state="on", dir=d)
                        if d and d != "center":
                            self._say(d, state="on", dir=d)
        except Exception as e:                         # 이 기능이 장애물 안내를 멈추게 하면 안 된다
            if time.monotonic() - self.err_at > 30:
                self.err_at = time.monotonic()
                self.log(f"[tactile_scout] {type(e).__name__}: {e}")
        finally:
            self.busy = False

    def _follow(self, mask, shape, capture_ms):
        now = time.monotonic()
        d, target = direction_of(mask) if mask.any() else (None, None)
        from tactile_feature.runtime import mask_png
        base = {"type": "tactile_follow", "state": "on", "W": shape[1], "H": shape[0], "c": capture_ms,
                "mask_png": mask_png(mask) if d else None, "target": target}
        if d is None:
            limit = MANUAL_GIVEUP_S if (self.manual_at and self.dir is None) else LOST_S
            if now - self.seen_at > limit:
                key = "giveup" if self.dir is None else "lost"
                self._off(); self.hits = 0
                self.cool_until = now + COOL_LOST_S
                self._say(key, state="off")
            else:
                self.emit({**base, "dir": None})
            return
        self.seen_at = now
        first = self.dir is None
        if d == self.dir:
            self.cand, self.cand_n = None, 0
            if d not in ("center", "straight") and now - self.said_at > REPEAT_S:
                return self._say(d, **{**base, "dir": d})
            return self.emit({**base, "dir": d})
        # center ↔ straight 는 같은 뜻 (정면) → 말 바꾸지 않음
        if {d, self.dir} == {"center", "straight"}:
            self.dir = d if d == "straight" else self.dir
            return self.emit({**base, "dir": self.dir})
        # 방향이 바뀐 것 같으면 두 번 이어 볼 때만 바꾼다 (한 장 흔들림에 말이 바뀌지 않게). 처음 찾은 순간은 바로.
        self.cand_n = self.cand_n + 1 if self.cand == d else 1
        self.cand = d
        if first or self.cand_n >= 2:
            self.dir, self.cand, self.cand_n = d, None, 0
            if first and self.manual_at:
                self._say("found", **{**base, "dir": d})
                if d != "center":
                    self._say(d, **{**base, "dir": d})
            else:
                self._say(d, **{**base, "dir": d})
            return
        self.emit({**base, "dir": self.dir})

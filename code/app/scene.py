"""특별 상황 묘사 — 위험은 아니지만 시각장애인이 알면 좋은 장면(불꽃놀이, 벚꽃길 등)을 멀티모달 모델로 설명한다.

위험(차량 · 장애물 · 신호)은 Trigger 쪽이 맡는다. 이 모듈은 '지금 눈앞에 이런 풍경이 있다'만 말한다.

동작
  · 프레임을 1 s 간격으로 모아 두고, 화면이 충분히 바뀌었거나 오래 안 물어봤으면 최근 3장(약 2 s 간격)을 모델에 보낸다.
  · 모델은 JSON {special, kind, description, new_info} 로 답한다. special=false면 아무 말도 안 한다.
  · 같은 종류(kind)는 new_info=true이거나 repeat_s가 지난 뒤에만 다시 말한다 (벚꽃길을 걷는 내내 반복하지 않게).
  · 모델 호출은 별도 스레드. 호출이 길어져도 경고 경로는 막히지 않는다.

백엔드
  mock       : 서버 · API 없이 UI 시험용 (벚꽃길 → 불꽃놀이 → 없음 순환)
  openai     : OpenAI 호환 서버 (팀의 vLLM Qwen2.5-VL 서버 그대로 쓸 수 있음)
  anthropic  : Claude API (환경변수 ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

KIND_KO = {
    "fireworks": "불꽃놀이", "cherry_blossom": "벚꽃길", "flowers": "꽃길", "autumn_leaves": "단풍길",
    "festival": "축제", "crowd": "사람이 많은 곳", "performance": "공연", "rainbow": "무지개",
    "sunset": "노을", "sunrise": "일출", "snow": "눈", "fountain": "분수", "night_view": "야경",
    "lights": "조명 장식", "landmark": "눈에 띄는 건물", "water": "물가", "other": "특별한 장면",
}

SYSTEM = (
    "You are the scene description module of a walking assistant for a blind pedestrian. "
    "The images are consecutive frames from a camera worn on the body and facing the walking direction "
    "(oldest first, about 1-2 seconds apart, the last image is the present). "
    "Collision hazards such as vehicles, obstacles, stairs and traffic lights are handled by another module, "
    "so never warn about them and never give walking instructions. "
    "Your only job is to notice a special situation that a blind person would want to know about "
    "because they cannot see it: fireworks, a street lined with cherry blossoms or other flowers or autumn leaves, "
    "a festival, a crowd, a performance, a rainbow, sunset or sunrise, snow, a fountain, an illuminated night view, "
    "or a striking landmark. Ordinary streets, buildings and cars are NOT special."
)

PROMPT = (
    "Reply with ONLY one JSON object, no other text:\n"
    '{"special": true|false, "kind": "<one of: %s>", '
    '"description": "<Korean>", "new_info": true|false}\n'
    "Rules for description: at most two short sentences in a warm spoken Korean style ending in ~요. "
    "Say where it is (왼쪽, 정면, 오른쪽, 멀리, 가까이) and, when the frames show it, what is changing "
    "(for example 연달아 터져요, 계속 이어져 있어요). Describe only what is visible. No guesses about people's feelings.\n"
    "special=false when nothing special is visible. "
    "new_info=true only if this adds something that the previously announced description did not say.\n"
    "Previously announced: %s"
)


def parse_json(text: str):
    if not text:
        return None
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def to_jpeg_b64(im: np.ndarray, long_side: int = 768, quality: int = 80) -> str:
    s = min(1.0, long_side / max(im.shape[:2]))
    if s < 1:
        im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def build_prompt(previous: str | None) -> str:
    return PROMPT % (", ".join(KIND_KO), previous or "none")


class MockScene:
    name = "mock"
    CANNED = [
        {"special": True, "kind": "cherry_blossom", "new_info": True,
         "description": "정면으로 벚꽃길이 길게 이어져 있어요. 길 양옆에 분홍색 벚꽃이 활짝 피어 있어요."},
        {"special": True, "kind": "fireworks", "new_info": True,
         "description": "하늘 높은 곳에서 불꽃놀이가 터지고 있어요. 오른쪽 위에서 불꽃이 연달아 터져요."},
        {"special": False, "kind": "other", "description": "", "new_info": False},
    ]

    def __init__(self, latency: float = 0.8):
        self.i, self.latency = 0, latency

    def __call__(self, images, previous):
        time.sleep(self.latency)
        r = self.CANNED[self.i % len(self.CANNED)]
        self.i += 1
        return r


class OpenAIScene:
    """OpenAI 호환 서버 (vLLM Qwen2.5-VL 등)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY", "local"))
        self.model = model
        self.name = f"openai:{model}"

    def __call__(self, images, previous):
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + to_jpeg_b64(im)}}
                   for im in images]
        content.append({"type": "text", "text": build_prompt(previous)})
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.2, max_tokens=300,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        return parse_json(r.choices[0].message.content)


class AnthropicScene:
    """Claude API. 모델은 SCENE_MODEL 환경변수 또는 인자로 지정."""

    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("SCENE_MODEL", "claude-sonnet-5-5")
        self.name = f"anthropic:{self.model}"

    def __call__(self, images, previous):
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                "data": to_jpeg_b64(im)}} for im in images]
        content.append({"type": "text", "text": build_prompt(previous)})
        r = self.client.messages.create(model=self.model, max_tokens=300, system=SYSTEM,
                                        messages=[{"role": "user", "content": content}])
        return parse_json("".join(b.text for b in r.content if getattr(b, "type", "") == "text"))


def make_backend(kind: str, url: str | None = None, model: str | None = None):
    if kind == "none":
        return None
    if kind == "mock":
        return MockScene()
    if kind == "openai":
        return OpenAIScene(url or "http://localhost:8000/v1", model or "Qwen/Qwen2.5-VL-7B-Instruct")
    if kind == "anthropic":
        return AnthropicScene(model)
    raise ValueError(f"unknown scene backend {kind} (none | mock | openai | anthropic)")


class SceneDescriber(threading.Thread):
    def __init__(self, backend, emit, min_interval=5.0, max_interval=12.0, change_thr=14.0,
                 repeat_s=30.0, n_frames=3, span_s=2.0, log=print):
        super().__init__(daemon=True)
        self.backend, self.emit, self.log = backend, emit, log
        self.min_interval, self.max_interval, self.change_thr = min_interval, max_interval, change_thr
        self.repeat_s, self.n_frames, self.span_s = repeat_s, n_frames, span_s
        self.enabled = True
        self.lock = threading.Lock()
        self.frames: deque = deque()          # (t, image)
        self.sig = None
        self.latest_t = -1e9
        self.last_call_t, self.last_sig = -1e9, None
        self.last_kind, self.last_text, self.last_emit_t, self.quiet = None, None, -1e9, 0
        self.stopped = threading.Event()
        self.name = "scene:" + getattr(backend, "name", "none")
        self.busy = False

    def push(self, t: float, image: np.ndarray):
        """처리 스레드에서 프레임마다 호출 (가벼움)."""
        g = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
        with self.lock:
            self.latest_t, self.sig = t, g.astype(np.float32)
            if not self.frames or t - self.frames[-1][0] >= 0.5:
                s = min(1.0, 768 / max(image.shape[:2]))
                small = cv2.resize(image, (round(image.shape[1] * s), round(image.shape[0] * s))) if s < 1 else image.copy()
                self.frames.append((t, small))
            while self.frames and t - self.frames[0][0] > 8.0:
                self.frames.popleft()

    def set_enabled(self, on: bool):
        self.enabled = bool(on)

    def _pick(self):
        with self.lock:
            if not self.frames:
                return None
            t_now = self.frames[-1][0]
            want = [t_now - self.span_s * (self.n_frames - 1 - k) / max(1, self.n_frames - 1) for k in range(self.n_frames)]
            out, used = [], set()
            for w in want:
                best = min(range(len(self.frames)), key=lambda i: abs(self.frames[i][0] - w))
                if best not in used:
                    used.add(best)
                    out.append(self.frames[best][1])
            return out, t_now

    def _due(self) -> bool:
        with self.lock:
            if self.sig is None or self.latest_t < 0:
                return False
            dt = self.latest_t - self.last_call_t
            if dt < self.min_interval:
                return False
            if dt >= self.max_interval or self.last_sig is None:
                return True
            return float(np.mean(np.abs(self.sig - self.last_sig))) >= self.change_thr

    def run(self):
        while not self.stopped.is_set():
            time.sleep(0.3)
            if not self.enabled or self.backend is None or not self._due():
                continue
            got = self._pick()
            if not got:
                continue
            imgs, t_now = got
            with self.lock:
                self.last_call_t, self.last_sig = self.latest_t, None if self.sig is None else self.sig.copy()
            self.busy = True
            t0 = time.perf_counter()
            try:
                res = self.backend(imgs, self.last_text)
            except Exception as e:            # 모델 오류가 다른 경로를 막으면 안 된다
                self.log(f"[scene] backend error: {e!r}")
                self.emit({"type": "scene_error", "error": repr(e)[:200]})
                res = None
            finally:
                self.busy = False
            self._emit_result(res, t_now, time.perf_counter() - t0)

    def _emit_result(self, res, t_now: float, took: float):
        if not isinstance(res, dict):
            return
        if not res.get("special"):
            self.quiet += 1
            if self.quiet >= 2:
                self.last_kind = None
            return
        self.quiet = 0
        text = str(res.get("description") or "").strip()
        kind = str(res.get("kind") or "other")
        if not text:
            return
        now = self.latest_t
        fresh = kind != self.last_kind or bool(res.get("new_info")) or now - self.last_emit_t >= self.repeat_s
        if not fresh:
            return
        self.last_kind, self.last_text, self.last_emit_t = kind, text, now
        self.emit({"type": "scene", "kind": kind, "label": KIND_KO.get(kind, KIND_KO["other"]), "text": text,
                   "t": round(t_now, 2), "took_s": round(took, 2)})

    def stop(self):
        self.stopped.set()

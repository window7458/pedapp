"""음성 인식 (말 → 글자) — Whisper large-v3-turbo 를 서버 GPU 에서.

폰 브라우저의 음성 인식(크롬 = 구글, 사파리 = 애플)은 앱 안 브라우저(카카오톡 등)에서 잘 안 되고, 다시 켤 때마다
마이크 허락을 묻고, 경고 소리에 자주 끊겼다. 그래서 폰은 마이크를 한 번 켜 두고 말한 구간(16 kHz PCM)만 보내고,
서버가 글자로 바꾼다.

- 모델: openai/whisper-large-v3-turbo (MIT). 한국어 고정. GPU 면 fp16, 없으면 CPU(느림, 개발용).
- 앱 명령 · 버스 번호 · 역 이름을 잘 받게 앞말(prompt)을 준다.
- 조용한 곳에서 Whisper 가 지어내는 말("시청해 주셔서 감사합니다" 같은 것)은 버린다.
- 위치: 환경변수 STT_MODEL (기본: 이미지에 넣어 둔 /srv/models/whisper-large-v3-turbo, 없으면 허깅페이스에서 받음).
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

import numpy as np

SR = 16000
MAX_S = 15.0
LOCAL = Path("/srv/models/whisper-large-v3-turbo")
PROMPT = ("세이아이, 시작, 그만, 다시 말해줘, 도와줘, 511번 버스 언제 와, 오백십일번 버스 탈게, 인하대역 가자, "
          "지하철 언제 와, 엘리베이터 어디야, 주변 설명해줘, 길 건널게, 실내 모드 켜줘.")
# 소리가 거의 없을 때 Whisper 가 자주 지어내는 말
_HALLU = re.compile(r"시청해\s*주셔서|구독|좋아요\s*와|MBC|KBS|SBS|자막|뉴스|^\s*(감사합니다|고맙습니다)[.!]?\s*$|^[\s.,!?…]*$")

_lock = threading.Lock()
_model = None
_proc = None
_dev = "cpu"
_prompt_ids = None
_err: str | None = None


def _source() -> str:
    env = os.environ.get("STT_MODEL")
    if env:
        return env
    return str(LOCAL) if (LOCAL / "config.json").exists() else "openai/whisper-large-v3-turbo"


def load() -> bool:
    """처음 한 번 모델을 올린다 (GPU 약 3초 · 1.6 GB). 실패하면 False → 폰은 브라우저 음성 인식으로 돌아간다."""
    global _model, _proc, _dev, _prompt_ids, _err
    if _model is not None:
        return True
    if _err:
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            src = _source()
            _dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if _dev == "cuda" else torch.float32
            t0 = time.time()
            _proc = WhisperProcessor.from_pretrained(src)
            m = WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=dtype, low_cpu_mem_usage=True)
            _model = m.to(_dev).eval()
            _prompt_ids = torch.tensor(_proc.get_prompt_ids(PROMPT), device=_dev)
            print(f"[stt] {src} · {_dev} · {time.time() - t0:.1f}초", flush=True)
            return True
        except Exception as e:  # 모델이 없거나 메모리 부족 → 브라우저 인식으로
            _err = repr(e)
            print(f"[stt] 못 켬: {_err}", flush=True)
            return False


def ready() -> bool:
    return _model is not None


def transcribe(pcm16: bytes) -> dict:
    """16 kHz · 모노 · int16 PCM → {"text", "ms"}. 너무 조용하거나 지어낸 말이면 text 는 ''."""
    a = np.frombuffer(pcm16[: int(MAX_S * SR) * 2], dtype="<i2").astype(np.float32) / 32768.0
    if a.size < SR * 0.25:
        return {"text": "", "ms": 0, "why": "짧음"}
    if float(np.sqrt(np.mean(a * a))) < 0.003:
        return {"text": "", "ms": 0, "why": "조용함"}
    if not load():
        raise RuntimeError(_err or "stt unavailable")
    import torch
    t0 = time.time()
    with _lock:  # GPU 하나를 여럿이 같이 쓰므로 한 번에 하나씩
        feats = _proc(a, sampling_rate=SR, return_tensors="pt").input_features.to(_dev, _model.dtype)
        with torch.inference_mode():
            ids = _model.generate(feats, language="ko", task="transcribe", prompt_ids=_prompt_ids,
                                  max_new_tokens=96, num_beams=1, do_sample=False)
        text = _proc.batch_decode(ids, skip_special_tokens=True)[0]
    text = text.replace(PROMPT, "").strip()           # 앞말이 결과에 섞여 나오는 경우
    ms = int((time.time() - t0) * 1000)
    if _HALLU.search(text):
        return {"text": "", "ms": ms, "why": "지어낸 말", "raw": text}
    return {"text": text, "ms": ms}

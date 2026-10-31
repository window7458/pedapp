"""⑥ TTS — 문장 → wav. 엔진은 전용 스레드 하나에서만 돈다 (COM · GPU 엔진 공통 제약). 같은 문장은 캐시.

엔진
  sapi   : Windows 내장 한국어 음성 (Microsoft Heami). 설치 없이 바로 됨 → 3단계 TTS 선정 전 임시 · 비교 기준선
  server : OpenAI 호환 /v1/audio/speech 서버 (supertonic serve 등) — p0/tts.py 와 같은 호출
새 엔진은 Engine을 상속해 _open() · _synth(text, path) 만 구현하면 된다.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
import wave
from concurrent.futures import Future
from pathlib import Path


def wav_dur(path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def trim_silence(path, thr=0.02, pad_s=0.05):
    """앞뒤 무음 제거 (엔진마다 0.3–1 s씩 붙임 → 그만큼 다음 소리가 밀림). 16 bit wav만."""
    import numpy as np
    with wave.open(str(path)) as w:
        params, sw = w.getparams(), w.getsampwidth()
        x = w.readframes(w.getnframes())
    if sw != 2:
        return
    a = np.frombuffer(x, np.int16).reshape(-1, params.nchannels)
    loud = np.flatnonzero(np.abs(a).max(1) > thr * 32767)
    if not len(loud):
        return
    pad = int(pad_s * params.framerate)
    a = a[max(0, loud[0] - pad):loud[-1] + pad]
    with wave.open(str(path), "wb") as w:
        w.setparams(params)
        w.writeframes(a.tobytes())


class Engine:
    name = "base"

    def __init__(self, cache_dir: Path):
        self.cache = Path(cache_dir) / self.name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _path(self, text):
        return self.cache / f"{hashlib.md5(text.encode()).hexdigest()[:16]}.wav"

    def _open(self):
        pass

    def _synth(self, text, path):
        raise NotImplementedError

    def _loop(self):
        self._open()
        while True:
            text, path, fut = self.q.get()
            if text is None:
                return
            t0 = time.perf_counter()
            try:
                self._synth(text, path)
                trim_silence(path)
                fut.set_result((path, wav_dur(path), {"cached": False, "synth_s": round(time.perf_counter() - t0, 3)}))
            except Exception as e:
                fut.set_exception(e)

    def synth(self, text) -> Future:
        """→ Future[(wav 경로, 길이 s, {"cached", "synth_s"})]. 캐시에 있으면 바로 완료된 Future."""
        p = self._path(text)
        fut = Future()
        if p.exists():
            fut.set_result((p, wav_dur(p), {"cached": True, "synth_s": 0.0}))
        else:
            self.q.put((text, p, fut))
        return fut

    def close(self):
        self.q.put((None, None, None))


class SapiEngine(Engine):
    name = "sapi_heami"

    def _open(self):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        self.voice = win32com.client.Dispatch("SAPI.SpVoice")
        for tok in self.voice.GetVoices():
            if "Heami" in tok.GetDescription():
                self.voice.Voice = tok
        self._w32 = win32com.client

    def _synth(self, text, path):
        fs = self._w32.Dispatch("SAPI.SpFileStream")
        fmt = self._w32.Dispatch("SAPI.SpAudioFormat")
        fmt.Type = 22                                   # 22 kHz 16 bit mono
        fs.Format = fmt
        fs.Open(str(path), 3)
        self.voice.AudioOutputStream = fs
        self.voice.Speak(text)
        fs.Close()


class ServerEngine(Engine):
    def __init__(self, cache_dir, base_url, model="supertonic", voice="F3"):
        self.name = f"server_{model}_{voice}"
        self.base_url, self.model, self.voice = base_url, model, voice
        super().__init__(cache_dir)

    def _open(self):
        pass

    def _synth(self, text, path):
        # openai 패키지 없이 urllib로 호출. lang="ko"를 주지 않으면 Supertonic이 언어를 추측해 발음이 달라짐
        import json
        import urllib.request
        body = json.dumps({"model": self.model, "voice": self.voice, "input": text,
                           "response_format": "wav", "lang": "ko"}).encode()
        req = urllib.request.Request(f"{self.base_url}/audio/speech", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())


def make_engine(spec: str, cache_dir: Path, **kw) -> Engine:
    if spec == "sapi":
        return SapiEngine(cache_dir)
    if spec == "server":
        return ServerEngine(cache_dir, kw["base_url"], kw.get("model", "supertonic"), kw.get("voice", "F3"))
    raise ValueError(f"unknown tts {spec} (sapi | server)")

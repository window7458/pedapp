"""TTS — 경고 템플릿은 미리 합성한 wav 재사용(T1), VLM 문장은 로컬 TTS 서버(T2 Supertonic 등)로 합성.

Supertonic README: `supertonic serve`가 OpenAI 호환 엔드포인트를 띄운다 → VLM과 같은 openai 클라이언트로
`/v1/audio/speech`를 부른다. (모델·음성 이름은 서버를 띄운 뒤 확인해서 --tts-model/--tts-voice로 넘긴다.)
외부 API는 부르지 않는다."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path


class TTS:
    def __init__(self, client, model: str, voice: str, cache_dir: Path):
        self.client, self.model, self.voice = client, model, voice
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str) -> Path:
        return self.cache / f"{hashlib.md5(f'{self.model}|{self.voice}|{text}'.encode()).hexdigest()[:16]}.wav"

    def synth(self, text: str) -> tuple[Path, dict]:
        """→ (wav 경로, {"cached": bool, "first_audio_s": 첫 오디오 바이트까지, "total_s"})."""
        p = self._path(text)
        if p.exists():
            return p, {"cached": True, "first_audio_s": 0.0, "total_s": 0.0}
        t0 = time.perf_counter()
        t_first = None
        with self.client.audio.speech.with_streaming_response.create(
                model=self.model, voice=self.voice, input=text, response_format="wav") as r:
            with open(p, "wb") as f:
                for chunk in r.iter_bytes():
                    if t_first is None:
                        t_first = time.perf_counter()
                    f.write(chunk)
        t1 = time.perf_counter()
        return p, {"cached": False, "first_audio_s": (t_first or t1) - t0, "total_s": t1 - t0}

    def presynth(self, texts):
        """경고 템플릿 미리 합성 (클래스 × 방향)."""
        for t in texts:
            self.synth(t)

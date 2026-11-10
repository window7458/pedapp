"""앱 음성: Google Cloud Text-to-Speech 의 ko-KR-Chirp3-HD-Zephyr (여자, 자연스러운 음성).

트리거 쪽은 윈도우 Heami(SAPI)를 쓰지만 Heami 는 윈도우에서만 돌아서 클라우드 서버(리눅스)에선 못 쓴다 → 앱은 이 음성.
- 문장 → MP3 를 DATA/tts_cache 에 둔다 (Cloud Run 에선 버킷이라 재시작해도 남음). 같은 문장은 한 번만 합성.
- 처음 나오는 문장만 0.6~1.5초 걸리고 그 뒤로는 바로 나온다. 같은 문장을 동시에 부르면 한 번만 요청한다.
- '1시 방향' → '한 시 방향' 처럼 트리거 tts.py 의 읽기 규칙(speak_text)을 그대로 빌려 쓴다 (그 파일은 고치지 않음).
- 인증: Cloud Run 에선 서버 계정 토큰(메타데이터 서버), PC 에선 gcloud 로그인 토큰.
- 서버로 가는 건 문장 글자뿐이다. 영상 · 위치는 보내지 않는다.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

VOICE = os.environ.get("TTS_VOICE", "ko-KR-Chirp3-HD-Zephyr")
RATE = float(os.environ.get("TTS_RATE", "1.1"))
MAX_CHARS = 600
NAV = Path(__file__).resolve().parent / "navigation_trigger"
API = "https://texttospeech.googleapis.com/v1/text:synthesize"


def _speak_text():
    """트리거 tts.py 의 읽기 규칙. 불러오지 못하면 그대로 읽는다."""
    try:
        spec = importlib.util.spec_from_file_location("nav_tts", NAV / "tts.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.speak_text
    except Exception:
        return lambda t: t


class _Token:
    def __init__(self):
        self.tok, self.exp, self.project = None, 0.0, os.environ.get("GOOGLE_CLOUD_PROJECT")

    def get(self) -> tuple[str, str | None]:
        if self.tok and time.time() < self.exp - 60:
            return self.tok, self.project
        try:                                   # Cloud Run: 메타데이터 서버 (Cloud Run 이 K_SERVICE 를 넣어 줌)
            if not os.environ.get("K_SERVICE"):
                raise RuntimeError("not cloud run")
            base = "http://metadata.google.internal/computeMetadata/v1/"
            h = {"Metadata-Flavor": "Google"}
            j = json.loads(urllib.request.urlopen(urllib.request.Request(base + "instance/service-accounts/default/token", headers=h), timeout=2).read())
            self.tok, self.exp = j["access_token"], time.time() + int(j.get("expires_in", 3000))
            if not self.project:
                self.project = urllib.request.urlopen(urllib.request.Request(base + "project/project-id", headers=h), timeout=2).read().decode()
            return self.tok, self.project
        except Exception:
            pass
        g = shutil.which("gcloud") or shutil.which("gcloud.cmd")   # PC: gcloud 로그인
        if not g:
            raise RuntimeError("인증 토큰을 얻을 수 없어요 (gcloud 없음)")
        out = subprocess.run([g, "auth", "print-access-token"], capture_output=True, text=True, timeout=20)
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError("gcloud 토큰 실패")
        self.tok, self.exp = out.stdout.strip(), time.time() + 1800
        if not self.project:
            p = subprocess.run([g, "config", "get-value", "project"], capture_output=True, text=True, timeout=20).stdout.strip()
            self.project = p or None
        return self.tok, self.project


class AppTTS:
    def __init__(self, cache_dir: Path, log=print, voice: str = VOICE, rate: float = RATE):
        self.dir, self.log, self.voice, self.rate = Path(cache_dir), log, voice, rate
        self.speak_text = _speak_text()
        self.token = _Token()
        self.inflight: dict[str, asyncio.Future] = {}
        self.ok = os.environ.get("TTS_OFF") != "1"

    def _key(self, spoken: str) -> Path:
        k = hashlib.sha256(json.dumps([self.voice, self.rate, spoken], ensure_ascii=False).encode()).hexdigest()[:24]
        return self.dir / f"{k}.mp3"

    def _synth(self, spoken: str, target: Path):
        tok, proj = self.token.get()
        body = {"input": {"text": spoken}, "voice": {"languageCode": "ko-KR", "name": self.voice},
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": self.rate}}
        h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
        if proj:
            h["x-goog-user-project"] = proj
        r = json.loads(urllib.request.urlopen(urllib.request.Request(API, data=json.dumps(body).encode(), headers=h), timeout=15).read())
        data = base64.b64decode(r["audioContent"])
        if not data:
            raise RuntimeError("빈 음성")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)

    async def get(self, text: str) -> Path | None:
        text = " ".join(str(text or "").split())[:MAX_CHARS]
        if not self.ok or not text:
            return None
        try:
            spoken = self.speak_text(text)
        except Exception:
            spoken = text
        target = self._key(spoken)
        if target.is_file() and target.stat().st_size > 0:
            return target
        fut = self.inflight.get(target.name)
        if fut is None:                         # 같은 문장이 동시에 오면 한 번만 합성
            fut = asyncio.ensure_future(asyncio.to_thread(self._synth, spoken, target))
            self.inflight[target.name] = fut
            fut.add_done_callback(lambda _f, n=target.name: self.inflight.pop(n, None))
        try:
            await asyncio.shield(fut)
            return target
        except Exception as e:
            self.log(f"[tts] 실패 {text[:30]!r}: {e!r}")
            return None

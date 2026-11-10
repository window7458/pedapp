"""Korean warning text -> cached MP3, independently of video processing.

Only the warning string is sent to Microsoft Edge TTS. This module does not
send images, video, detection boxes, or event timestamps to the service.
"""
import argparse
import asyncio
import hashlib
import json
import re
import ssl
import tempfile
from pathlib import Path

DEFAULT_VOICE = 'ko-KR-SunHiNeural'
DEFAULT_RATE = '+15%'
CLOCK = {1: '한', 2: '두', 3: '세', 4: '네', 5: '다섯', 6: '여섯',
         7: '일곱', 8: '여덟', 9: '아홉', 10: '열', 11: '열한', 12: '열두'}


def speak_text(text):
    """Preserve the requested polite warning while pronouncing clock directions."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Warning text must be a nonempty string.')
    text = text.strip().replace('주의하세요', '유의하세요')
    # Compatibility with the old export's abbreviated ending.
    text = text.replace('에 주의.', '에 유의하세요.')
    text = re.sub(r'\s+주의\.(?=\s|$)', '에 유의하세요.', text)
    return re.sub(r'(?<!\d)(1[0-2]|[1-9])\s*시(?=\s*방향)',
                  lambda m: CLOCK[int(m[1])] + ' 시', text)


async def _synthesize_remote(text, path, voice, rate):
    # Lazy imports keep the rule-only runtime usable without TTS dependencies.
    import aiohttp
    import edge_tts
    connector = aiohttp.TCPConnector(ssl=ssl.create_default_context())
    try:
        await edge_tts.Communicate(text, voice, rate=rate, connector=connector,
                                   connect_timeout=10, receive_timeout=30).save(str(path))
    finally:
        await connector.close()


class SpeechSynthesizer:
    """Reuse one instance on one asyncio event loop; identical clips are cached."""
    def __init__(self, cache_dir='tts_cache', voice=DEFAULT_VOICE,
                 rate=DEFAULT_RATE, timeout_s=45.0):
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive.')
        self.cache_dir = Path(cache_dir)
        self.voice = voice
        self.rate = rate
        self.timeout_s = float(timeout_s)
        self._lock = asyncio.Lock()

    async def synthesize(self, text):
        """Return a local MP3 Path. Raises on timeout/service failure."""
        spoken = speak_text(text)
        key = hashlib.sha256(json.dumps([self.voice, self.rate, spoken],
                                       ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / (key + '.mp3')
        # Serialize service requests and coalesce identical concurrent warnings.
        async with self._lock:
            if target.is_file() and target.stat().st_size > 0:
                return target
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.part', delete=False) as f:
                temporary = Path(f.name)
            try:
                await asyncio.wait_for(
                    _synthesize_remote(spoken, temporary, self.voice, self.rate),
                    timeout=self.timeout_s)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError('TTS returned an empty audio file.')
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            return target

    async def synthesize_event(self, event):
        """Pass a Pipeline event; only event['message'] is sent to TTS."""
        return await self.synthesize(event['message'])


def read_events(path):
    text = Path(path).read_text(encoding='utf-8-sig').strip()
    if not text:
        return []
    data = json.loads(text) if text.startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError('Expected a JSON list or JSONL event file.')
    for event in data:
        if not isinstance(event, dict):
            raise ValueError('Each event must be a JSON object.')
        speak_text(event.get('message'))
    return data


async def synthesize_events(events, synthesizer):
    """Synthesize in input order; timestamps are metadata, not a playback schedule."""
    results = []
    for index, event in enumerate(events):
        path = await synthesizer.synthesize_event(event)
        results.append(dict(index=index, frame=event.get('frame'), time_s=event.get('time_s'),
                            message=event['message'], tts_text=speak_text(event['message']),
                            audio_path=str(path.resolve()), voice=synthesizer.voice,
                            rate=synthesizer.rate))
    return results


async def main(args):
    synthesizer = SpeechSynthesizer(args.cache_dir, args.voice, args.rate)
    events = read_events(args.events) if args.events else [dict(message=args.text)]
    results = await synthesize_events(events, synthesizer)
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Generated/cached:', len(results), 'warnings; manifest:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text', help='One warning sentence to synthesize.')
    source.add_argument('--events', help='Pipeline events.jsonl or JSON list.')
    parser.add_argument('--cache-dir', default='tts_cache')
    parser.add_argument('--manifest', default='tts_manifest.json')
    parser.add_argument('--voice', default=DEFAULT_VOICE)
    parser.add_argument('--rate', default=DEFAULT_RATE)
    asyncio.run(main(parser.parse_args()))

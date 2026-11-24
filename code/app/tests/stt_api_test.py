"""서버 음성 인식(/api/stt): Whisper 대신 가짜로 바꿔 서버 연결 · 길이 제한 · 꺼짐(503)만 본다. 실제 모델 확인은 stt.py 를 직접."""
import os, sys, tempfile
from pathlib import Path
os.environ["DATA_DIR"] = tempfile.mkdtemp(); os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"; os.environ["ALLOW_CPU"] = "1"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
import stt
import server
from pipeline import Config

got = []
stt.transcribe = lambda pcm: (got.append(len(pcm)), {"text": "511번 버스 언제 와", "ms": 5})[1]
stt.ready = lambda: True
app = server.build_app(Config(), None, pin="424242")
with TestClient(app) as c:
    assert c.get("/api/stt").json()["on"] is True
    r = c.post("/api/stt", content=b"\0\0" * 16000, headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200 and r.json()["text"] == "511번 버스 언제 와", r.text
    assert got == [32000]
    assert c.post("/api/stt", content=b"\0" * (16000 * 2 * 17)).status_code == 413     # 16초 넘음
    server.STT_ON = False
    assert c.get("/api/stt").json()["on"] is False
    assert c.post("/api/stt", content=b"\0\0" * 100).status_code == 503                 # 꺼지면 폰은 브라우저 인식으로
print("stt api ok")

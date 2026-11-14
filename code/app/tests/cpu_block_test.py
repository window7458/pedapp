"""CPU 서버면 보행자 연결을 받지 않는다 (server_cpu 후 끊음). 보호자 · 시험용 가짜 인식기는 영향 없음."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TTS_OFF", "1"); os.environ.pop("ALLOW_CPU", None)
from fastapi.testclient import TestClient
from fakes import FakePerception
from pipeline import Config
from server import build_app

d = Path(tempfile.mkdtemp())
cfg = Config(perception="yolo", weights="none.pt", device="cpu", trigger="fakes:FakeTrigger")
with TestClient(build_app(cfg, None, pin="1", alerts_dir=d / "a")) as c:
    with c.websocket_connect("/ws?u=abc") as ws:
        m = json.loads(ws.receive_text()); assert m["type"] == "server_cpu", m
print("cpu blocked ok")
with TestClient(build_app(Config(perception=FakePerception(), trigger="fakes:FakeTrigger"), None, pin="1", alerts_dir=d / "b")) as c:
    with c.websocket_connect("/ws?u=abc") as ws:
        m = json.loads(ws.receive_text()); assert m["type"] == "hello", m
print("fake perception ok")

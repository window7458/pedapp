"""Cloud Run: 연결 · 요청이 IDLE_EXIT_S 동안 없으면 서버가 스스로 끝남 (os._exit). 연결이 있으면 안 끝남. (약 70초)"""
import os, sys, time, tempfile, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(TTS_OFF="1", K_SERVICE="test", IDLE_EXIT_S="5")
import server
from fastapi.testclient import TestClient
from fakes import FakePerception
from pipeline import Config
server.IDLE_EXIT_S = 5
exited = []
server.os._exit = lambda code: exited.append(time.time())
with TestClient(server.build_app(Config(perception=FakePerception(), trigger="fakes:FakeTrigger"), None, pin="1",
                                 alerts_dir=Path(tempfile.mkdtemp()))) as c:
    with c.websocket_connect("/ws?u=idle1") as ws:
        ws.receive_text(); time.sleep(32)
        assert not exited, "연결이 있는데 꺼짐"
    time.sleep(36)
    assert exited, "연결이 없는데 안 꺼짐"
print("idle exit ok")

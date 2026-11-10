"""보행 시작(set_source phone)하면 풍경 서버(VLM)의 /api/health 를 불러 미리 깨운다. 걷는 동안엔 4분에 한 번만."""
import os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
hits = []
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path); self.send_response(200); self.end_headers(); self.wfile.write(b'{"vlm": true}')
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 8793), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["SCENE_URL"] = "http://127.0.0.1:8793"; os.environ["DATA_DIR"] = tempfile.mkdtemp()
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app
import cv2, numpy as np
cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
with TestClient(build_app(cfg, None, pin="424242")) as c, c.websocket_connect("/ws") as p:
    p.receive_json()
    p.send_json({"type": "set_source", "mode": "phone"}); time.sleep(0.5)
    assert hits == ["/api/health"], hits
    ok, b = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    for _ in range(5): p.send_bytes(b.tobytes())
    time.sleep(0.5)
    assert len(hits) == 1, hits          # 걷는 중엔 4분에 한 번만
print("풍경 서버 미리 깨우기 통과", hits)

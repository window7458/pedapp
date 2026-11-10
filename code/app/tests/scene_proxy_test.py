"""'주변 설명해줘': 앱 서버 /api/scene → 풍경 서버 /api/text (session · text · image · want_image=false) → answer.
같은 보행자는 같은 session. 풍경 서버가 아직이면 503 + 다시 깨우기."""
import json, os, sys, tempfile, threading, time, cgi, io
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
calls, mode = [], {"ok": True}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        calls.append(("GET", self.path, None)); self.send_response(200); self.end_headers(); self.wfile.write(b'{"vlm": true}')
    def do_POST(self):
        env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]}
        fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=env)
        f = {k: (fs[k].value if not fs[k].filename else len(fs[k].value)) for k in fs.keys()}
        calls.append(("POST", self.path, f))
        if not mode["ok"]:
            self.send_response(500); self.end_headers(); return
        ans = "좁은 골목이에요. 12시 방향으로 길이 이어져요." if "주변" in f["text"] else "간판에는 '세탁'이라고 써 있어요."
        body = json.dumps({"answer": ans, "intent": {"intent": "describe" if "주변" in f["text"] else "question"}, "level": "short", "timing": {"total_ms": 12}}, ensure_ascii=False).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 8794), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["SCENE_URL"] = "http://127.0.0.1:8794"; os.environ["DATA_DIR"] = tempfile.mkdtemp()
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app
cfg = Config(perception=FakePerception(), trigger="fakes:FakeTrigger", max_fps=30.0)
with TestClient(build_app(cfg, None, pin="424242")) as c:
    r = c.post("/api/scene", data={"text": "주변 설명해줘"}, files={"image": ("f.jpg", b"\xff\xd8jpegdata", "image/jpeg")})
    assert r.status_code == 200 and "골목" in r.json()["answer"], r.text
    p = [x for x in calls if x[0] == "POST"][-1]
    assert p[1] == "/api/text" and p[2]["want_image"] == "false" and p[2]["image"] == 10 and p[2]["session"], p
    r2 = c.post("/api/scene", data={"text": "간판에 뭐라고 써 있어?"})
    assert "세탁" in r2.json()["answer"] and r2.json()["intent"] == "question"
    p2 = [x for x in calls if x[0] == "POST"][-1]
    assert p2[2]["session"] == p[2]["session"] and "image" not in p2[2]          # 같은 사람 = 같은 대화
    mode["ok"] = False
    r3 = c.post("/api/scene", data={"text": "주변 설명해줘"}); time.sleep(0.3)
    assert r3.status_code == 503 and calls[-1][:2] == ("GET", "/api/health"), (r3.status_code, calls[-1])   # 준비 중 → 다시 깨움
print("주변 설명 연결 테스트 통과")

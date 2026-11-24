"""집 주소 찾기(/api/place)와 건물 안 판단(/api/indoor): 길찾기 · 건물 조회를 가짜로 바꿔 서버 연결만 본다."""
import os, sys, tempfile
from pathlib import Path
os.environ["DATA_DIR"] = tempfile.mkdtemp(); os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
import nav
from pipeline import Config
from server import build_app


def fake_place(self, q, lat, lon):
    if q == "없는주소":
        raise LookupError("못 찾음")
    return dict(name=q, lat=37.45, lon=126.65, full="")


nav.Nav.place = fake_place
nav.Nav.in_building = lambda self, lat, lon: lat > 37.0
app = build_app(Config(), None, pin="424242")
with TestClient(app) as c:
    r = c.get("/api/place", params={"q": "인하로 100"}); assert r.status_code == 200 and r.json()["lat"] == 37.45, r.text
    assert c.get("/api/place", params={"q": "없는주소"}).status_code == 404
    assert c.get("/api/indoor", params={"lat": 37.45, "lon": 126.65}).json() == {"building": True}
    assert c.get("/api/indoor", params={"lat": 10, "lon": 126.65}).json() == {"building": False}
print("home/indoor api ok")

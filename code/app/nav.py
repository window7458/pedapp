"""도보 길 안내 — "인하대역 가자" → 장소 찾기 → 걷는 길 → 갈림길 · 횡단보도마다 음성 안내.

  기본: TMAP (환경 변수 TMAP_KEY)
    장소 찾기  GET  apis.openapi.sk.com/tmap/pois          (관련도 순 → 이름이 맞는 것 중 가까운 곳)
    보행자 길  POST apis.openapi.sk.com/tmap/routes/pedestrian  searchOption=30 (최단거리 + 계단 제외)
              → 횡단보도(turnType 211~217)를 따로 알려 준다
  예비: TMAP 키가 없거나 실패하면 OpenStreetMap (Nominatim 장소 + OSRM 도보, 키 없음, 가벼운 사용만)
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.parse
import urllib.request

UA = "SayEye/1.0 (blind navigation capstone demo)"
TMAP = "https://apis.openapi.sk.com/tmap"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
OSRM = "https://routing.openstreetmap.de/routed-foot/route/v1/foot"
CROSSWALK = set(range(211, 218))          # TMAP turnType: 횡단보도 (좌측 · 우측 · 8시 … 방향)
_lock = threading.Lock()
_last = [0.0]


def dist_m(lat1, lon1, lat2, lon2):
    k = 111320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


def _meters(m: float) -> str:
    m = int(round(m / 10.0) * 10) if m >= 30 else int(round(m))
    return f"{m}미터"


def _speak(desc: str) -> str:
    """TMAP 안내문 → 말하기 좋게: '좌회전 후 보행자도로를 따라 111m 이동' → '좌회전한 뒤 보행자도로를 따라 110미터 가세요.'"""
    s = desc.strip()
    s = re.sub(r"(\d+)m 이동", lambda m: _meters(int(m.group(1))) + " 가세요", s)
    s = s.replace("좌회전 후", "왼쪽으로 돈 뒤").replace("우회전 후", "오른쪽으로 돈 뒤")
    s = s.replace("횡단보도 후", "횡단보도를 건넌 뒤").replace("직진 후", "곧장 간 뒤")
    s = re.sub(r"(\d+)시 방향 ", r"\1시 방향으로 ", s)
    return s.rstrip(".") + "."


class Nav:
    def __init__(self, key: str | None = None):
        self.key = key if key is not None else (os.environ.get("TMAP_KEY") or "")

    # ── TMAP ──
    def _tmap_get(self, path, params):
        q = urllib.parse.urlencode(dict(version="1", reqCoordType="WGS84GEO", resCoordType="WGS84GEO", **params))
        req = urllib.request.Request(f"{TMAP}/{path}?{q}", headers={"appKey": self.key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as r:
            body = r.read().decode("utf-8")
        return json.loads(body) if body else {}

    def _tmap_search(self, q, lat, lon):
        d = self._tmap_get("pois", dict(searchKeyword=q, count=10, centerLat=lat, centerLon=lon, searchtypCd="A"))
        pois = (((d.get("searchPoiInfo") or {}).get("pois") or {}).get("poi")) or []
        if not pois:
            return None
        key = re.sub(r"\s+", "", q)

        def rank(p):                            # 이름이 똑같음 > 이름이 그걸로 시작(부평역[인천1호선] · 인하대역 4번출구) > 이름에 들어 있음 > 나머지
            n = re.sub(r"\s+", "", p["name"])
            base = re.sub(r"\[.*?\]|\(.*?\)", "", n)          # '부평역[인천1호선]' → '부평역'
            exact = base == key or re.fullmatch(re.escape(key) + r"\d+번출구", base)   # 역 이름 · 그 역의 출구
            r = 0 if exact else 1 if n.startswith(key) else 2 if key in n else 3
            return (r, float(p.get("radius") or 999))
        p = min(pois, key=rank)
        plat, plon = float(p.get("frontLat") or p["noorLat"]), float(p.get("frontLon") or p["noorLon"])
        return dict(name=p["name"], lat=plat, lon=plon, full=p.get("upperAddrName", ""))

    def _tmap_route(self, lat, lon, dlat, dlon, name):
        body = json.dumps(dict(startX=str(lon), startY=str(lat), endX=str(dlon), endY=str(dlat),
                               startName=urllib.parse.quote("출발"), endName=urllib.parse.quote(name or "도착"),
                               searchOption="30")).encode()
        req = urllib.request.Request(f"{TMAP}/routes/pedestrian?version=1", data=body,
                                     headers={"appKey": self.key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode("utf-8"))
        feats = d.get("features") or []
        if not feats:
            raise LookupError("걷는 길을 찾지 못했어요")
        p0 = feats[0]["properties"]
        steps, line = [], []
        for f in feats:
            g, pr = f["geometry"], f["properties"]
            if g["type"] == "LineString":
                line += [[c[1], c[0]] for c in g["coordinates"]]
                continue
            lon_, lat_ = g["coordinates"]
            tt = int(pr.get("turnType") or 0)
            if tt == 200:                       # 출발
                first = _speak(pr.get("description") or "")
                continue
            if tt in (201, 203):                # 도착
                steps.append(dict(lat=lat_, lon=lon_, kind="arrive", text="도착했어요."))
                continue
            steps.append(dict(lat=lat_, lon=lon_, kind="crosswalk" if tt in CROSSWALK else "turn",
                              text=_speak(pr.get("description") or "")))
        return dict(distance=int(p0.get("totalDistance") or 0), minutes=max(1, round(int(p0.get("totalTime") or 60) / 60)),
                    steps=steps, line=line, first=locals().get("first") or "길을 따라 가세요.", source="TMAP")

    # ── OpenStreetMap (예비) ──
    def _osm_get(self, url):
        with _lock:                              # 공개 서버 규칙: 초당 1번 이내
            wait = 1.05 - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko"})
        with urllib.request.urlopen(req, timeout=12) as r:
            return json.loads(r.read().decode("utf-8"))

    def _osm_search(self, q, lat, lon):
        for bounded in (1, 0):
            vb = f"{lon - .25},{lat + .2},{lon + .25},{lat - .2}"
            res = self._osm_get(NOMINATIM + "?" + urllib.parse.urlencode(
                {"format": "json", "limit": 5, "countrycodes": "kr", "accept-language": "ko", "viewbox": vb, "bounded": bounded, "q": q}))
            if res:
                b = min(res, key=lambda r: dist_m(lat, lon, float(r["lat"]), float(r["lon"])))
                return dict(name=b.get("name") or b["display_name"].split(",")[0], lat=float(b["lat"]), lon=float(b["lon"]), full=b["display_name"])
        return None

    def _osm_route(self, lat, lon, dlat, dlon, name):
        d = self._osm_get(f"{OSRM}/{lon},{lat};{dlon},{dlat}?steps=true&overview=full&geometries=geojson")
        if d.get("code") != "Ok":
            raise LookupError("걷는 길을 찾지 못했어요")
        r = d["routes"][0]
        raw = r["legs"][0]["steps"]
        turn = {"left": "왼쪽으로", "right": "오른쪽으로", "slight left": "약간 왼쪽으로", "slight right": "약간 오른쪽으로",
                "sharp left": "왼쪽으로 크게", "sharp right": "오른쪽으로 크게", "uturn": "뒤로"}
        steps = []
        for i, s in enumerate(raw):
            m = s["maneuver"]
            lon_, lat_ = m["location"]
            if m["type"] == "depart":
                continue
            if m["type"] == "arrive":
                steps.append(dict(lat=lat_, lon=lon_, kind="arrive", text="도착했어요."))
                continue
            t = turn.get(m.get("modifier", ""))
            seg = _meters(s["distance"])
            steps.append(dict(lat=lat_, lon=lon_, kind="turn",
                              text=(f"{t} 돈 뒤 {seg} 가세요." if t else f"{seg} 곧장 가세요.")))
        first = f"{_meters(raw[0]['distance'])} 곧장 가세요." if raw else "길을 따라 가세요."
        return dict(distance=round(r["distance"]), minutes=max(1, round(r["duration"] / 60)), steps=steps,
                    line=[[c[1], c[0]] for c in r["geometry"]["coordinates"]], first=first, source="OSM")

    # ── 밖에서 부르는 것 ──
    def route(self, lat, lon, dlat, dlon, name=""):
        if self.key:
            try:
                return self._tmap_route(lat, lon, dlat, dlon, name)
            except Exception:
                pass
        return self._osm_route(lat, lon, dlat, dlon, name)

    def plan(self, q: str, lat: float, lon: float) -> dict:
        dest = None
        if self.key:
            try:
                dest = self._tmap_search(q, lat, lon)
            except Exception:
                dest = None
        dest = dest or self._osm_search(q, lat, lon)
        if dest is None:
            raise LookupError(f"'{q}'을 찾지 못했어요")
        r = self.route(lat, lon, dest["lat"], dest["lon"], dest["name"])
        intro = f"{dest['name']}까지 {_meters(r['distance'])}, 걸어서 약 {r['minutes']}분이에요. {r['first']}"
        return dict(dest=dest, intro=intro, **r)

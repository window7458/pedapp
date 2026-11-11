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
    s = re.sub(r"(\d+)시 방향으로 (왼쪽|오른쪽)으로 돈 뒤", r"\1시 방향으로 돈 뒤", s)   # '2시 방향으로 오른쪽으로' → '2시 방향으로'
    return s.rstrip(".") + "."


def _bends(line, steps, look=15.0, min_deg=35.0, near_step=30.0, gap=40.0):
    """TMAP 이 따로 알려 주지 않는 '휘는 길' 찾기 → 안내 단계(kind='curve')를 경로 순서대로 끼워 넣는다.
    경로 선을 5 m 간격으로 따라가며 앞뒤 15 m 방향을 비교, 같은 쪽으로 계속 꺾이는 구간을 묶어 모두 합쳐 35° 넘게 꺾이면 그 시작점.
    갈림길·횡단보도 안내 지점 30 m 안이면 그쪽 안내가 있으니 빼고, 휘는 곳끼리는 40 m 넘게 떨어뜨린다."""
    if len(line) < 3:
        return steps
    lat0 = line[0][0]
    k, kx = 111320.0, 111320.0 * math.cos(math.radians(lat0))
    xy = [((p[1] - line[0][1]) * kx, (p[0] - lat0) * k) for p in line]
    cum = [0.0]
    for a, b in zip(xy, xy[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cum[-1]
    if total < 2 * look + 10:
        return steps

    def at(d):                                  # 경로 위 d 미터 지점 (x, y, 위도, 경도)
        d = min(max(d, 0.0), total)
        i = max(0, min(len(cum) - 2, next((j for j in range(1, len(cum)) if cum[j] >= d), len(cum) - 1) - 1))
        seg = cum[i + 1] - cum[i]
        t = (d - cum[i]) / seg if seg > 0 else 0.0
        x, y = xy[i][0] + (xy[i + 1][0] - xy[i][0]) * t, xy[i][1] + (xy[i + 1][1] - xy[i][1]) * t
        return x, y, lat0 + y / k, line[0][1] + x / kx

    def turn(d):                                # +면 왼쪽, -면 오른쪽 (도)
        a, b, c = at(d - look), at(d), at(d + look)
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        return (math.degrees(h2 - h1) + 180) % 360 - 180

    def pos(lat, lon, lo=0.0):                  # 안내 지점이 경로 몇 미터쯤인지 (앞쪽으로만 찾음)
        x, y = (lon - line[0][1]) * kx, (lat - lat0) * k
        best, bd = lo, 1e18
        d = lo
        while d <= total:
            p = at(d)
            e = (p[0] - x) ** 2 + (p[1] - y) ** 2
            if e < bd:
                best, bd = d, e
            d += 5.0
        return best

    s_pos, lo = [], 0.0
    for st in steps:
        lo = pos(st["lat"], st["lon"], lo)
        s_pos.append(lo)

    def heading(p, q):
        return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))

    def close(run):                             # 휘는 구간 전체로 얼마나 꺾였나: 들어갈 때 방향 vs 나올 때 방향
        d0, d1 = run[0][0], run[-1][0]
        a = (heading(at(d1), at(d1 + look)) - heading(at(d0 - look), at(d0)) + 180) % 360 - 180
        if abs(a) >= min_deg:
            found.append((d0, a))               # 휘기 시작하는 곳에서 말한다 (앱은 25 m 앞에서 읽음)

    found, run = [], []                         # (위치, 꺾인 각)
    d = look + 5.0
    while d <= total - look - 5.0:
        a = turn(d)
        bend = abs(a) >= 12.0                   # 조금씩이라도 계속 같은 쪽으로 꺾이면 한 구간
        if bend and (not run or (a > 0) == (run[-1][1] > 0)):
            run.append((d, a))
        else:
            if run:
                close(run)
            run = [(d, a)] if bend else []
        d += 5.0
    if run:
        close(run)

    curves = []
    for d, a in found:
        if any(abs(d - sp) < near_step for sp in s_pos) or d < 20 or d > total - 20:
            continue
        if curves and d - curves[-1][0] < gap:
            continue
        _, _, la, lo_ = at(d)
        side = "왼쪽" if a > 0 else "오른쪽"
        how = "크게 " if abs(a) >= 70 else ""
        curves.append((d, dict(lat=la, lon=lo_, kind="curve", text=f"길이 {side}으로 {how}휘어져요. 길을 따라 가세요.")))

    merged = sorted([(sp, i, st) for i, (sp, st) in enumerate(zip(s_pos, steps))] +
                    [(d, -1, st) for d, st in curves], key=lambda r: (r[0], r[1]))
    return [st for _, _, st in merged]


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
        r = None
        if self.key:
            try:
                r = self._tmap_route(lat, lon, dlat, dlon, name)
            except Exception:
                r = None
        r = r or self._osm_route(lat, lon, dlat, dlon, name)
        try:
            r["steps"] = _bends(r["line"], r["steps"])
        except Exception:
            pass                                 # 휘는 길 계산이 틀려도 길 안내는 그대로
        return r

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

"""버스 도착 안내 — 보행자 위치에서 가장 가까운 정류장의 도착 예정 버스.

  서울 : 서울특별시_정류소정보조회 서비스 (getStationByPos → getStationByUid)          data.go.kr 15000303
  그 밖(인천 · 경기 · 광역시 …) : 국토교통부 TAGO 버스정류소정보 + 버스도착정보               data.go.kr 15098534 · 15098530
  인증키: 공공데이터포털 일반 인증키 하나 (환경 변수 DATA_GO_KR_KEY). 서비스마다 '활용신청'이 돼 있어야 한다.
  서울 범위 안이면 서울 API 를 먼저 쓰고, 정류장이 안 나오면 TAGO 로 넘어간다.
"""
from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request

SEOUL = "http://ws.bus.go.kr/api/rest/stationinfo"
TAGO_STOP = "https://apis.data.go.kr/1613000/BusSttnInfoInqireService"
TAGO_ARR = "https://apis.data.go.kr/1613000/ArvlInfoInqireService"


def in_seoul(lat: float, lon: float) -> bool:
    return 37.413 <= lat <= 37.715 and 126.764 <= lon <= 127.184


def _get(url: str, params: dict, timeout=8) -> dict:
    q = urllib.parse.urlencode(params, safe="%")
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "SayEye"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _items(d: dict) -> list:
    """TAGO 응답: response.body.items.item (하나면 dict, 없으면 '')"""
    try:
        it = d["response"]["body"]["items"]
        it = it.get("item", []) if isinstance(it, dict) else []
        return it if isinstance(it, list) else [it]
    except Exception:
        return []


def _dist_m(lat1, lon1, lat2, lon2):
    k = 111320.0
    return math.hypot((lat1 - lat2) * k, (lon1 - lon2) * k * math.cos(math.radians(lat1)))


class Bus:
    def __init__(self, key: str | None = None):
        self.key = key or os.environ.get("DATA_GO_KR_KEY") or ""

    @property
    def ok(self) -> bool:
        return bool(self.key)

    # ── 서울 ──
    def _seoul(self, lat, lon):
        d = _get(f"{SEOUL}/getStationByPos", {"serviceKey": self.key, "tmX": lon, "tmY": lat, "radius": 400, "resultType": "json"})
        if d.get("error") or d.get("status") == 401:
            raise PermissionError(d.get("message") or "서울 API 키 오류")
        stops = (d.get("msgBody") or {}).get("itemList") or []
        if not stops:
            return None
        s = min(stops, key=lambda x: float(x.get("dist") or 1e9))
        a = _get(f"{SEOUL}/getStationByUid", {"serviceKey": self.key, "arsId": s["arsId"], "resultType": "json"})
        arr = []
        for it in (a.get("msgBody") or {}).get("itemList") or []:
            msg = str(it.get("arrmsg1") or "")
            sec = it.get("traTime1") or it.get("exps1")
            mins = None
            try:
                mins = max(0, round(int(sec) / 60)) if sec not in (None, "", "0") else None
            except (TypeError, ValueError):
                pass
            if "곧 도착" in msg:
                mins = 0
            left = None
            if "번째 전" in msg:
                try:
                    left = int(msg.split("[")[1].split("번째")[0])
                except Exception:
                    pass
            arr.append(dict(route=str(it.get("rtNm") or ""), min=mins, stops_left=left, msg=msg,
                            ended="운행종료" in msg or "출발대기" in msg))
        return dict(provider="서울", stop=s.get("stationNm"), stop_id=s.get("arsId"), dist_m=round(float(s.get("dist") or 0)), arrivals=arr)

    # ── TAGO (인천 · 경기 · 광역시 …) ──
    def _tago(self, lat, lon):
        d = _get(f"{TAGO_STOP}/getCrdntPrxmtSttnList", {"serviceKey": self.key, "_type": "json", "gpsLati": lat, "gpsLong": lon, "numOfRows": 10})
        if "OpenAPI_ServiceResponse" in json.dumps(d):
            raise PermissionError("TAGO 버스정류소정보 키 오류 (활용신청 필요)")
        stops = _items(d)
        if not stops:
            return None
        s = min(stops, key=lambda x: _dist_m(lat, lon, float(x.get("gpslati") or 0), float(x.get("gpslong") or 0)))
        a = _get(f"{TAGO_ARR}/getSttnAcctoArvlPrearngeInfoList", {"serviceKey": self.key, "_type": "json",
                                                                  "cityCode": s["citycode"], "nodeId": s["nodeid"], "numOfRows": 50})
        arr = []
        for it in _items(a):
            sec = it.get("arrtime")
            arr.append(dict(route=str(it.get("routeno") or ""), min=round(int(sec) / 60) if sec not in (None, "") else None,
                            stops_left=it.get("arrprevstationcnt"), msg="", ended=False))
        return dict(provider="TAGO", stop=s.get("nodenm"), stop_id=s.get("nodeid"),
                    dist_m=round(_dist_m(lat, lon, float(s.get("gpslati") or 0), float(s.get("gpslong") or 0))), arrivals=arr)

    def near(self, lat: float, lon: float, route: str | None = None) -> dict:
        if not self.ok:
            raise PermissionError("버스 정보 키가 없어요 (DATA_GO_KR_KEY)")
        errs = []
        out = None
        for fn in ((self._seoul, self._tago) if in_seoul(lat, lon) else (self._tago, self._seoul)):
            try:
                out = fn(lat, lon)
                if out:
                    break
            except Exception as e:
                errs.append(f"{fn.__name__}: {e}")
        if not out:
            raise LookupError("가까운 정류장을 찾지 못했어요" + (f" ({'; '.join(errs)})" if errs else ""))
        arr = sorted(out["arrivals"], key=lambda a: (a["min"] is None, a["min"] if a["min"] is not None else 999))
        if route:
            r = route.replace(" ", "")
            arr = [a for a in arr if a["route"].replace(" ", "") == r]
        out["arrivals"] = arr
        out["sentence"] = sentence(out, route)
        return out


def sentence(out: dict, route: str | None) -> str:
    """보행자에게 읽어 줄 한 문장."""
    stop = out.get("stop") or "가까운 정류장"
    arr = [a for a in out["arrivals"] if not a.get("ended")]
    if route:
        if not arr:
            return f"{stop} 정류장에 {route}번 도착 정보가 없어요."
        a = arr[0]
        when = "곧 도착해요" if a["min"] == 0 else (f"{a['min']}분 뒤에 와요" if a["min"] is not None else "도착 시간을 알 수 없어요")
        left = f", {a['stops_left']}정거장 전이에요" if a.get("stops_left") not in (None, "", 0) else ""
        return f"{stop} 정류장, {route}번은 {when}{left}."
    if not arr:
        return f"{stop} 정류장에 곧 오는 버스가 없어요."
    when = lambda m: "곧" if m == 0 else (f"{m}분 뒤" if m is not None else "")
    first, nxt = {}, {}                       # 같은 노선은 한 번만: "511번 1분 뒤, 다음 차 5분 뒤"
    for a in arr:
        if a["route"] not in first:
            first[a["route"]] = a
        elif a["route"] not in nxt:
            nxt[a["route"]] = a
    parts = []
    for r, a in list(first.items())[:3]:
        s = f"{r}번 {when(a['min'])}".strip()
        if r in nxt and nxt[r]["min"] is not None:
            s += f", 다음 차 {when(nxt[r]['min'])}"
        parts.append(s)
    return f"{stop} 정류장이에요. " + ". ".join(parts) + "."

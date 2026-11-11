"""지하철역 엘리베이터 안내 — "인하대역 엘리베이터 어디야" · (역 근처에서) "엘리베이터 어디 있어".

  철도산업정보센터(KRIC) 편의정보 stationElevator  openapi.kric.go.kr/openapi/convenientInfo/stationElevator
    요청: railOprIsttCd(운영기관) · lnCd(노선) · stinCd(역)   인증키: 환경 변수 KRIC_KEY
  역 이름 → 코드: kric_stations.csv (KRIC 역사 코드표, 전국 1,108역. 공개 저장소 paradomi/tooktar 의 data/station_codes.csv)
  역 이름을 말하지 않으면 TMAP 장소 찾기로 가장 가까운 지하철역을 찾는다.
  한 역에 노선이 여럿이면(부평: 1호선 · 인천1호선) 모두 묻는다.
  말하기: 밖(지상)으로 나가는 엘리베이터는 출구 번호로, 승강장으로 가는 것은 짧은 위치 설명으로.
"""
from __future__ import annotations

import csv
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://openapi.kric.go.kr/openapi/convenientInfo/stationElevator"
CSV = Path(__file__).with_name("kric_stations.csv")


def _norm(name: str) -> str:
    """'인하대역[수인분당선]' · '인하대 역' · '인하대역 4번출구' → '인하대'"""
    n = re.sub(r"\[.*?\]|\(.*?\)|\s+", "", name or "")
    n = re.sub(r"\d+번(출구|출입구)$", "", n)
    return re.sub(r"역$", "", n)


def _floor(kind: str, n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    return f"지하 {n}층" if kind == "지하" else f"{n}층"


def _place(dtl: str) -> str:
    """'(1F)4번출입구앞(B1)임대상가앞(B2)고객안내센터옆' → 첫 구간만 '4번출입구앞' — 길게 읽지 않게."""
    parts = [p for p in re.split(r"\((?:B[\d!]+|\d+F)\)", dtl or "") if p.strip()]
    p = (parts[0] if parts else (dtl or "")).strip()
    p = re.sub(r"([가-힣])(앞|옆|뒤|근처|사이)", r"\1 \2", p)
    p = p.replace("엘리베이터", "").replace("출입구", " 출입구").replace("출입문", " 출입문").replace("타는곳", " 타는 곳")
    p = re.sub(r"(앞|옆|뒤|게이트|방향|방면|승강장)(?=[가-힣\d])", r"\1 ", p)   # '우측게이트옆인천행' → '우측게이트 옆 인천행'
    p = re.sub(r"(?<=[가-힣])(?=\d)", " ", p)                                  # '방면8-1' → '방면 8-1'
    return re.sub(r"\s+", " ", p).strip()


class Subway:
    def __init__(self, key: str | None = None, nav=None):
        self.key = key if key is not None else (os.environ.get("KRIC_KEY") or "")
        self.nav = nav                          # 가장 가까운 역 찾기 (TMAP)
        self.rows = []
        if CSV.exists():
            with open(CSV, encoding="utf-8-sig") as f:
                self.rows = list(csv.DictReader(f))

    @property
    def ok(self) -> bool:
        return bool(self.key and self.rows)

    def stations(self, name: str) -> list[dict]:
        key = _norm(name)
        return [r for r in self.rows if _norm(r["STIN_NM"]) == key]

    def nearest(self, lat: float, lon: float) -> str | None:
        """TMAP 에서 지하철역을 가까운 순으로 → 코드표에 있는 첫 역 이름."""
        if not (self.nav and self.nav.key):
            return None
        d = self.nav._tmap_get("pois", dict(searchKeyword="지하철역", count=10, centerLat=lat, centerLon=lon,
                                            searchtypCd="R", radius="1"))
        pois = (((d.get("searchPoiInfo") or {}).get("pois") or {}).get("poi")) or []
        for p in sorted(pois, key=lambda p: float(p.get("radius") or 999)):
            if self.stations(p.get("name", "")):
                return _norm(p["name"])
        return None

    def _get(self, r: dict) -> list[dict]:
        q = urllib.parse.urlencode(dict(serviceKey=self.key, format="json", railOprIsttCd=r["RAIL_OPR_ISTT_CD"],
                                        lnCd=r["LN_CD"], stinCd=r["STIN_CD"]))
        with urllib.request.urlopen(f"{API}?{q}", timeout=10) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        code = (d.get("header") or {}).get("resultCode")
        if code == "30":
            raise PermissionError("KRIC 키 오류")
        return d.get("body") or []

    def elevators(self, name: str | None = None, lat: float | None = None, lon: float | None = None) -> dict:
        if not self.ok:
            raise PermissionError("엘리베이터 정보 키가 없어요 (KRIC_KEY)")
        if not name and lat is not None:
            name = self.nearest(lat, lon)
            if not name:
                raise LookupError("가까운 지하철역을 찾지 못했어요")
        rows = self.stations(name or "")
        if not rows:
            raise LookupError(f"'{name}' 역을 찾지 못했어요")
        out, seen = [], set()
        for r in rows:
            for e in self._get(r):
                k = (e.get("exitNo"), e.get("dtlLoc"))
                if k in seen:
                    continue
                seen.add(k)
                fr, to = (e.get("grndDvNmFr"), e.get("runStinFlorFr")), (e.get("grndDvNmTo"), e.get("runStinFlorTo"))
                outside = "지상" in (fr[0], to[0]) and (bool(e.get("exitNo")) or "출" in (e.get("dtlLoc") or ""))
                out.append(dict(line=r["LN_NM"], exit=e.get("exitNo"), place=_place(e.get("dtlLoc") or ""),
                                raw=e.get("dtlLoc"), outside=outside,
                                floors=f"{_floor(*fr)}에서 {_floor(*to)}"))
        st = _norm(name or "")
        return dict(station=st, lines=[r["LN_NM"] for r in rows], elevators=out, sentence=sentence(st, out))


def _exit_words(x: str) -> str:
    """'1,2' → '1번 2번 출구 사이' · '1~4' → '1번에서 4번 출구 쪽' · '4' → '4번 출구'"""
    x = str(x).replace(" ", "")
    if "~" in x:
        a, b = x.split("~", 1)
        return f"{a}번에서 {b}번 출구 쪽"
    if "," in x:
        return " ".join(f"{p}번" for p in x.split(",") if p) + " 출구 사이"
    return f"{x}번 출구"


def sentence(station: str, els: list[dict]) -> str:
    if not els:
        return f"{station}역 엘리베이터 정보가 없어요."
    outs = [e for e in els if e["outside"]]
    ins = [e for e in els if not e["outside"]]
    s = f"{station}역 엘리베이터는 {len(els)}곳이에요."
    if outs:
        where = []
        for e in outs:
            w = _exit_words(e["exit"]) if e["exit"] else e["place"]
            if w and w not in where:
                where.append(w)
        s += " 밖으로 나가는 건 " + ", ".join(where[:4]) + "에 있어요."
    if ins:
        where = []
        for e in ins:
            if e["place"] and e["place"] not in where:
                where.append(e["place"])
        if where:
            s += " 승강장으로 가는 건 " + ", ".join(where[:3]) + (" 등" if len(where) > 3 else "") + "에 있어요."
    return s

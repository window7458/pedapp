"""지하철역 안내 — 철도산업정보센터(KRIC) 공개 API. 인증키: 환경 변수 KRIC_KEY (키 하나로 아래 셋 다).

  ① 엘리베이터      convenientInfo/stationElevator   "인하대역 엘리베이터 어디야" · "엘리베이터 어디 있어"
       밖(지상)으로 나가는 엘리베이터는 출구 번호로, 승강장으로 가는 것은 짧은 위치 설명으로.
  ② 다음 열차       trainUseInfo/subwayTimetable     "인하대역 지하철 언제 와"   (dayCd 8 평일 · 9 휴일, 시간표 기준)
       응답에 방향이 없어서, 옆 역 시간표에서 같은 열차가 더 늦게 지나가면 그 역 쪽으로 가는 열차로 본다.
  ③ 승강장 가는 길  handicapped/stationMovement      "인하대역 1번 출구에서 송도 방면 가는 길"
       출입구 → 승강장 단계별 안내. 출구 · 방향을 말하지 않았으면 고를 수 있는 것을 알려 준다.

  역 이름 → 코드: kric_stations.csv (KRIC 역사 코드표, 전국 1,108역. 공개 저장소 paradomi/tooktar 의 data/station_codes.csv)
  역 이름을 말하지 않으면 TMAP 장소 찾기로 가장 가까운 지하철역을 찾는다. 노선이 여럿인 역(부평: 1호선 · 인천1호선)은 모두 본다.
  KRIC 요청: 시간표 말고는 분기 · 반기마다 바뀌니 하루 1번 정도만 받기 → 받은 것은 cache_dir 에 저장해 다시 쓴다.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

KRIC = "https://openapi.kric.go.kr/openapi"
CSV = Path(__file__).with_name("kric_stations.csv")
KST = dt.timezone(dt.timedelta(hours=9))          # 한국은 서머타임 없음 (서버는 UTC)


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
    """'(1F)4번출입구앞(B1)임대상가앞(B2)고객안내센터옆' → 첫 구간만 '4번 출입구 앞' — 길게 읽지 않게."""
    parts = [p for p in re.split(r"\((?:B[\d!]+|\d+F)\)", dtl or "") if p.strip()]
    p = (parts[0] if parts else (dtl or "")).strip()
    p = re.sub(r"([가-힣])(앞|옆|뒤|근처|사이)", r"\1 \2", p)
    p = p.replace("엘리베이터", "").replace("출입구", " 출입구").replace("출입문", " 출입문").replace("타는곳", " 타는 곳")
    p = re.sub(r"(앞|옆|뒤|게이트|방향|방면|승강장)(?=[가-힣\d])", r"\1 ", p)   # '우측게이트옆인천행' → '우측게이트 옆 인천행'
    p = re.sub(r"(?<=[가-힣])(?=\d)", " ", p)                                  # '방면8-1' → '방면 8-1'
    return re.sub(r"\s+", " ", p).strip()


def _step(t: str) -> str:
    """'3) (B1) 대합실로 이동' → '지하 1층 대합실로 이동' · '1/2번 출입구' → '1번 2번 출입구'"""
    t = re.sub(r"^\s*\d+\)\s*", "", t or "")
    t = re.sub(r"\(B(\d+)\)", r"지하 \1층", t)
    t = re.sub(r"\((\d+)F\)", r"\1층", t)
    t = re.sub(r"(\d+)/(\d+)번", r"\1번 \2번", t)
    t = re.sub(r"지하(\d)층", r"지하 \1층", t)
    return re.sub(r"\s+", " ", t).strip()


class Subway:
    def __init__(self, key: str | None = None, nav=None, cache_dir: Path | None = None):
        self.key = key if key is not None else (os.environ.get("KRIC_KEY") or "")
        self.nav = nav                          # 가장 가까운 역 찾기 (TMAP)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict[str, tuple[float, list]] = {}
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

    def _call(self, op: str, r: dict, ttl: float = 86400.0, **extra) -> list[dict]:
        """KRIC 한 번 부르기. 같은 요청은 ttl 동안(기본 하루) 메모리 · 저장 폴더에 둔 것을 다시 쓴다."""
        ck = re.sub(r"[^\w.-]", "_", f"{op}_{r['RAIL_OPR_ISTT_CD']}_{r['LN_CD']}_{r['STIN_CD']}_"
                    + "_".join(f"{k}{v}" for k, v in extra.items()))
        now = time.time()
        hit = self._mem.get(ck)
        if hit and now - hit[0] < ttl:
            return hit[1]
        f = self.cache_dir / f"{ck}.json" if self.cache_dir else None
        if f and f.exists() and now - f.stat().st_mtime < ttl:
            try:
                body = json.loads(f.read_text(encoding="utf-8"))
                self._mem[ck] = (f.stat().st_mtime, body)
                return body
            except Exception:
                pass
        q = urllib.parse.urlencode(dict(serviceKey=self.key, format="json", railOprIsttCd=r["RAIL_OPR_ISTT_CD"],
                                        lnCd=r["LN_CD"], stinCd=r["STIN_CD"], **extra))
        with urllib.request.urlopen(f"{KRIC}/{op}?{q}", timeout=12) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        if (d.get("header") or {}).get("resultCode") == "30":
            raise PermissionError("KRIC 키 오류")
        body = d.get("body") or []
        body = body if isinstance(body, list) else [body]
        self._mem[ck] = (now, body)
        if f:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return body

    def _resolve(self, name, lat, lon) -> tuple[str, list[dict]]:
        if not self.ok:
            raise PermissionError("지하철 정보 키가 없어요 (KRIC_KEY)")
        if not name and lat is not None:
            name = self.nearest(lat, lon)
            if not name:
                raise LookupError("가까운 지하철역을 찾지 못했어요")
        rows = self.stations(name or "")
        if not rows:
            raise LookupError(f"'{name}' 역을 찾지 못했어요")
        return _norm(name or ""), rows

    # ── ① 엘리베이터 ──
    def elevators(self, name: str | None = None, lat: float | None = None, lon: float | None = None) -> dict:
        st, rows = self._resolve(name, lat, lon)
        out, seen = [], set()
        for r in rows:
            for e in self._call("convenientInfo/stationElevator", r, ttl=7 * 86400):
                k = (e.get("exitNo"), e.get("dtlLoc"))
                if k in seen:
                    continue
                seen.add(k)
                fr, to = (e.get("grndDvNmFr"), e.get("runStinFlorFr")), (e.get("grndDvNmTo"), e.get("runStinFlorTo"))
                outside = "지상" in (fr[0], to[0]) and (bool(e.get("exitNo")) or "출" in (e.get("dtlLoc") or ""))
                out.append(dict(line=r["LN_NM"], exit=e.get("exitNo"), place=_place(e.get("dtlLoc") or ""),
                                raw=e.get("dtlLoc"), outside=outside, floors=f"{_floor(*fr)}에서 {_floor(*to)}"))
        return dict(station=st, lines=[r["LN_NM"] for r in rows], elevators=out, sentence=sentence(st, out))

    # ── ② 시간표 기준 다음 열차 ──
    def _neighbors(self, r: dict) -> list[tuple[dict, list[dict]]]:
        """[(바로 옆 역, [그쪽으로 1 · 2 · 3 정거장 떨어진 역들]), …] — 급행은 바로 옆 역에 안 설 수 있어서 더 멀리도 본다."""
        line = sorted((x for x in self.rows if x["RAIL_OPR_ISTT_CD"] == r["RAIL_OPR_ISTT_CD"] and x["LN_CD"] == r["LN_CD"]),
                      key=lambda x: x["STIN_CD"])
        i = next((j for j, x in enumerate(line) if x["STIN_CD"] == r["STIN_CD"]), None)
        if i is None:
            return []
        out = []
        for side in (-1, 1):
            far = [line[i + side * k] for k in (1, 2, 3) if 0 <= i + side * k < len(line)]
            if far:
                out.append((far[0], far))
        return out

    def trains(self, name=None, lat=None, lon=None, now: dt.datetime | None = None) -> dict:
        st, rows = self._resolve(name, lat, lon)
        now = now or dt.datetime.now(KST)
        day = "9" if now.weekday() >= 5 else "8"
        hms = now.hour * 3600 + now.minute * 60 + now.second
        out = []
        for r in rows[:3]:
            tt = self._call("trainUseInfo/subwayTimetable", r, dayCd=day) or \
                self._call("trainUseInfo/subwayTimetable", r, dayCd="8")
            if not tt:
                continue
            mine = {x["trnNo"]: x.get("dptTm") or x.get("arvTm") for x in tt}
            where, fast = {}, set()             # 열차 번호 → 다음 역 이름 (옆 역에서 더 늦게 지나가는 쪽) · 급행
            for first, far in self._neighbors(r):
                label = re.sub(r"\(.*?\)", "", first["STIN_NM"])
                for k, nb in enumerate(far):
                    left = [t for t in mine if t not in where]
                    if not left:
                        break
                    try:
                        other = {x["trnNo"]: x.get("dptTm") or x.get("arvTm")
                                 for x in self._call("trainUseInfo/subwayTimetable", nb, dayCd=day)}
                    except Exception:
                        continue
                    for t in left:
                        if other.get(t) and mine[t] and other[t] > mine[t]:
                            where[t] = label
                            if k > 0:
                                fast.add(t)
                    if k == 0 and len(where) >= len(mine) * 0.95:
                        break
            nxt = []
            for x in tt:
                tm = x.get("dptTm") or x.get("arvTm") or ""
                if len(tm) < 4:
                    continue
                sec = int(tm[:2]) * 3600 + int(tm[2:4]) * 60 + int(tm[4:6] or 0)
                if sec >= hms:
                    nxt.append(dict(no=x["trnNo"], sec=sec, min=round((sec - hms) / 60), to=where.get(x["trnNo"], ""),
                                    express=x["trnNo"] in fast,
                                    time=f"{int(tm[:2]) % 24}시 {int(tm[2:4])}분"))
            nxt.sort(key=lambda a: a["sec"])
            out.append(dict(line=r["LN_NM"], next=nxt[:8]))
        return dict(station=st, lines=out, sentence=train_sentence(st, out))

    # ── ③ 출입구 → 승강장 이동 경로 ──
    def movement(self, name=None, exit_no=None, direction=None, lat=None, lon=None) -> dict:
        st, rows = self._resolve(name, lat, lon)
        paths = {}
        for r in rows:
            for e in self._call("handicapped/stationMovement", r, ttl=7 * 86400):
                p = paths.setdefault((r["LN_CD"], e.get("mvPathMgNo")),
                                     dict(line=r["LN_NM"], start=_step(e.get("stMovePath") or ""),
                                          end=(e.get("edMovePath") or "").strip(), steps=[]))
                p["steps"].append((int(e.get("exitMvTpOrdr") or 0), e.get("mvContDtl") or ""))
        cand = list(paths.values())
        for p in cand:
            p["steps"] = [_step(t) for _, t in sorted(p["steps"])]
        if exit_no:
            cand = [p for p in cand if re.search(rf"(?<!\d){exit_no}(?!\d)", p["start"])] or cand
        if direction:
            cand = [p for p in cand if _norm(direction) in re.sub(r"\s+", "", p["end"])] or cand
        return dict(station=st, paths=cand, sentence=move_sentence(st, cand))


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


def train_sentence(st: str, lines: list[dict]) -> str:
    if not lines or not any(l["next"] for l in lines):
        return f"{st}역은 지금 남은 열차가 없거나 시간표를 받지 못했어요."
    def when(a):                                # 90분 넘게 남으면(막차 뒤 · 새벽) "342분 뒤" 대신 "5시 31분"
        t = "곧" if a["min"] <= 0 else (f"{a['min']}분 뒤" if a["min"] <= 90 else f"{a['time']}")
        return t + (" 급행" if a.get("express") else "")
    parts = []
    for l in lines:
        if not l["next"]:
            continue
        by: dict[str, list] = {}
        known = any(a["to"] for a in l["next"])
        for a in l["next"]:
            if known and not a["to"]:
                continue                        # 방향을 모르는 열차는 방향을 아는 열차가 있으면 빼고 말한다
            by.setdefault(a["to"], []).append(a)
        segs = []
        for to, arr in list(by.items())[:2]:
            s = (f"{to} 쪽은 " if to else "") + when(arr[0])
            if len(arr) > 1:
                s += f", 다음 차 {when(arr[1])}"
            segs.append(s)
        parts.append(f"{l['line']} " + ". ".join(segs))
    return f"{st}역 시간표로 보면, " + ". ".join(parts) + ". 실제 도착은 조금 다를 수 있어요."


def move_sentence(st: str, paths: list[dict]) -> str:
    if not paths:
        return f"{st}역 승강장 가는 길 정보가 없어요."
    if len(paths) > 1:
        starts = list(dict.fromkeys(p["start"] for p in paths))
        ends = list(dict.fromkeys(p["end"] for p in paths))
        n = re.search(r"\d+", starts[0])
        return (f"{st}역은 {', '.join(starts[:4])}에서 출발할 수 있고, 방향은 {', '.join(ends[:4])}이 있어요. "
                f"예를 들어 {st}역 {n.group(0) if n else '1'}번 출구에서 {ends[0]} 가는 길이라고 말해 주세요.")
    p = paths[0]
    steps = " ".join(f"{i}, {s}." for i, s in enumerate(p["steps"], 1))
    return f"{st}역 {p['start']}에서 {p['end']} 승강장까지예요. {steps}"

"""보호자 알림 — 펫캠처럼 '일이 생긴 순간'을 사진과 함께 쌓고, 보호자 화면으로 바로 보낸다.

보행자 앱이 위험 경고를 내면 그 순간의 프레임을 작게 저장하고(박스 표시), 알림 내역에 기록한다.
  · 단계: urgent(멈춰야 하는 위험) · caution(피하세요 등) · info(위치 안내) · sos(보행자가 직접 호출)
          · offline/online(영상 끊김/다시 옴) · battery(보행자 폰 배터리 부족)
  · 묶기: 같은 종류가 12초 안에 또 오면 새 알림이 아니라 '×N'으로 합친다 (펫캠이 움직임을 한 건으로 묶는 것처럼)
  · 저장: alerts/log.jsonl + alerts/snap/<id>.jpg (최근 500건). 사진에는 보행자 주변이 찍히니 이 폴더를 공유하지 말 것.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2

MERGE_S = 12.0           # 같은 종류가 이 안에 또 오면 한 건으로 묶음
OFFLINE_S = 6.0          # 영상이 이만큼 안 오면 '끊김'
KEEP = 500
SNAP_SIDE = 480

LEVEL_SNAP = {"urgent", "caution", "sos"}          # 사진을 남기는 단계


def classify(msg: dict) -> str:
    """warn 메시지 → 보호자 단계. 트리거의 긴급도(prio)와 위험 수준(level)을 그대로 쓴다."""
    prio, level, motion = msg.get("prio", 1), msg.get("level"), msg.get("motion")
    if prio >= 3 or (level == "danger" and motion in ("approaching", "crossing")):
        return "urgent"
    if prio == 2 or level == "danger":
        return "caution"
    return "info"


class AlertCenter:
    def __init__(self, folder: Path, broadcast, log=print):
        self.dir, self.snap_dir = Path(folder), Path(folder) / "snap"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.broadcast, self.log = broadcast, log
        self.lock = threading.Lock()
        self.items: dict[int, dict] = {}
        self.next_id = 1
        self.active, self.zero_since, self.offline_said = False, None, False
        self.low_battery_said = 100
        self._load()

    # ── 저장 ──────────────────────────────────────────
    def _load(self):
        f = self.dir / "log.jsonl"
        if f.exists():
            for ln in f.read_text(encoding="utf-8").splitlines():
                try:
                    a = json.loads(ln)
                    self.items[a["id"]] = a
                except Exception:
                    pass
        ids = sorted(self.items)[-KEEP:]
        self.items = {i: self.items[i] for i in ids}
        self.next_id = (ids[-1] + 1) if ids else 1
        keep = set(ids)
        for p in self.snap_dir.glob("*.jpg"):
            if p.stem.isdigit() and int(p.stem) not in keep:
                p.unlink(missing_ok=True)
        f.write_text("".join(json.dumps(self.items[i], ensure_ascii=False) + "\n" for i in ids), encoding="utf-8")

    def _append(self, a: dict):
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")

    def _snapshot(self, aid: int, image, box=None, W=None) -> bool:
        if image is None:
            return False
        H0, W0 = image.shape[:2]
        s = min(1.0, SNAP_SIDE / max(H0, W0))
        im = cv2.resize(image, (round(W0 * s), round(H0 * s))) if s < 1 else image.copy()
        if box is not None and W:
            k = im.shape[1] / W
            x1, y1, x2, y2 = (int(v * k) for v in box)
            cv2.rectangle(im, (x1, y1), (x2, y2), (60, 60, 255), 3)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            (self.snap_dir / f"{aid}.jpg").write_bytes(buf.tobytes())
        return bool(ok)

    # ── 알림 만들기 ───────────────────────────────────
    def add(self, level: str, text: str, cls: str = "", image=None, box=None, W=None, merge_key=None) -> dict:
        now = time.time()
        with self.lock:
            key = merge_key or (level, cls)
            last = next((a for a in sorted(self.items.values(), key=lambda a: -a["id"])[:8]
                         if a.get("key") == key and now - a["last_ts"] <= MERGE_S), None)
            if last is not None:                                  # 같은 종류가 이어짐 → 한 건으로 묶음
                last.update(count=last["count"] + 1, last_ts=now, text=text)
                a, merged = last, True
            else:
                aid = self.next_id
                self.next_id += 1
                a = dict(id=aid, ts=now, last_ts=now, level=level, text=text, cls=cls, count=1, key=key,
                         snap=(level in LEVEL_SNAP and self._snapshot(aid, image, box, W)))
                self.items[aid] = a
                merged = False
                for old in sorted(self.items)[:-KEEP]:
                    self.items.pop(old, None)
                    (self.snap_dir / f"{old}.jpg").unlink(missing_ok=True)
            self._append(a)
        self.broadcast({"type": "alert", "merged": merged, **{k: v for k, v in a.items() if k != "key"}})
        return a

    # ── 파이프라인 · 서버가 부르는 곳 ─────────────────
    def on_warn(self, msg: dict, ev, image):
        level = classify(msg)
        cls = (ev.cls if ev is not None else msg.get("cls")) or ""
        self.add(level, msg["text"], cls, image, getattr(ev, "box", None), getattr(ev, "W", None))

    def on_sos(self, image):
        self.add("sos", "보행자가 도움을 요청했어요.", "sos", image, merge_key=("sos",))

    def on_client_event(self, ev: dict):
        if ev.get("kind") == "battery":
            pct = int(ev.get("pct", 0))
            if pct < self.low_battery_said:
                self.low_battery_said = pct
                self.add("battery", f"보행자 폰 배터리가 {pct}% 남았어요.", "battery", merge_key=("battery", pct // 10))

    def on_status(self, st: dict):
        """1초마다 오는 상태로 영상이 끊겼는지/다시 왔는지 본다."""
        now = time.time()
        if st.get("fps", 0) > 0:
            self.zero_since = None
            if not self.active:
                self.active = True
                if self.offline_said:
                    self.add("online", "영상이 다시 들어와요.", "online", merge_key=("online", now))
                else:
                    self.add("online", "보행 안내를 시작했어요.", "online", merge_key=("online", now))
                self.offline_said = False
        else:
            if self.zero_since is None:
                self.zero_since = now
            elif self.active and now - self.zero_since >= OFFLINE_S:
                self.active = False
                self.offline_said = True
                self.add("offline", "영상이 끊겼어요. 보행을 끝냈거나 연결에 문제가 있을 수 있어요.", "offline",
                         merge_key=("offline", now))

    # ── 읽기 ──────────────────────────────────────────
    def list(self, since: int = 0) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in a.items() if k != "key"} for a in sorted(self.items.values(), key=lambda a: a["id"])
                    if a["id"] > since]

    def snap(self, aid: int) -> bytes | None:
        p = self.snap_dir / f"{aid}.jpg"
        return p.read_bytes() if p.exists() else None

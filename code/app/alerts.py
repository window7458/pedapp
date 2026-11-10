"""보호자 알림 — 펫캠처럼 '일이 생긴 순간'을 사진과 함께 쌓고, 보호자 화면으로 바로 보낸다.

보행자 앱이 위험 경고를 내면 그 순간의 프레임을 작게 저장하고(박스 표시), 알림 내역에 기록한다.
  · 단계: urgent(멈춰야 하는 위험) · caution(피하세요 등) · info(위치 안내) · sos(보행자가 직접 호출)
          · offline/online(영상 끊김/다시 옴) · battery(보행자 폰 배터리 부족)
  · 묶기: 같은 종류가 12초 안에 또 오면 새 알림이 아니라 '×N'으로 합친다 (펫캠이 움직임을 한 건으로 묶는 것처럼)
  · 장애물 경고는 alerts/evidence/<id>.jpg(박스 없는 원본) + <id>.json(그 프레임의 모델 검출 전체)도 남긴다 — 사람을 장애물로 말하는 원인 확인용.
  · 저장: alerts/log.jsonl + alerts/snap/<id>.jpg. 30일이 지나면 지운다 (교통사고 처리처럼 한 달은 남겨 둠).
  · 사생활: 위치 · 실시간 영상은 '공유 중'일 때만 보호자에게 간다. 공유는 SOS(넘어짐 포함) 때 30분 자동,
    보호자가 요청하고 보행자가 허락하면 10분. 평소엔 알림만, 위치는 긴급 · 호출 알림에 그 순간 한 점만 붙는다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
import time
from pathlib import Path

import cv2

MERGE_S = 12.0           # 같은 종류가 이 안에 또 오면 한 건으로 묶음
OFFLINE_S = 6.0          # 영상이 이만큼 안 오면 '끊김'
KEEP = 5000                 # 안전 상한
RETAIN_S = 30 * 86400       # 30일 보관
SHARE_SOS_S, SHARE_ASK_S = 30 * 60, 10 * 60
LOC_LEVELS = {"urgent", "sos"}   # 공유 중이 아니어도 위치 한 점을 붙이는 단계
SNAP_SIDE = 480

LEVEL_SNAP = {"urgent", "caution", "sos"}          # 사진을 남기는 단계
LEVEL_CLIP = {"urgent", "caution", "sos"}          # 위험 구간을 영상으로 남기는 단계
CLIP_CLASSES = {"person", "car", "bus", "motorcycle", "other_vehicle", "bicycle", "scooter", "sos"}   # 영상은 사람 · 탈것(+보행자 호출)만
EVIDENCE_CLASSES = {"obstacle", "pole", "pole_long", "pole_short"}   # 오분류 확인용 원본을 남기는 이름 (보호자 화면에는 안 나옴)
CLIP_PRE_S, CLIP_POST_S, CLIP_MAX_S, CLIP_SIDE = 5.0, 4.0, 30.0, 640


def classify(msg: dict) -> str:
    """warn 메시지 → 보호자 단계. 트리거의 긴급도(prio)와 위험 수준(level)을 그대로 쓴다."""
    prio, level, motion = msg.get("prio", 1), msg.get("level"), msg.get("motion")
    if prio >= 3 or (level == "danger" and motion in ("approaching", "crossing")):
        return "urgent"
    if prio == 2 or level == "danger":
        return "caution"
    return "info"


class ClipRecorder:
    """위험이 생긴 구간만 영상으로 남긴다 (평소에는 최근 5초만 메모리에 들고 있다가 버림).
    경고 5초 전 ~ 4초 후를 mp4(H.264)로 저장 — 나중에 사고 확인이나 보행자 재확인용.
    같은 위험이 이어지면 끝을 늘린다(최대 30초). 파일: alerts/clips/<알림id>.mp4"""

    def __init__(self, folder: Path, on_done, log=print):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.on_done, self.log = on_done, log
        self.buf: deque = deque()                 # (t, 작은 프레임)
        self.jobs: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.ok = shutil.which("ffmpeg") is not None
        if not self.ok:
            log("[clips] ffmpeg 가 없어 위험 구간 영상을 저장하지 않아요")

    def push(self, im):
        if not self.ok or im is None:
            return
        now = time.time()
        s = min(1.0, CLIP_SIDE / max(im.shape[:2]))
        small = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s))) if s < 1 else im.copy()
        done = []
        with self.lock:
            self.buf.append((now, small))
            while self.buf and now - self.buf[0][0] > CLIP_PRE_S + 1:
                self.buf.popleft()
            for aid, j in self.jobs.items():
                j["frames"].append((now, small))
                if now >= j["end"]:
                    done.append(aid)
            for aid in done:
                j = self.jobs.pop(aid)
                threading.Thread(target=self._write, args=(aid, j["frames"]), daemon=True).start()

    def trigger(self, aid: int):
        if not self.ok:
            return
        now = time.time()
        with self.lock:
            pre = [f for f in self.buf if now - f[0] <= CLIP_PRE_S]
            self.jobs[aid] = dict(frames=pre, start=now, end=now + CLIP_POST_S)

    def extend(self, aid: int):
        with self.lock:
            j = self.jobs.get(aid)
            if j:
                j["end"] = min(time.time() + CLIP_POST_S, j["start"] + CLIP_MAX_S)

    def path(self, aid: int) -> Path:
        return self.dir / f"{aid}.mp4"

    def _write(self, aid: int, frames):
        try:
            if len(frames) < 3:
                return
            dur = max(frames[-1][0] - frames[0][0], 0.5)
            fps = max(2.0, min(15.0, (len(frames) - 1) / dur))
            h, w = frames[0][1].shape[:2]
            raw = self.dir / f"{aid}.raw.mp4"
            vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for _, f in frames:
                vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
            vw.release()
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path(aid))],
                               capture_output=True, timeout=60)
            raw.unlink(missing_ok=True)
            if r.returncode == 0:
                self.on_done(aid)
            else:
                self.log(f"[clips] ffmpeg: {r.stderr.decode(errors='ignore')[:200]}")
        except Exception as e:
            self.log(f"[clips] {e!r}")


class AlertCenter:
    def __init__(self, folder: Path, broadcast, log=print):
        self.dir, self.snap_dir = Path(folder), Path(folder) / "snap"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.broadcast, self.log = broadcast, log
        self.lock = threading.Lock()
        self.items: dict[int, dict] = {}
        self.next_id = 1
        self.active, self.zero_since, self.offline_said = False, None, False
        self.low_battery_said = 100
        self.clips = ClipRecorder(self.dir / "clips", self._clip_done, log)
        self.loc: dict | None = None             # 보행자 폰이 보낸 최근 위치 {lat, lon, acc, ts}
        self.share_until, self.share_reason = 0.0, ""
        self.always_share = False                # 만 13세 미만: 보호자가 항상 위치 · 영상을 본다
        self._pruned = time.time()
        self.on_new = None                       # (alert, merged) — 푸시 알림이 붙는다
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
        old = time.time() - RETAIN_S
        ids = [i for i in sorted(self.items) if self.items[i].get("last_ts", self.items[i].get("ts", 0)) >= old][-KEEP:]
        self.items = {i: self.items[i] for i in ids}
        self.next_id = (ids[-1] + 1) if ids else 1
        keep = set(ids)
        for p in self.snap_dir.glob("*.jpg"):
            if p.stem.isdigit() and int(p.stem) not in keep:
                p.unlink(missing_ok=True)
        for p in (self.dir / "clips").glob("*.mp4"):
            if p.stem.isdigit() and int(p.stem) not in keep:
                p.unlink(missing_ok=True)
        for p in self.evidence_dir.glob("*"):
            if p.stem.isdigit() and int(p.stem) not in keep:
                p.unlink(missing_ok=True)
        for a in self.items.values():
            a["clip"] = (self.dir / "clips" / f"{a['id']}.mp4").exists()
        f.write_text("".join(json.dumps(self.items[i], ensure_ascii=False) + "\n" for i in ids), encoding="utf-8")

    def _prune_old(self):
        """한 시간에 한 번: 30일 지난 알림과 사진 · 영상 · 원본을 지운다."""
        now = time.time()
        if now - self._pruned < 3600:
            return
        self._pruned = now
        with self.lock:
            old = [i for i, a in self.items.items() if a.get("last_ts", a.get("ts", 0)) < now - RETAIN_S]
            if not old:
                return
            for i in old:
                self.items.pop(i, None)
                self._drop_files(i)
            (self.dir / "log.jsonl").write_text("".join(json.dumps(self.items[i], ensure_ascii=False) + "\n"
                                                        for i in sorted(self.items)), encoding="utf-8")

    def _drop_files(self, i: int):
        (self.snap_dir / f"{i}.jpg").unlink(missing_ok=True)
        self.clips.path(i).unlink(missing_ok=True)
        (self.evidence_dir / f"{i}.jpg").unlink(missing_ok=True)
        (self.evidence_dir / f"{i}.json").unlink(missing_ok=True)

    # ── 위치 · 영상 공유 ───────────────────────────────
    def sharing(self) -> bool:
        return self.always_share or time.time() < self.share_until

    def share_state(self) -> dict:
        if self.always_share:
            return {"type": "share", "on": True, "until": 0, "reason": "child"}
        on = self.sharing()
        return {"type": "share", "on": on, "until": self.share_until if on else 0, "reason": self.share_reason if on else ""}

    def set_always(self, on: bool):
        if bool(on) != self.always_share:
            self.always_share = bool(on)
            self.broadcast(self.share_state())

    def share(self, seconds: float, reason: str):
        self.share_until = max(self.share_until, time.time() + seconds)
        self.share_reason = reason
        self.broadcast(self.share_state())
        if self.loc:
            self.broadcast({"type": "loc", **self.loc})

    def share_stop(self):
        self.share_until, self.share_reason = 0.0, ""       # 만 13세 미만이면 always_share 가 남아 계속 공유
        self.broadcast(self.share_state())

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
                if self.loc and (level in LOC_LEVELS or self.sharing()):
                    last["loc"] = self.loc
                self.clips.extend(last["id"])
                a, merged = last, True
            else:
                aid = self.next_id
                self.next_id += 1
                a = dict(id=aid, ts=now, last_ts=now, level=level, text=text, cls=cls, count=1, key=key,
                         snap=(level in LEVEL_SNAP and self._snapshot(aid, image, box, W)))
                if self.loc and now - self.loc.get("ts", 0) < 120 and (level in LOC_LEVELS or self.sharing()):   # 2분 안의 위치만
                    a["loc"] = self.loc
                self.items[aid] = a
                merged = False
                if level in LEVEL_CLIP and cls in CLIP_CLASSES:
                    self.clips.trigger(aid)
                for old in sorted(self.items)[:-KEEP]:
                    self.items.pop(old, None)
                    self._drop_files(old)
            self._append(a)
        out = {k: v for k, v in a.items() if k != "key"}
        self.broadcast({"type": "alert", "merged": merged, **out})
        self._prune_old()
        if self.on_new is not None:
            try:
                self.on_new(out, merged)
            except Exception as e:
                self.log(f"[push] {e!r}")
        return a

    def _clip_done(self, aid: int):
        with self.lock:
            a = self.items.get(aid)
            if not a:
                self.clips.path(aid).unlink(missing_ok=True)
                return
            a["clip"] = True
            self._append(a)
        self.broadcast({"type": "alert", "merged": True, "clip_only": True, **{k: v for k, v in a.items() if k != "key"}})

    # ── 파이프라인 · 서버가 부르는 곳 ─────────────────
    def on_warn(self, msg: dict, ev, image, dets=None):
        level = classify(msg)
        cls = (ev.cls if ev is not None else msg.get("cls")) or ""
        a = self.add(level, msg["text"], cls, image, getattr(ev, "box", None), getattr(ev, "W", None))
        if cls in EVIDENCE_CLASSES and a["count"] == 1:     # 묶인 알림은 첫 장면만
            self._save_evidence(a["id"], msg, ev, image, dets)

    def _save_evidence(self, aid: int, msg: dict, ev, image, dets):
        """실패해도 경고 흐름을 막지 않는다."""
        try:
            if image is not None:
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    (self.evidence_dir / f"{aid}.jpg").write_bytes(buf.tobytes())
            rows = [dict(track=getattr(d, "track_id", None), cls=getattr(d, "cls", None), conf=getattr(d, "conf", None),
                         box=list(getattr(d, "box", ()) or ())) for d in (dets or [])]
            info = dict(id=aid, text=msg.get("text"), spoken_cls=getattr(ev, "cls", None), track=getattr(ev, "track_id", None),
                        box=list(ev.box) if getattr(ev, "box", None) is not None else None,
                        W=getattr(ev, "W", None), H=getattr(ev, "H", None), dets=rows)
            (self.evidence_dir / f"{aid}.json").write_text(json.dumps(info, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            self.log(f"[evidence] {e!r}")

    def on_location(self, m: dict):
        try:
            lat, lon = float(m["lat"]), float(m["lon"])
        except (KeyError, TypeError, ValueError):
            return
        self.loc = dict(lat=round(lat, 6), lon=round(lon, 6), acc=round(float(m.get("acc") or 0), 1), ts=time.time())
        if self.sharing():                       # 평소엔 서버만 알고 있고, 공유 중일 때만 보호자에게
            self.broadcast({"type": "loc", **self.loc})

    def on_sos(self, image, text: str = "보행자가 도움을 요청했어요.", cls: str = "sos"):
        self.share(SHARE_SOS_S, "sos")           # 먼저 공유를 켜야 알림에 위치가 붙는다
        self.add("sos", text + " 30분 동안 위치와 영상을 볼 수 있어요.", cls, image, merge_key=("sos",))

    def on_client_event(self, ev: dict, image=None):
        if ev.get("kind") == "ping":             # "알림 보내줘": 위치 없이 안부만
            self.add("notice", "보행자가 알림을 보냈어요. 잘 가고 있어요.", "ping", merge_key=("ping", int(time.time() // 60)))
        elif ev.get("kind") == "fall":           # 넘어짐 감지 후 30초 동안 대답이 없음 → SOS 와 같게
            self.on_sos(image, "보행자가 넘어진 것 같아요. 30초 동안 대답이 없었어요.", "fall")
        elif ev.get("kind") == "fall_ok":
            self.add("notice", "넘어짐이 감지됐지만 보행자가 괜찮다고 답했어요.", "fall_ok")
        elif ev.get("kind") == "battery":
            pct = int(ev.get("pct", 0))
            if pct < self.low_battery_said:
                self.low_battery_said = pct
                self.add("battery", f"보행자 폰 배터리가 {pct}% 남았어요.", "battery", merge_key=("battery", pct // 10))

    def on_status(self, st: dict):
        """1초마다 오는 상태로 영상이 끊겼는지/다시 왔는지 본다."""
        now = time.time()
        if st.get("rx_fps", st.get("fps", 0)) > 0:     # 받은 프레임 기준: 처리가 잠깐 밀린 건 끊김이 아니다
            self.zero_since = None
            if not self.active:
                self.active = True
                if self.offline_said:
                    self.add("online", "영상이 다시 들어와요.", "online", merge_key=("online", now))
                else:
                    self.add("online", "보행자가 보행을 시작했어요.", "online", merge_key=("online", now))
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

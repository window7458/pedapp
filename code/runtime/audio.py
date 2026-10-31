"""⑦ 재생 큐 — 소리는 한 번에 하나.

  · 경고(warn)는 설명(desc)을 끊고 바로 재생. 경고끼리는 먼저 온 순서 (끊지 않음)
  · 설명은 재생할 차례에 마감 시각이 지났으면 버림 (늦은 설명은 혼란만 줌)
  · 경고도 마감(트리거 + warn_ttl)이 지나도록 못 틀었으면 버림
  · 새 경고가 나가면 그보다 앞선 이벤트의 대기 중 설명은 버림 (최신 정보 우선, drop_superseded=False로 끔)
실제 스피커 재생(--play)은 선택. 끄면 '언제 틀었을지'만 계산해 timeline에 남기고, render.py가 영상에 합친다.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class AudioQueue(threading.Thread):
    def __init__(self, clock, log, play=False, drop_superseded=True):
        super().__init__(daemon=True)
        self.clock, self.log, self.play, self.drop_superseded = clock, log, play, drop_superseded
        self.lock = threading.Lock()
        self.warn_q, self.desc_q = deque(), deque()
        self.cur = None
        self.last_warn_t = -1e9         # 마지막으로 튼 경고의 트리거 시각
        self.timeline = []              # 재생 · 폐기 기록 (render · 채점용)
        self.stopped = False

    def push(self, kind, text, wav, dur, t_event, deadline, **meta):
        item = dict(kind=kind, text=text, wav=str(wav), dur=round(dur, 3), t_event=t_event, deadline=deadline,
                    t_req=round(self.clock.now(), 3), **meta)
        with self.lock:
            (self.warn_q if kind == "warn" else self.desc_q).append(item)
        self.log("audio_req", kind=kind, text=text, t_event=t_event, dur=item["dur"])

    def _start(self, it, now):
        it["t_start"] = round(now, 3)
        it["t_end_plan"] = now + it["dur"]
        self.cur = it
        self.log("audio_start", kind=it["kind"], text=it["text"], t_event=it["t_event"],
                 delay_s=round(now - it["t_event"], 3))
        if self.play:
            import winsound
            winsound.PlaySound(it["wav"], winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

    def _finish(self, now, cut=False):
        it = self.cur
        it["t_end"] = round(min(now, it["t_end_plan"]), 3)
        it["status"] = "cut" if cut else "played"
        self.timeline.append(it)
        self.log("audio_cut" if cut else "audio_end", kind=it["kind"], text=it["text"], t_event=it["t_event"])
        if cut and self.play:
            import winsound
            winsound.PlaySound(None, 0)
        self.cur = None

    def _drop(self, it, now, why):
        it.update(status="dropped", why=why, t_drop=round(now, 3))
        self.timeline.append(it)
        self.log("audio_drop", kind=it["kind"], text=it["text"], t_event=it["t_event"], why=why)

    def run(self):
        while not self.stopped:
            now = self.clock.now()
            with self.lock:
                if self.cur is not None and now >= self.cur["t_end_plan"]:
                    self._finish(now)
                if self.warn_q:
                    if self.cur is not None and self.cur["kind"] == "desc":
                        self._finish(now, cut=True)
                    while self.cur is None and self.warn_q:
                        it = self.warn_q.popleft()
                        if now > it["deadline"]:
                            self._drop(it, now, "warn_late")
                        else:
                            self._start(it, now)
                            self.last_warn_t = max(self.last_warn_t, it["t_event"])
                            if self.drop_superseded:
                                for d in [d for d in self.desc_q if d["t_event"] < it["t_event"]]:
                                    self.desc_q.remove(d)
                                    self._drop(d, now, "superseded")
                if self.cur is None and not self.warn_q:
                    while self.cur is None and self.desc_q:
                        it = self.desc_q.popleft()
                        if now > it["deadline"]:
                            self._drop(it, now, "desc_late")
                        elif self.drop_superseded and it["t_event"] < self.last_warn_t:
                            self._drop(it, now, "superseded")      # 대기 중이 아니라 나중에 도착한 낡은 설명
                        else:
                            self._start(it, now)
            time.sleep(0.005)

    def idle(self):
        with self.lock:
            return self.cur is None and not self.warn_q and not self.desc_q

    def stop(self):
        self.stopped = True

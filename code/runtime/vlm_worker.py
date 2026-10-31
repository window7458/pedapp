"""⑤ VLM — 경고를 막지 않도록 별도 스레드. 입력 이미지는 트리거 순간에 만들어 넘긴다 (그 뒤 프레임은 안 씀).

백엔드 (모두 같은 반환 형식 {"out": JSON 5칸|None, "lat": {...}})
  server : vLLM OpenAI 호환 서버 (A5000) — p0/describe.py 그대로
  pt     : VLM 없음. 트리거 정보만으로 같은 형식의 답 (기준선, 지연 0)
  mock   : pt 답 + 지정한 지연만큼 대기 — 서버 없이 '설명이 늦게 오는' 상황을 시험
작업이 밀리면 가장 최근 것만 남긴다 (늦은 설명은 쓸모가 없음).

하이브리드 후처리 (docs/13 0단계): 대상 · 움직임 · 위험 여부 = VLM, 방향 = 박스 위치, 행동 = 규칙표
"""
from __future__ import annotations

import threading
import time

import core  # noqa: F401
from config import COARSE, MODELS
from frames import _crop, _draw, _resize, window_times
from ko_template import action_ok, direction_of, pt_output, scenario_of


def build_inputs(buffer, frame, ev, setting="F8", bbox_mode="draw"):
    """트리거 시각까지의 프레임 창 → (images, times, boxes_원본px). 마지막 프레임 = 지금 프레임 (원본 해상도)."""
    times = window_times(ev.t, setting)
    imgs, boxes = [], []
    for k, t in enumerate(times):
        last = k == len(times) - 1
        if last:
            im, s = frame.image, 1.0
            box = ev.box
        else:
            got = buffer.image_at(t)
            if got is None:
                continue
            im, s = got
            box = buffer.box_at(ev.track_id, t) if ev.track_id is not None else None
        boxes.append(box)
        bs = None if box is None else tuple(v * s for v in box)
        if bbox_mode in ("draw", "draw+text") and bs is not None:
            im = _draw(im.copy(), bs)
        if bbox_mode == "crop" and bs is not None:
            im = _crop(im, bs)
        imgs.append(_resize(im))
    times = times[-len(imgs):]
    if bbox_mode == "crop":
        full = frame.image if ev.box is None else _draw(frame.image.copy(), ev.box)
        imgs.append(_resize(full, 224))
    return imgs, times, boxes


def hybrid(out: dict, ev) -> dict:
    o = dict(out)
    o.setdefault("hazard", True)                      # 2칸 출력(G3): 위험은 트리거 몫
    if ev.box is not None:
        o["direction"] = direction_of(ev.box, ev.W)
    o.setdefault("direction", "front")
    o.setdefault("action", "caution")
    sc = ev.scenario or scenario_of("other" if ev.cls == "obstacle" else ev.cls)
    o["action"] = (action_ok(sc, o["direction"], o["motion"]) or [o["action"]])[0]
    return o


def coarse_match(out: dict, ev) -> bool:
    return COARSE.get(out.get("target"), "obstacle") == (ev.cls if ev.cls in COARSE else "obstacle")


class PTBackend:
    name = "pt"

    def __call__(self, ev, imgs, times, boxes):
        return {"out": pt_output(dict(cls=ev.cls, **ev.extra), ev.box, ev.W), "lat": {"total_s": 0.0}}


class MockBackend(PTBackend):
    def __init__(self, latency_s=1.0):
        self.latency = latency_s
        self.name = f"mock{latency_s:g}s"

    def __call__(self, ev, imgs, times, boxes):
        time.sleep(self.latency)
        r = super().__call__(ev, imgs, times, boxes)
        r["lat"] = {"total_s": self.latency}
        return r


class ServerBackend:
    def __init__(self, base_url, model_key="qwen2_5-vl-7b", bbox_mode="draw", prompt="v0"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="local")       # 로컬 vLLM — 외부 API 아님
        self.served = MODELS[model_key]["hf"]
        self.bbox_mode, self.prompt = bbox_mode, prompt
        self.name = f"server:{model_key}:{prompt}"

    def __call__(self, ev, imgs, times, boxes):
        from describe import describe
        return describe(self.client, self.served, dict(cls=ev.cls, **ev.extra), imgs, times, boxes,
                        self.bbox_mode, (ev.W, ev.H), prompt=self.prompt)


class VLMWorker(threading.Thread):
    def __init__(self, backend, on_result, log):
        super().__init__(daemon=True)
        self.backend, self.on_result, self.log = backend, on_result, log
        self.cv = threading.Condition()
        self.slot = None
        self.busy = False
        self.stopped = False

    def submit(self, job: dict):
        with self.cv:
            if self.slot is not None:
                self.log("vlm_replaced", ev_t=self.slot["ev"].t, by_ev_t=job["ev"].t)
            self.slot = job
            self.cv.notify()

    def idle(self):
        with self.cv:
            return self.slot is None and not self.busy

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while self.slot is None and not self.stopped:
                    self.cv.wait()
                if self.stopped:
                    return
                job, self.slot, self.busy = self.slot, None, True
            ev = job["ev"]
            self.log("vlm_start", ev_t=ev.t, n_img=len(job["imgs"]))
            try:
                res = self.backend(ev, job["imgs"], job["times"], job["boxes"])
            except Exception as e:                  # 서버 오류가 경고 경로를 멈추면 안 된다
                res = {"out": None, "error": repr(e), "lat": {}}
            self.log("vlm_end", ev_t=ev.t, out=res.get("out"), raw=res.get("raw"), usage=res.get("usage"), lat=res.get("lat"),
                     error=res.get("error"))
            try:
                self.on_result(job, res)
            finally:
                with self.cv:
                    self.busy = False

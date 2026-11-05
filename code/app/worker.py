"""GPU 작업 서버 — 필요할 때만 켜는 GPU 기계에서 돌린다. 상시 서버에 '밖으로' 연결하므로 GPU 쪽에는 공개 주소가 필요 없다.

    python worker.py --hub https://sayeye.xxxx.ts.net --key 접속키 --token 작업서버토큰 --device 0

  · 상시 서버가 보행자 폰 영상(번호 + JPEG)을 보내 주면, 여기서 팀 YOLO + v5_29 히트박스 + VLM 을 돌려
    dets · warn · desc · scene · status 를 돌려보낸다 (판단 코드는 단독 서버와 같은 LivePipeline 그대로).
  · 경고가 나면 '몇 번 프레임에서 무엇을 봤는지'(warn_ctx)도 보내서, 상시 서버가 그 장면으로 보호자 알림을 만든다.
  · 연결이 끊기면 몇 초 뒤 다시 붙는다. 상시 서버가 재시작돼도 알아서 이어진다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def ws_url(hub: str, key: str | None, token: str) -> str:
    base = hub.rstrip("/")
    base = "wss://" + base[8:] if base.startswith("https://") else "ws://" + base[7:] if base.startswith("http://") else base
    q = f"role=worker&wt={quote(token)}" + (f"&k={quote(key)}" if key else "")
    return f"{base}/ws?{q}"


def _num(v):
    return None if v is None else (int(v) if float(v).is_integer() else float(v))


async def run(pipe, sink: dict, url: str, log=print):
    import websockets
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue(maxsize=500)

    def send(m: dict):                         # 파이프라인 스레드 → 이벤트 루프
        def put():
            if out.full():
                try:
                    out.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            out.put_nowait(json.dumps(m, ensure_ascii=False))
        loop.call_soon_threadsafe(put)

    def on_warn(msg, ev, image, dets):
        send({"type": "warn_ctx", "tag": pipe.cur_tag, "msg": msg,
              "ev": {"cls": ev.cls, "box": [float(v) for v in ev.box] if ev.box is not None else None,
                     "W": _num(ev.W), "H": _num(ev.H), "track_id": _num(ev.track_id)},
              "dets": [[_num(d.track_id), d.cls, float(d.conf), *[float(v) for v in d.box]] for d in (dets or [])]})

    sink["emit"] = send
    pipe.on_warn = on_warn
    delay = 2
    while True:
        try:
            async with websockets.connect(url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
                hello = json.loads(await ws.recv())
                if hello.get("type") == "auth_error":
                    log("[worker] 상시 서버가 거절했어요: 접속 키나 작업 서버 토큰을 확인하세요. 30초 뒤 다시 시도해요.")
                    await asyncio.sleep(30)
                    continue
                await ws.send(json.dumps({"type": "worker_info", "info": pipe.info()}, ensure_ascii=False))
                log(f"[worker] 상시 서버에 연결됨 ({url.split('?')[0]})")
                delay = 2
                while not out.empty():
                    out.get_nowait()

                async def sender():
                    last_info = 0.0
                    while True:
                        try:
                            m = await asyncio.wait_for(out.get(), 5.0)
                            await ws.send(m)
                        except asyncio.TimeoutError:
                            pass
                        if time.time() - last_info > 10:          # VLM 이 켜지고 꺼지는 것 등을 알려 준다
                            last_info = time.time()
                            await ws.send(json.dumps({"type": "worker_info", "info": pipe.info()}, ensure_ascii=False))

                task = asyncio.create_task(sender())
                try:
                    async for m in ws:
                        if isinstance(m, bytes):
                            if len(m) < 5:
                                continue
                            seq = struct.unpack("!I", m[:4])[0]
                            im = cv2.imdecode(np.frombuffer(m[4:], np.uint8), cv2.IMREAD_COLOR)
                            if im is not None:
                                pipe.submit(im, tag=seq)
                        else:
                            try:
                                c = json.loads(m)
                            except json.JSONDecodeError:
                                continue
                            if c.get("type") == "set_scene":
                                pipe.set_scene(bool(c.get("on", True)))
                finally:
                    task.cancel()
        except Exception as e:
            log(f"[worker] 연결 끊김: {e!r} — {delay}초 뒤 다시 연결")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30)


def main():
    ap = argparse.ArgumentParser(description="SayEye GPU 작업 서버")
    ap.add_argument("--hub", default=env("SAYEYE_HUB"), help="상시 서버 주소 (https://sayeye.xxxx.ts.net)")
    ap.add_argument("--key", default=env("SAYEYE_KEY"), help="상시 서버 접속 키")
    ap.add_argument("--token", default=env("SAYEYE_WORKER_TOKEN"), help="작업 서버 토큰 (상시 서버 info 에 나옴)")
    ap.add_argument("--weights", default=env("SAYEYE_WEIGHTS", "auto"))
    ap.add_argument("--device", default=env("SAYEYE_DEVICE", "0"))
    ap.add_argument("--trigger", default=env("SAYEYE_TRIGGER", "hitbox"))
    ap.add_argument("--conf", type=float, default=env("SAYEYE_CONF"))
    ap.add_argument("--cam-h", type=float, default=env("SAYEYE_CAM_H"))
    ap.add_argument("--vlm", default=env("SAYEYE_VLM", "server"), choices=["server", "none"])
    ap.add_argument("--vlm-url", default=env("SAYEYE_VLM_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--vlm-model", default=env("SAYEYE_VLM_MODEL", "qwen2_5-vl-7b"))
    ap.add_argument("--scene-interval", type=float, default=float(env("SAYEYE_SCENE_INTERVAL", 5.0)))
    ap.add_argument("--max-fps", type=float, default=float(env("SAYEYE_MAX_FPS", 15.0)))
    a = ap.parse_args()
    if not a.hub or not a.token:
        ap.error("--hub 와 --token 이 필요해요 (상시 서버에서 bash deploy/sayeye.sh info)")
    if a.cam_h:
        os.environ["CAM_H"] = str(a.cam_h)

    from pipeline import Config, LivePipeline, find_weights
    w = find_weights(a.weights)
    if w is None:
        print(f"[오류] 팀 가중치(.pt)를 찾지 못했어요. {APP / 'weights'} 에 final_10class_best.pt 를 넣으세요.", flush=True)
        sys.exit(1)
    vlm = a.vlm
    cfg = Config(perception="yolo", weights=str(w), device=a.device, conf=a.conf, trigger=a.trigger, vlm=vlm,
                 vlm_url=a.vlm_url, vlm_model=a.vlm_model, scene="openai" if vlm == "server" else "none",
                 scene_url=a.vlm_url, scene_interval=a.scene_interval, max_fps=a.max_fps)
    sink = {"emit": lambda m: None}               # 연결되면 상시 서버로 보내는 함수로 바뀐다
    pipe = LivePipeline(cfg, lambda m: sink["emit"](m))
    print(f"가중치={w.name} · 트리거={a.trigger} · 장치={a.device} · VLM={a.vlm_url if vlm == 'server' else '꺼짐'}", flush=True)
    asyncio.run(run(pipe, sink, ws_url(a.hub, a.key, a.token)))


if __name__ == "__main__":
    main()

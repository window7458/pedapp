"""VLM 배선 확인: vLLM 과 같은 모양의 가짜 서버(OpenAI 호환, 스트리밍)를 띄우고
경고(피하세요·멈추세요) → VLM 설명(desc), 특별 상황(scene) 이 실제 클라이언트 코드로 도는지 본다.
진짜 Qwen2.5-VL 이 아니라 응답 모양만 같은 가짜라, 설명 '내용'이 아니라 '연결'만 확인한다."""
import json, os, sys, threading, time
from pathlib import Path
import cv2, numpy as np
os.environ["HB_LIGHTS"] = os.environ["HB_SURFACE"] = "0"
APP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(APP))
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.testclient import TestClient
from fakes import FakePerception  # noqa: E402
from pipeline import Config
from server import build_app, vlm_alive

PORT = 8799
fake = FastAPI()
calls = {"desc": 0, "scene": 0}


@fake.get("/v1/models")
async def models():
    return {"data": [{"id": "Qwen/Qwen2.5-VL-7B-Instruct"}]}


@fake.post("/v1/chat/completions")
async def chat(req: Request):
    j = await req.json()
    if j.get("stream"):                                   # 경고 설명 (SJ describe: json_schema + 스트리밍)
        calls["desc"] += 1
        body = json.dumps({"target": "car", "motion": "approaching"})
        def gen():
            for part in (body[:10], body[10:]):
                yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": j["model"],
                                             "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": j["model"], "choices": [],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    calls["scene"] += 1                                   # 특별 상황
    txt = json.dumps({"special": True, "kind": "cherry_blossom", "new_info": True,
                      "description": "정면으로 벚꽃길이 이어져 있어요."}, ensure_ascii=False)
    return JSONResponse({"id": "y", "object": "chat.completion", "created": 0, "model": j["model"],
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


threading.Thread(target=lambda: uvicorn.run(fake, host="127.0.0.1", port=PORT, log_level="warning"), daemon=True).start()
time.sleep(2)
url = f"http://127.0.0.1:{PORT}/v1"
assert vlm_alive(url) and not vlm_alive("http://127.0.0.1:1/v1")

cfg = Config(perception=FakePerception(), trigger="hitbox", vlm="server", vlm_url=url, scene="openai", scene_url=url,
             scene_interval=1.0, max_fps=30.0, vlm_min_prio=1)   # 가짜 장면은 '다가와요'(긴급도 1)만 나와서 1로 낮춤
rng = np.random.default_rng(2)
base = cv2.GaussianBlur(rng.integers(0, 255, (640, 360, 3), dtype=np.uint8), (0, 0), 3)
seen = {}
with TestClient(build_app(cfg)) as c:
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_source", "mode": "phone"})
        t0, i = time.time(), 0
        while time.time() - t0 < 14:
            s = 1.0 + 0.004 * (i % 60)
            M = np.array([[s, 0, (1 - s) * 180], [0, s, (1 - s) * 320]], np.float32)
            ws.send_bytes(cv2.imencode(".jpg", cv2.warpAffine(base, M, (360, 640), borderMode=cv2.BORDER_REFLECT))[1].tobytes())
            i += 1
            time.sleep(1 / 12)
        t1 = time.time()
        while time.time() - t1 < 4:
            m = ws.receive_json()
            seen.setdefault(m["type"], []).append(m)
warns = seen.get("warn", [])
print("경고:", len(warns), [w["text"] for w in warns[:3]])
print("설명:", [d["text"] for d in seen.get("desc", [])[:3]], "· 특별 상황:", [s_["text"] for s_ in seen.get("scene", [])[:1]])
print("가짜 VLM 호출:", calls)
ok = calls["desc"] > 0 and seen.get("desc") and calls["scene"] > 0 and seen.get("scene")
print("결과:", "통과" if ok else "실패")
sys.exit(0 if ok else 1)

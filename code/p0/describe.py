"""VLM 단계 경계: describe(event, images, ...) → JSON 4칸 + 지연.
모든 비교군(P0–P4)이 이 함수 모양을 맞춘다 (docs/08 §2-1). 서버는 vLLM OpenAI 호환 API — 외부 API를 부르지 않는다."""
from __future__ import annotations

import json
import time

from config import ACTIONS, DIRECTIONS, GEN, MOTIONS, OUTPUT_SCHEMA, OUTPUT_SCHEMA_H2, TARGETS
from frames import to_data_url

SYSTEM = (
    "You are the vision module of a walking assistant for a blind pedestrian. "
    "The camera is worn at eye level and faces the walking direction. "
    "You will see frames ending at the current moment. A hazard detector has flagged one object. "
    "Describe ONLY that object and answer with the JSON schema, choosing each value from the allowed lists."
)

FIELD_GUIDE = (
    f"target: one of {TARGETS}. Use the fine-grained obstacle type when the object is a static obstacle.\n"
    f"direction: where the object is relative to the walking path in the LAST frame, one of {DIRECTIONS}.\n"
    f"motion: one of {MOTIONS}. For moving or static objects use approaching/static/receding/crossing "
    "(relative to the pedestrian); for a pedestrian traffic light use red/green/to_green/to_red; for stairs use up/down.\n"
    f"action: the single best instruction for the pedestrian, one of {ACTIONS}.\n"
    "hazard: true if the object requires a warning right now, otherwise false."
)


def build_prompt(event: dict, times: list[float], boxes: list, bbox_mode: str, img_wh: tuple[int, int],
                 guide: str = FIELD_GUIDE) -> str:
    t0 = times[-1]
    lines = [f"Frames at t = {', '.join(f'{t - t0:+.1f}s' for t in times)} (last = now)."]
    if bbox_mode == "crop":
        lines.append("The first images are close-up crops of the flagged object over time; "
                     "the last image is the full current view with the object in a red box.")
    elif bbox_mode in ("draw", "draw+text"):
        lines.append("The flagged object is marked with a red box.")
    if bbox_mode in ("text", "draw+text") and boxes[-1] is not None:
        W, H = img_wh
        x1, y1, x2, y2 = boxes[-1]
        n = [round(1000 * v) for v in (x1 / W, y1 / H, x2 / W, y2 / H)]
        lines.append(f"Flagged object box in the last frame (x1, y1, x2, y2, normalized 0-1000): {n}.")
    if bbox_mode == "none" or boxes[-1] is None:
        lines.append(f"The detector labeled the flagged object as '{event.get('cls', 'unknown')}'.")
    lines.append(guide)
    return "\n".join(lines)


def describe(client, model: str, event: dict, images: list, times: list[float], boxes: list,
             bbox_mode: str, img_wh: tuple[int, int], prompt: str = "v0") -> dict:
    """스트리밍으로 받아 TTFT를 잰다. 반환: {"out": dict|None, "raw": str, "lat": {...}, "usage": {...}}"""
    t_start = time.perf_counter()
    content = [{"type": "image_url", "image_url": {"url": to_data_url(im)}} for im in images]
    from prompts import PROMPTS, SCHEMA_OF           # 지시문 판 (G2 · G3). v0 = 지금까지 쓴 SYSTEM · FIELD_GUIDE와 같음
    system, guide = PROMPTS[prompt]
    schema = OUTPUT_SCHEMA_H2 if SCHEMA_OF.get(prompt) == "h2" else OUTPUT_SCHEMA
    content.append({"type": "text", "text": build_prompt(event, times, boxes, bbox_mode, img_wh, guide)})
    t_prep = time.perf_counter()

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "hazard_report", "schema": schema, "strict": True}},
        stream=True, stream_options={"include_usage": True}, **GEN)
    raw, t_first, usage = "", None, {}
    for ch in stream:
        if ch.choices and ch.choices[0].delta.content:
            if t_first is None:
                t_first = time.perf_counter()
            raw += ch.choices[0].delta.content
        if getattr(ch, "usage", None):
            usage = {"prompt_tokens": ch.usage.prompt_tokens, "completion_tokens": ch.usage.completion_tokens}
    t_end = time.perf_counter()

    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = None
    return {"out": out, "raw": raw, "usage": usage,
            "lat": {"prep_s": t_prep - t_start,
                    "ttft_s": (t_first or t_end) - t_prep,
                    "decode_s": t_end - (t_first or t_end),
                    "total_s": t_end - t_start}}

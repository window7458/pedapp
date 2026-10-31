"""P0 설정 — 모델 후보 · 프레임 설정 · 출력 어휘. 근거는 docs/11_P0_설계근거.md."""
from __future__ import annotations

# ── 모델 후보 (S-M) ─────────────────────────────────────────────
# viabench_t4 = VIABench Table 4 (정답 트리거 직전 32프레임 → 문장 생성, MPS) 평균 / OA 열 (원문)
# serve      = vLLM 서버 인자. 프레임당 시각 토큰을 약 256개로 맞춘다 (docs/11 §3)
MODELS = {
    "internvl3_5-8b": dict(
        hf="OpenGVLab/InternVL3_5-8B", viabench_t4=(43.9, 40.8),
        serve=["--trust-remote-code", "--hf-overrides", '{"max_dynamic_patch": 1}']),
    "internvl3_5-4b": dict(
        hf="OpenGVLab/InternVL3_5-4B", viabench_t4=(41.4, 28.5),
        serve=["--trust-remote-code", "--hf-overrides", '{"max_dynamic_patch": 1}']),
    "qwen2_5-vl-7b": dict(
        hf="Qwen/Qwen2.5-VL-7B-Instruct", viabench_t4=(34.9, 37.8),
        serve=["--mm-processor-kwargs", '{"max_pixels": 200704}']),      # 448*448 → 16*16 = 256 토큰
    "qwen2_5-vl-3b": dict(
        hf="Qwen/Qwen2.5-VL-3B-Instruct", viabench_t4=(36.8, 23.0),
        serve=["--mm-processor-kwargs", '{"max_pixels": 200704}']),
    "qwen3-vl-8b": dict(
        hf="Qwen/Qwen3-VL-8B-Instruct", viabench_t4=None,                 # VIABench에 없음 → 직접 잰다
        serve=["--mm-processor-kwargs", '{"max_pixels": 262144}']),      # 512*512, patch 16*2 → 256 토큰 (확인 필요)
}

# ── 프레임 설정 (S-F): 논문 설정 그대로 ─────────────────────────
# n = 장 수, fps = 샘플링, 마지막 프레임 = 트리거 시각
FRAMES = {
    "F1":  dict(n=1,  fps=None, src="EgoBlind (NeurIPS'25 D&B) 단일 프레임 실험"),
    "F3":  dict(n=3,  fps=2.0,  src="WalkVLM (ICCV'25) N=3, 2 FPS"),
    "F8":  dict(n=8,  fps=1.0,  src="VIABench Table 5 최소 설정"),
    "F32": dict(n=32, fps=1.0,  src="VIABench Table 4 공통 설정"),
}

# ── 대상 지정 방식 (S1) ─────────────────────────────────────────
BBOX_MODES = ["none", "draw", "text", "draw+text", "crop"]

IMG_LONG_SIDE = 448          # 클라이언트에서 먼저 줄여 보냄 (서버 쪽 토큰 상한과 같은 규모)

# ── 출력 어휘 (docs/09 §3) ──────────────────────────────────────
TARGETS = ["person", "bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle",
           "tree", "pole", "bollard", "sign", "stall", "parked_bicycle", "parked_scooter",
           "bench", "barrier", "stairs", "traffic_light", "other"]
DIRECTIONS = ["left", "front", "right"]
MOTIONS = ["approaching", "static", "receding", "crossing",
           "red", "green", "to_green", "to_red", "up", "down"]
ACTIONS = ["stop", "veer_left", "veer_right", "wait", "cross", "caution"]

# fine → 검출기 10클래스 (채점 시 coarse 비교용)
COARSE = {t: t for t in ["person", "bicycle", "scooter", "motorcycle", "car", "bus",
                         "other_vehicle", "stairs", "traffic_light"]}
COARSE.update({t: "obstacle" for t in ["tree", "pole", "bollard", "sign", "stall", "parked_bicycle",
                                       "parked_scooter", "bench", "barrier", "other"]})

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hazard": {"type": "boolean"},
        "target": {"type": "string", "enum": TARGETS},
        "direction": {"type": "string", "enum": DIRECTIONS},
        "motion": {"type": "string", "enum": MOTIONS},
        "action": {"type": "string", "enum": ACTIONS},
    },
    "required": ["hazard", "target", "direction", "motion", "action"],
    "additionalProperties": False,
}

# G3 하이브리드 정식: VLM은 대상 · 움직임만 (방향 = 상자, 행동 = 규칙표, 위험 = 트리거)
OUTPUT_SCHEMA_H2 = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "enum": TARGETS},
        "motion": {"type": "string", "enum": MOTIONS},
    },
    "required": ["target", "motion"],
    "additionalProperties": False,
}

GEN = dict(temperature=0.0, max_tokens=64)

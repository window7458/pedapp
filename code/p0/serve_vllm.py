"""vLLM 서버 실행 (A5000). 모든 후보를 같은 vLLM 버전 · 같은 공통 인자로 띄운다 (docs/11 §3).

  python serve_vllm.py qwen2_5-vl-7b            # 실행
  python serve_vllm.py internvl3_5-8b --print   # 명령만 출력
"""
from __future__ import annotations

import argparse
import os
import shlex

from config import FRAMES, MODELS

MAX_IMAGES = max(f["n"] for f in FRAMES.values()) + 1        # F32 + crop 모드의 전체 프레임 1장

COMMON = [
    "--port", "8000",
    "--dtype", "bfloat16",
    "--max-model-len", "16384",                               # 33장 × 약 256 토큰 + 프롬프트
    "--limit-mm-per-prompt", f'{{"image": {MAX_IMAGES}}}',
    "--gpu-memory-utilization", "0.90",
    "--seed", "0",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--gpu-util", default="0.90", help="실행 시스템과 GPU를 나눠 쓸 때 낮춤 (YOLO 몫)")
    a = ap.parse_args()
    m = MODELS[a.model]
    # Runyour AI 템플릿엔 CUDA 툴킷이 없고 pip nvcc(13.4)는 런타임 헤더(13.0)와 안 맞음
    # → JIT 빌드가 필요한 flashinfer 샘플러를 끄고 PyTorch 샘플러 사용 (어텐션은 원래 FLASH_ATTN)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    common = [a.gpu_util if c == "0.90" else c for c in COMMON]
    cmd = ["vllm", "serve", m["hf"], *common, *m["serve"]]
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if not a.print:
        os.execvp(cmd[0], cmd)

#!/bin/bash
# 1) Qwen3.5-9B 를 vLLM 으로 띄우고 (뒤에서, 올리기에 몇 분)  2) SJ server.py 를 바로 연다.
#    server.py 는 곧바로 응답하고, /api/health 의 "vlm" 이 true 가 되면 묘사를 할 수 있다 (앱이 미리 깨움).
# 모델 파일은 저장 버킷(/data/models/...)에서 읽는다 — Cloud Run 의 디스크는 메모리라 17.6 GB 를 받아 두면 메모리가 모자란다.
# MODEL_PATH 가 없으면 Hugging Face 에서 받는다(PC · 시험용). 이름은 SJ server.py 가 부르는 VLM_MODEL 로 맞춘다.
# SJ serve_qwen.sh 와 같은 설정. Whisper 를 GPU 에 안 올리므로 메모리 비율만 0.92 → 0.90.
vllm serve "${MODEL_PATH:-${VLM_MODEL:-Qwen/Qwen3.5-9B}}" --served-model-name "${VLM_MODEL:-Qwen/Qwen3.5-9B}" \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":1,"video":0}' &
exec python3 -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8080}"

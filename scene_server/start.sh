#!/bin/bash
# 1) Qwen3.5-9B 를 vLLM 으로 띄우고 (뒤에서, 올리기에 몇 분)  2) SJ server.py 를 바로 연다.
#    server.py 는 곧바로 응답하고, /api/health 의 "vlm" 이 true 가 되면 묘사를 할 수 있다 (앱이 미리 깨움).
# 모델 파일은 저장 버킷(/data/models/...)에서 읽는다 — Cloud Run 의 디스크는 메모리라 17.6 GB 를 받아 두면 메모리가 모자란다.
# MODEL_PATH 가 없으면 Hugging Face 에서 받는다(PC · 시험용). 이름은 SJ server.py 가 부르는 VLM_MODEL 로 맞춘다.
# SJ serve_qwen.sh 와 같은 설정. Whisper 를 GPU 에 안 올리므로 메모리 비율만 0.92 → 0.90.
# 빠르게 올리기 (11/24): Run:ai Model Streamer 로 모델 조각을 여러 개 동시에 읽는다 (버킷 붙인 폴더는 한 줄로 읽으면 느림).
#   STREAMER=0 이면 안 씀. 실패하면 예전 방식으로 한 번 더 띄운다. 걸린 시간은 로그 "[start] vLLM 준비" 줄.
MODEL="${MODEL_PATH:-${VLM_MODEL:-Qwen/Qwen3.5-9B}}"
serve() {
  vllm serve "$MODEL" --served-model-name "${VLM_MODEL:-Qwen/Qwen3.5-9B}" \
    --host 127.0.0.1 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --enforce-eager --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image":1,"video":0}' "$@"
}
(
  T0=$(date +%s)
  ( for i in $(seq 1 900); do
      curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "[start] vLLM 준비 $(( $(date +%s) - T0 ))초"; exit 0; }
      sleep 2
    done ) &
  if [ "${STREAMER:-1}" != "0" ] && python3 -c "import runai_model_streamer" 2>/dev/null; then
    echo "[start] 모델 읽기: runai_streamer (동시 ${STREAMER_CONCURRENCY:-16})"
    serve --load-format runai_streamer \
      --model-loader-extra-config "{\"concurrency\":${STREAMER_CONCURRENCY:-16},\"memory_limit\":${STREAMER_MEM:-4294967296}}" \
      || { echo "[start] runai_streamer 실패 → 기본 방식으로 다시"; serve; }
  else
    echo "[start] 모델 읽기: 기본 방식"
    serve
  fi
) &
exec python3 -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-8080}"

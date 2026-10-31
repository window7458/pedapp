#!/usr/bin/env bash
# S-M / S-F 일괄 실행 (A5000). 모델마다 vLLM 서버를 띄우고 → 준비될 때까지 기다리고 → 정답 시각 23개 추론 → 서버 종료.
#   bash run_sm.sh "qwen2_5-vl-7b internvl3_5-8b ..." "F8" [repeat]
# 결과: results/vlm/p0_<model>_<F>_draw_oracle/  ·  서버 로그: logs/vllm_<model>.log
set -uo pipefail
MODELS=${1:-"qwen2_5-vl-7b qwen2_5-vl-3b internvl3_5-8b internvl3_5-4b qwen3-vl-8b"}
FRAMES=${2:-"F8"}
REPEAT=${3:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=/home/ubuntu/runyourai/vlmcall/logs
export HF_HOME=${HF_HOME:-/home/ubuntu/fast/hf}
source /home/ubuntu/fast/venv/p0/bin/activate
cd "$HERE"

for m in $MODELS; do
  echo "=== $m $(date +%T)"
  python serve_vllm.py "$m" > "$LOGS/vllm_$m.log" 2>&1 &
  SP=$!
  ok=0
  for i in $(seq 1 120); do                                   # 최대 10분
    if curl -sf localhost:8000/v1/models > /dev/null; then ok=1; break; fi
    if ! kill -0 $SP 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! $m 서버 시작 실패 — $LOGS/vllm_$m.log 확인"; tail -5 "$LOGS/vllm_$m.log"
    kill $SP 2>/dev/null; wait $SP 2>/dev/null; continue
  fi
  echo "    ready $(date +%T)"
  for f in $FRAMES; do
    python run_p0.py --events oracle --model "$m" --frames "$f" --bbox draw --repeat "$REPEAT" \
      > "$LOGS/run_${m}_${f}.log" 2>&1 && echo "    done $f $(date +%T)" || { echo "!!! run 실패 $m $f"; tail -5 "$LOGS/run_${m}_${f}.log"; }
  done
  kill $SP; wait $SP 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; pkill -f EngineCore 2>/dev/null
  sleep 5
done
echo "=== all done $(date +%T)"

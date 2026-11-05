#!/usr/bin/env bash
# SayEye GPU 작업 서버 — 필요할 때만 켜는 GPU 기계(A5000 등)에서. 상시 서버에 알아서 연결된다.
#   bash deploy/gpu.sh setup --hub https://sayeye.xxxx.ts.net --key 접속키 --token 작업서버토큰 [--weights 파일 | --weights-gdrive ID]
#        (처음 한 번. 위 명령은 상시 서버에서 'bash deploy/sayeye.sh info' 를 하면 그대로 나와요)
#   bash deploy/gpu.sh start      GPU 를 켤 때마다 (설정은 기억해요)
#   bash deploy/gpu.sh stop       다 쓰고 GPU 를 반납하기 전에 (안 해도 상시 서버가 알아서 CPU 인식으로 바꿔요)
#   bash deploy/gpu.sh status | logs [worker|vllm]
set -uo pipefail
. "$(dirname "$0")/lib.sh"
SELF=$REPO/deploy/gpu.sh
ENVF=$LOCAL/gpu.env
VENV_APP=$LOCAL/venv-gpu
VENV_VLM=$LOCAL/venv-vllm

setup() {
  local HUB="" KEY="" TOKEN="" WSRC="" GID=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --hub) HUB=$2; shift 2 ;; --key) KEY=$2; shift 2 ;; --token) TOKEN=$2; shift 2 ;;
      --weights) WSRC=$2; shift 2 ;; --weights-gdrive) GID=$2; shift 2 ;;
      *) echo "모르는 옵션 $1"; exit 1 ;;
    esac
  done
  if [ ! -f "$ENVF" ] && { [ -z "$HUB" ] || [ -z "$TOKEN" ]; }; then
    echo "--hub 와 --token 이 필요해요. 상시 서버에서 'bash deploy/sayeye.sh info' 에 나온 명령을 그대로 쓰세요."; exit 1
  fi
  set -e
  say "1/4 시스템 패키지"
  apt_base
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[경고] GPU 가 보이지 않아요"
  say "2/4 파이썬 환경 (처음엔 10~20분, 다음부터는 금방)"
  PY=$(pick_python)
  [ -x "$VENV_APP/bin/python" ] || "$PY" -m venv "$VENV_APP"
  "$VENV_APP/bin/pip" install -q -U pip wheel
  "$VENV_APP/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
  "$VENV_APP/bin/pip" install -q -r "$APP/requirements.txt" -r "$APP/requirements-gpu.txt" websockets
  [ -x "$VENV_VLM/bin/python" ] || "$PY" -m venv "$VENV_VLM"
  "$VENV_VLM/bin/pip" install -q -U pip wheel
  "$VENV_VLM/bin/pip" install -q "vllm>=0.11" huggingface_hub
  say "3/4 팀 가중치"
  if ! get_weights "$VENV_APP" "$WSRC" "$GID"; then
    echo "[오류] $APP/weights/ 에 final_10class_best.pt 가 없어요. --weights 또는 --weights-gdrive 로 다시 실행하세요."; exit 1
  fi
  say "4/4 설정 ($ENVF)"
  if [ -n "$HUB" ]; then
    cat > "$ENVF" <<CONF
# SayEye GPU 작업 서버 설정 — 공유하지 마세요
SAYEYE_HUB=$HUB
SAYEYE_KEY=$KEY
SAYEYE_WORKER_TOKEN=$TOKEN
SAYEYE_DEVICE=0
SAYEYE_CAM_H=1.4
SAYEYE_VLM=server
SAYEYE_VLM_URL=http://127.0.0.1:8000/v1
SAYEYE_VLM_MODEL=qwen2_5-vl-7b
VLLM_GPU_UTIL=0.80
HF_HOME=$LOCAL/hf
CONF
    chmod 600 "$ENVF"
  fi
  set +e
  at_reboot "bash $SELF start"
  start
}

run_worker() { loadenv "$ENVF"; cd "$APP"; exec "$VENV_APP/bin/python" worker.py; }
run_vllm() {
  loadenv "$ENVF"; cd "$REPO/code/p0"
  [ "${SAYEYE_VLM:-server}" = "none" ] && exec sleep infinity
  export PATH="$VENV_VLM/bin:$PATH"
  exec "$VENV_VLM/bin/python" serve_vllm.py "${SAYEYE_VLM_MODEL:-qwen2_5-vl-7b}" --gpu-util "${VLLM_GPU_UTIL:-0.80}" --host 127.0.0.1
}

start() {
  [ -f "$ENVF" ] || { echo "먼저 setup 을 해 주세요 (sed -n 2,4p deploy/gpu.sh)"; exit 1; }
  spawn vllm "$SELF" run-vllm
  spawn worker "$SELF" run-worker
  echo "GPU 작업 서버를 켰어요. 몇십 초 안에 상시 서버에 연결돼요 (보행자 앱이 '인식 서버가 연결됐어요'라고 말해요)."
  echo "VLM(설명)은 처음엔 모델을 받느라 10분 이상 걸릴 수 있어요. 그동안도 위험 경고는 나가요."
}
stop() { kill_tree worker; kill_tree vllm; echo "껐어요. 상시 서버는 CPU 인식으로 바뀌어요."; }

case "${1:-}" in
  setup) shift; setup "$@" ;;
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  run-worker) run_worker ;;
  run-vllm) run_vllm ;;
  _forever) forever "$2" "$SELF" "$3" ;;
  status) for n in worker vllm; do alive $n && echo "$n: 실행 중" || echo "$n: 꺼짐"; done
          tail -n 3 "$LOCAL/logs/worker.log" 2>/dev/null ;;
  logs) tail -n 100 "$LOCAL/logs/${2:-worker}.log" ;;
  *) sed -n 2,8p "$SELF"; exit 1 ;;
esac

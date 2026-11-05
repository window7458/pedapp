#!/usr/bin/env bash
# SayEye 서버 관리
#   bash deploy/sayeye.sh info       주소 · 접속 키 · 보호자 확인번호 · 상태
#   bash deploy/sayeye.sh status     서비스 상태
#   bash deploy/sayeye.sh logs       최근 로그 (app | vllm)
#   bash deploy/sayeye.sh restart    설정을 바꾼 뒤 다시 시작
#   bash deploy/sayeye.sh update     git pull 후 다시 시작
#   bash deploy/sayeye.sh start|stop
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOCAL=$REPO/deploy/local
ENVF=$LOCAL/sayeye.env
[ -f "$LOCAL/paths" ] && . "$LOCAL/paths"
VENV_APP=${VENV_APP:-$LOCAL/venv-app}; VENV_VLM=${VENV_VLM:-$LOCAL/venv-vllm}
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
HAS_SYSTEMD=0; [ -d /run/systemd/system ] && [ -f /etc/systemd/system/sayeye.service ] && HAS_SYSTEMD=1
loadenv() { set -a; . "$ENVF"; set +a; }

# ── 실제 실행 (systemd 또는 아래 감시 루프가 부른다) ─────────────
run_app() {
  loadenv; cd "$REPO/code/app"
  PUB=$(public_url); [ -n "$PUB" ] && export SAYEYE_PUBLIC_URL=$PUB
  exec "$VENV_APP/bin/python" server.py
}
run_vllm() {
  loadenv; cd "$REPO/code/p0"
  [ "${SAYEYE_VLM:-server}" = "none" ] && exec sleep infinity
  export PATH="$VENV_VLM/bin:$PATH"
  exec "$VENV_VLM/bin/python" serve_vllm.py "${SAYEYE_VLM_MODEL:-qwen2_5-vl-7b}" --gpu-util "${VLLM_GPU_UTIL:-0.80}" --host 127.0.0.1
}

# ── systemd 가 없는 곳(컨테이너): 죽으면 다시 띄우는 감시 루프 ─────
forever() {   # forever 이름 명령...
  local name=$1; shift
  while true; do
    echo "[$(date '+%F %T')] $name 시작" >> "$LOCAL/logs/$name.log"
    "$@" >> "$LOCAL/logs/$name.log" 2>&1
    echo "[$(date '+%F %T')] $name 종료($?) — 5초 뒤 다시 시작" >> "$LOCAL/logs/$name.log"
    sleep 5
  done
}
alive() { [ -f "$LOCAL/$1.pid" ] && kill -0 "$(cat "$LOCAL/$1.pid")" 2>/dev/null; }
spawn() {     # spawn 이름 하위명령
  alive "$1" && return
  mkdir -p "$LOCAL/logs"
  nohup setsid bash "$0" _forever "$1" "$2" >/dev/null 2>&1 &
  echo $! > "$LOCAL/$1.pid"
}
kill_tree() { alive "$1" && kill -- -"$(cat "$LOCAL/$1.pid")" 2>/dev/null; rm -f "$LOCAL/$1.pid"; }

start_tailscaled() {
  pgrep -x tailscaled >/dev/null && return
  command -v tailscaled >/dev/null || { echo "tailscaled 가 없어요 (install.sh 5단계)"; return; }
  mkdir -p "$LOCAL/logs"; $SUDO mkdir -p /var/lib/tailscale /var/run/tailscale
  # TUN 장치가 없는 컨테이너에서도 되는 사용자 공간 모드 (Funnel 은 이 모드에서도 동작)
  $SUDO nohup tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state \
        --socket=/var/run/tailscale/tailscaled.sock >> "$LOCAL/logs/tailscaled.log" 2>&1 &
  sleep 3
}

public_url() {
  command -v tailscale >/dev/null || return 0
  tailscale status --json 2>/dev/null | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin)["Self"]["DNSName"].rstrip(".")
  print("https://"+d if d else "")
except Exception: pass'
}

info() {
  loadenv
  PUB=$(public_url); [ -z "$PUB" ] && PUB="(Tailscale 주소 없음 — install.sh 의 5단계 확인)"
  H=$(curl -s -m 3 "http://127.0.0.1:${SAYEYE_PORT:-8080}/api/health?k=$SAYEYE_KEY" || true)
  cat <<INFO

  ┌ SayEye ───────────────────────────────────────────────
  │ 보행자 (폰에서 열고 '홈 화면에 추가'):
  │   $PUB/?k=$SAYEYE_KEY
  │ 보호자:
  │   $PUB/guardian?k=$SAYEYE_KEY      확인번호 $SAYEYE_GUARDIAN_PIN
  │ 개발자 화면:  $PUB/dev?k=$SAYEYE_KEY
  │ 상태: ${H:-서버 응답 없음 (bash deploy/sayeye.sh logs app)}
  └───────────────────────────────────────────────────────
  vLLM 은 처음 켤 때 모델(약 16GB)을 받느라 10분 이상 걸릴 수 있어요. 그동안은 경고만 나가고, 켜지면 설명이 자동으로 붙어요.
INFO
}

case "${1:-info}" in
  run-app) run_app ;;
  run-vllm) run_vllm ;;
  _forever) forever "$2" bash "$0" "$3" ;;
  tailscaled) start_tailscaled ;;
  start)
    if [ $HAS_SYSTEMD = 1 ]; then $SUDO systemctl start sayeye-vllm sayeye
    else start_tailscaled; $SUDO tailscale funnel --bg 8080 >/dev/null 2>&1 || true
         spawn vllm run-vllm; spawn app run-app; fi ;;
  stop)
    if [ $HAS_SYSTEMD = 1 ]; then $SUDO systemctl stop sayeye sayeye-vllm
    else kill_tree app; kill_tree vllm; fi ;;
  restart) bash "$0" stop; sleep 2; bash "$0" start; sleep 5; info ;;
  update) git -C "$REPO" pull --ff-only && bash "$0" restart ;;
  status)
    if [ $HAS_SYSTEMD = 1 ]; then systemctl --no-pager status sayeye sayeye-vllm | head -30
    else for n in app vllm; do alive $n && echo "$n: 실행 중" || echo "$n: 꺼짐"; done; fi
    tailscale funnel status 2>/dev/null || true ;;
  logs)
    n=${2:-app}; u=$([ "$n" = vllm ] && echo sayeye-vllm || echo sayeye)
    if [ $HAS_SYSTEMD = 1 ]; then journalctl -u "$u" -n 100 --no-pager
    else tail -n 100 "$LOCAL/logs/$n.log"; fi ;;
  info) info ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac

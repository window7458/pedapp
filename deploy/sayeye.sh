#!/usr/bin/env bash
# SayEye 상시 서버 (Vultr 서울 등 우분투 서버, GPU 없음). 24시간 켜 두고 고정 주소를 맡는다.
#   bash deploy/sayeye.sh install [--weights 파일 | --weights-gdrive 파일ID]   처음 한 번
#   bash deploy/sayeye.sh info       주소 · 접속 키 · 보호자 확인번호 · GPU 연결 명령
#   bash deploy/sayeye.sh status     상태          bash deploy/sayeye.sh logs     최근 로그
#   bash deploy/sayeye.sh restart    설정을 바꾼 뒤  bash deploy/sayeye.sh update   최신 코드 받고 재시작
set -uo pipefail
. "$(dirname "$0")/lib.sh"
SELF=$REPO/deploy/sayeye.sh
ENVF=$LOCAL/sayeye.env
VENV=$LOCAL/venv-hub
HAS_SYSTEMD=0; [ -d /run/systemd/system ] && HAS_SYSTEMD=1

public_url() {
  command -v tailscale >/dev/null || return 0
  tailscale status --json 2>/dev/null | python3 -c 'import json,sys
try:
  d=json.load(sys.stdin)["Self"]["DNSName"].rstrip(".")
  print("https://"+d if d else "")
except Exception: pass'
}

start_tailscaled() {
  [ $HAS_SYSTEMD = 1 ] && { $SUDO systemctl enable --now tailscaled; return; }
  pgrep -x tailscaled >/dev/null && return
  command -v tailscaled >/dev/null || return
  $SUDO mkdir -p /var/lib/tailscale /var/run/tailscale
  $SUDO nohup tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state \
        --socket=/var/run/tailscale/tailscaled.sock >> "$LOCAL/logs/tailscaled.log" 2>&1 &
  sleep 3
}

install() {
  local WSRC="" GID=""
  while [ $# -gt 0 ]; do
    case "$1" in --weights) WSRC=$2; shift 2 ;; --weights-gdrive) GID=$2; shift 2 ;; *) echo "모르는 옵션 $1"; exit 1 ;; esac
  done
  set -e
  say "1/5 시스템 패키지"
  apt_base
  # 메모리 4GB 미만 서버: 설치 · CPU 인식 중 메모리 부족으로 멈추지 않게 스왑 4GB
  if [ "$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)" -lt 4000 ] && ! swapon --show | grep -q /swapfile; then
    $SUDO fallocate -l 4G /swapfile && $SUDO chmod 600 /swapfile && $SUDO mkswap /swapfile >/dev/null && $SUDO swapon /swapfile \
      && echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null && echo "스왑 4GB 추가"
  fi
  say "2/5 파이썬 환경 (CPU, 처음엔 5~10분)"
  PY=$(pick_python); "$PY" --version
  [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip wheel
  "$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  "$VENV/bin/pip" install -q -r "$APP/requirements.txt"
  say "3/5 팀 가중치 (GPU 가 꺼져 있을 때 CPU 인식용 — 없어도 됨)"
  get_weights "$VENV" "$WSRC" "$GID" && ls "$APP"/weights/*.pt || echo "가중치 없음 → GPU 가 꺼져 있을 땐 인식을 쉬어요"
  say "4/5 설정 ($ENVF)"
  if [ ! -f "$ENVF" ]; then
    cat > "$ENVF" <<CONF
# SayEye 상시 서버 설정 — 바꾼 뒤: bash deploy/sayeye.sh restart
# 접속 키 · 확인번호 · 작업 서버 토큰이 있으니 공유하지 마세요 (저장소에는 안 올라가요).
SAYEYE_ENGINE=hub
SAYEYE_KEY=$(rand_key 9)
SAYEYE_GUARDIAN_PIN=$(rand_pin)
SAYEYE_WORKER_TOKEN=$(rand_key 18)
# GPU 가 꺼져 있을 때: cpu(이 서버로 느리게 인식) | none(인식 쉼)
SAYEYE_FALLBACK=cpu
SAYEYE_PORT=8080
SAYEYE_HOST=127.0.0.1
SAYEYE_DEVICE=cpu
SAYEYE_CAM_H=1.4
HB_SURFACE=0
# 애플 푸시 알림은 실제 연락처를 요구해요: mailto:본인메일
SAYEYE_PUSH_SUBJECT=mailto:sayeye@example.com
CONF
    chmod 600 "$ENVF"
  fi
  echo ok
  say "5/5 Tailscale Funnel (고정 HTTPS 주소) + 자동 시작"
  command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | $SUDO sh
  start_tailscaled
  if ! $SUDO tailscale status >/dev/null 2>&1; then
    echo "아래 링크를 브라우저로 열어 Tailscale 에 로그인해 주세요 (가입한 구글 계정)."
    $SUDO tailscale up --hostname=sayeye ${TS_AUTHKEY:+--authkey=$TS_AUTHKEY}
  fi
  echo "※ 'Funnel 을 켜려면 이 링크를 여세요' 가 나오면 열어서 허용한 뒤 install 을 한 번 더 실행하세요."
  $SUDO tailscale funnel --bg 8080
  set +e
  if [ $HAS_SYSTEMD = 1 ]; then
    sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$(id -un)#g" "$REPO/deploy/systemd/sayeye.service" | $SUDO tee /etc/systemd/system/sayeye.service >/dev/null
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable sayeye
    $SUDO systemctl restart sayeye
  else
    start
    at_reboot "bash $SELF start"
  fi
  sleep 8
  info
}

run_app() {
  loadenv "$ENVF"; cd "$APP"
  PUB=$(public_url); [ -n "$PUB" ] && export SAYEYE_PUBLIC_URL=$PUB
  exec "$VENV/bin/python" server.py
}

start() {
  if [ $HAS_SYSTEMD = 1 ] && [ -f /etc/systemd/system/sayeye.service ]; then $SUDO systemctl start sayeye
  else start_tailscaled; $SUDO tailscale funnel --bg 8080 >/dev/null 2>&1 || true; spawn app "$SELF" run-app; fi
}
stop() {
  if [ $HAS_SYSTEMD = 1 ] && [ -f /etc/systemd/system/sayeye.service ]; then $SUDO systemctl stop sayeye
  else kill_tree app; fi
}

info() {
  loadenv "$ENVF"
  PUB=$(public_url); [ -z "$PUB" ] && PUB="https://(Tailscale 주소 없음 — install 다시 실행)"
  H=$(curl -s -m 3 "http://127.0.0.1:${SAYEYE_PORT:-8080}/api/health?k=$SAYEYE_KEY" | python3 -c 'import json,sys
try:
  j=json.load(sys.stdin); e=j.get("engine",{}); print(e.get("label","?"))
except Exception: print("서버 응답 없음 (bash deploy/sayeye.sh logs)")')
  cat <<INFO

  ┌ SayEye 상시 서버 ─────────────────────────────────────
  │ 보행자 (폰에서 열고 '홈 화면에 추가'):
  │   $PUB/?k=$SAYEYE_KEY
  │ 보호자:
  │   $PUB/guardian?k=$SAYEYE_KEY      확인번호 $SAYEYE_GUARDIAN_PIN
  │ 지금 인식: $H
  └───────────────────────────────────────────────────────

  GPU 를 켰을 때 GPU 서버에서 (처음 한 번):
    git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
    bash deploy/gpu.sh setup --hub $PUB --key $SAYEYE_KEY --token $SAYEYE_WORKER_TOKEN --weights <가중치 경로>
  다음부터는: bash deploy/gpu.sh start

INFO
}

case "${1:-info}" in
  install) shift; install "$@" ;;
  run-app) run_app ;;
  _forever) forever "$2" "$SELF" "$3" ;;
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start; sleep 8; info ;;
  update) git -C "$REPO" pull --ff-only && "$VENV/bin/pip" install -q -r "$APP/requirements.txt"; stop; sleep 2; start; sleep 8; info ;;
  status)
    if [ $HAS_SYSTEMD = 1 ] && [ -f /etc/systemd/system/sayeye.service ]; then systemctl --no-pager status sayeye | head -15
    else alive app && echo "상시 서버: 실행 중" || echo "상시 서버: 꺼짐"; fi
    tailscale funnel status 2>/dev/null || true ;;
  logs)
    if [ $HAS_SYSTEMD = 1 ] && [ -f /etc/systemd/system/sayeye.service ]; then journalctl -u sayeye -n 100 --no-pager
    else tail -n 100 "$LOCAL/logs/app.log"; fi ;;
  info) info ;;
  *) sed -n 2,7p "$SELF"; exit 1 ;;
esac

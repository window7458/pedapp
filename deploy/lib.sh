# deploy 스크립트들이 같이 쓰는 함수 (직접 실행하지 않음)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP=$REPO/code/app
LOCAL=$REPO/deploy/local
mkdir -p "$LOCAL/logs"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
rand_key() { python3 -c "import secrets;print(secrets.token_urlsafe(${1:-9}))"; }
rand_pin() { python3 -c "import secrets;print(''.join(secrets.choice('0123456789') for _ in range(6)))"; }
loadenv() { set -a; . "$1"; set +a; }

apt_base() {
  if command -v apt-get >/dev/null; then
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential ffmpeg curl \
      ca-certificates libgl1 libglib2.0-0 cron git >/dev/null
  fi
}
pick_python() { command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3; }

# 팀 가중치: --weights 경로 | --weights-gdrive 파일ID
get_weights() {   # get_weights <venv> <경로> <드라이브ID>
  mkdir -p "$APP/weights"
  [ -n "$2" ] && cp "$2" "$APP/weights/final_10class_best.pt"
  if [ -n "$3" ]; then "$1/bin/pip" install -q gdown && "$1/bin/gdown" -q "$3" -O "$APP/weights/final_10class_best.pt"; fi
  ls "$APP"/weights/*.pt >/dev/null 2>&1
}

# ── systemd 가 없는 곳(컨테이너)에서 죽으면 다시 띄우는 감시 루프 ──
alive() { [ -f "$LOCAL/$1.pid" ] && kill -0 "$(cat "$LOCAL/$1.pid")" 2>/dev/null; }
spawn() {        # spawn <이름> <스크립트> <하위명령>
  alive "$1" && return
  nohup setsid bash "$2" _forever "$1" "$3" >/dev/null 2>&1 &
  echo $! > "$LOCAL/$1.pid"
}
forever() {      # forever <이름> <스크립트> <하위명령>
  local name=$1 script=$2 sub=$3
  while true; do
    echo "[$(date '+%F %T')] $name 시작" >> "$LOCAL/logs/$name.log"
    bash "$script" "$sub" >> "$LOCAL/logs/$name.log" 2>&1
    echo "[$(date '+%F %T')] $name 종료($?) — 5초 뒤 다시 시작" >> "$LOCAL/logs/$name.log"
    sleep 5
  done
}
kill_tree() { alive "$1" && kill -- -"$(cat "$LOCAL/$1.pid")" 2>/dev/null; rm -f "$LOCAL/$1.pid"; }
at_reboot() {    # at_reboot <명령>
  command -v crontab >/dev/null || return 0
  ( crontab -l 2>/dev/null | grep -vF "$1" ; echo "@reboot $1" ) | crontab -
  (command -v service >/dev/null && $SUDO service cron start >/dev/null 2>&1) || true
}

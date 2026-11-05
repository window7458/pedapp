#!/usr/bin/env bash
# SayEye 상시 서버 설치 (GPU 서버에서 한 번만 실행). 끝나면 터미널을 닫아도, 재부팅해도 계속 돌아간다.
#
#   bash deploy/install.sh                      # 처음 설치 (중간에 Tailscale 로그인 링크를 한 번 연다)
#   bash deploy/install.sh --weights /경로/final_10class_best.pt
#   bash deploy/install.sh --weights-gdrive <구글 드라이브 파일 ID>   # '링크가 있는 모든 사용자' 공유일 때
#   TS_AUTHKEY=tskey-... bash deploy/install.sh   # 로그인 링크 대신 Tailscale 인증 키
#
# 하는 일
#   1) 시스템 패키지 (ffmpeg 등)                 2) 파이썬 가상환경 두 개: 앱(YOLO·히트박스) / vLLM(Qwen2.5-VL)
#   3) 팀 가중치 확인                            4) 설정 파일 deploy/local/sayeye.env (접속 키 · 보호자 확인번호 자동 생성)
#   5) Tailscale Funnel: 도메인 없이 고정 HTTPS 주소 (https://<기계이름>.<tailnet>.ts.net)
#   6) 자동 시작: systemd 가 있으면 서비스로, 없으면(컨테이너) 감시 스크립트 + @reboot
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
APP=$REPO/code/app
LOCAL=$REPO/deploy/local
ENVF=$LOCAL/sayeye.env
VENV_APP=${VENV_APP:-$LOCAL/venv-app}
VENV_VLM=${VENV_VLM:-$LOCAL/venv-vllm}
WEIGHTS_SRC="" ; GDRIVE_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --weights) WEIGHTS_SRC=$2; shift 2 ;;
    --weights-gdrive) GDRIVE_ID=$2; shift 2 ;;
    *) echo "모르는 옵션: $1"; exit 1 ;;
  esac
done
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
say() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
mkdir -p "$LOCAL" "$LOCAL/logs"

say "1/6 시스템 패키지"
if command -v apt-get >/dev/null; then
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev build-essential ffmpeg curl ca-certificates libgl1 libglib2.0-0 cron >/dev/null
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[경고] GPU가 보이지 않아요. CPU로 돌면 매우 느려요."
PY=${PYTHON:-$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)}
"$PY" --version

say "2/6 파이썬 환경 (처음엔 10~20분)"
[ -x "$VENV_APP/bin/python" ] || "$PY" -m venv "$VENV_APP"
"$VENV_APP/bin/pip" install -q -U pip wheel
"$VENV_APP/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
"$VENV_APP/bin/pip" install -q -r "$APP/requirements.txt" -r "$APP/requirements-gpu.txt"
[ -x "$VENV_VLM/bin/python" ] || "$PY" -m venv "$VENV_VLM"
"$VENV_VLM/bin/pip" install -q -U pip wheel
"$VENV_VLM/bin/pip" install -q "vllm>=0.11" huggingface_hub

say "3/6 팀 가중치"
mkdir -p "$APP/weights"
if [ -n "$WEIGHTS_SRC" ]; then cp "$WEIGHTS_SRC" "$APP/weights/final_10class_best.pt"; fi
if [ -n "$GDRIVE_ID" ]; then
  "$VENV_APP/bin/pip" install -q gdown
  "$VENV_APP/bin/gdown" -q "$GDRIVE_ID" -O "$APP/weights/final_10class_best.pt"
fi
if ! ls "$APP"/weights/*.pt >/dev/null 2>&1; then
  echo "[오류] $APP/weights/ 에 final_10class_best.pt 가 없어요."
  echo "       --weights 경로 또는 --weights-gdrive 파일ID 로 다시 실행하세요."
  exit 1
fi
ls -la "$APP"/weights/*.pt

say "4/6 설정 파일 ($ENVF)"
if [ ! -f "$ENVF" ]; then
  KEY=$("$VENV_APP/bin/python" -c "import secrets;print(secrets.token_urlsafe(9))")
  PIN=$("$VENV_APP/bin/python" -c "import secrets;print(''.join(secrets.choice('0123456789') for _ in range(6)))")
  cat > "$ENVF" <<CONF
# SayEye 설정 — 바꾼 뒤: bash deploy/sayeye.sh restart
# 이 파일에는 접속 키 · 확인번호가 있으니 공유하지 마세요 (저장소에는 안 올라가요).
SAYEYE_KEY=$KEY
SAYEYE_GUARDIAN_PIN=$PIN
SAYEYE_PORT=8080
SAYEYE_HOST=127.0.0.1
SAYEYE_DEVICE=0
SAYEYE_CAM_H=1.4
SAYEYE_VLM=server
SAYEYE_VLM_URL=http://127.0.0.1:8000/v1
SAYEYE_VLM_MODEL=qwen2_5-vl-7b
# 애플 푸시 알림은 실제 연락처를 요구해요: mailto:본인메일
SAYEYE_PUSH_SUBJECT=mailto:sayeye@example.com
# vLLM 이 GPU 메모리를 얼마나 쓸지 (나머지는 YOLO · 바닥 분할 몫)
VLLM_GPU_UTIL=0.80
HF_HOME=$LOCAL/hf
CONF
  chmod 600 "$ENVF"
fi
cat > "$LOCAL/paths" <<P
REPO=$REPO
VENV_APP=$VENV_APP
VENV_VLM=$VENV_VLM
P
echo "ok"

say "5/6 Tailscale Funnel (고정 HTTPS 주소)"
if ! command -v tailscale >/dev/null; then curl -fsSL https://tailscale.com/install.sh | $SUDO sh; fi
HAS_SYSTEMD=0; [ -d /run/systemd/system ] && HAS_SYSTEMD=1
if [ $HAS_SYSTEMD = 1 ]; then
  $SUDO systemctl enable --now tailscaled
else
  bash "$REPO/deploy/sayeye.sh" tailscaled          # 컨테이너: 사용자 공간 네트워킹으로 띄움
fi
if ! $SUDO tailscale status >/dev/null 2>&1; then
  echo "아래에 나오는 링크를 폰이나 PC 브라우저로 열어 로그인해 주세요 (처음 한 번)."
  $SUDO tailscale up --hostname=sayeye ${TS_AUTHKEY:+--authkey=$TS_AUTHKEY}
fi
# Funnel: 인터넷 → https://sayeye.<tailnet>.ts.net → 이 서버 127.0.0.1:8080
# 처음엔 '관리 화면에서 Funnel 켜기' 링크가 나올 수 있어요. 그 링크를 열어 허용한 뒤 이 스크립트를 다시 실행하세요.
$SUDO tailscale funnel --bg 8080

say "6/6 자동 시작"
if [ $HAS_SYSTEMD = 1 ]; then
  for u in sayeye-vllm sayeye; do
    sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$(id -un)#g" "$REPO/deploy/systemd/$u.service" | $SUDO tee /etc/systemd/system/$u.service >/dev/null
  done
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now sayeye-vllm sayeye
  $SUDO systemctl restart sayeye
else
  bash "$REPO/deploy/sayeye.sh" start
  if command -v crontab >/dev/null; then
    ( crontab -l 2>/dev/null | grep -v 'deploy/sayeye.sh' ; echo "@reboot bash $REPO/deploy/sayeye.sh start" ) | crontab -
    (command -v service >/dev/null && $SUDO service cron start >/dev/null 2>&1) || true
  fi
  echo "[참고] 컨테이너는 컨테이너 자체가 꺼지면 같이 멈춰요. 다시 켜면 @reboot 로 자동 시작돼요"
  echo "       (cron 이 없는 환경이면 bash deploy/sayeye.sh start 를 한 번 실행)."
fi

bash "$REPO/deploy/sayeye.sh" info

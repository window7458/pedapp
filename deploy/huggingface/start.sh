#!/usr/bin/env bash
# Hugging Face Space 가 켜질 때 실행: 최신 코드 받기 → 가중치 받기 → 상시 서버 시작
cd "$HOME/pedapp" || exit 1
git pull --ff-only origin "${SAYEYE_BRANCH:-wip/phone-only-handoff}" 2>&1 | tail -1 || true

miss=""
for v in SAYEYE_KEY SAYEYE_GUARDIAN_PIN SAYEYE_WORKER_TOKEN; do [ -z "${!v:-}" ] && miss="$miss $v"; done
if [ -n "$miss" ]; then
  echo "[오류] Space 의 Settings → Variables and secrets 에 다음 Secret 을 넣어 주세요:$miss"
  echo "       넣은 뒤 Space 를 Restart 하세요."
  exec sleep infinity
fi

W=code/app/weights/final_10class_best.pt
if [ -n "${WEIGHTS_GDRIVE_ID:-}" ] && [ ! -s "$W" ]; then
  echo "팀 가중치 받는 중..."
  gdown -q "$WEIGHTS_GDRIVE_ID" -O "$W" || echo "[알림] 가중치를 받지 못했어요 (드라이브 공유가 '링크가 있는 모든 사용자'인지 확인)"
fi
[ -s "$W" ] || rm -f "$W"

[ -n "${SPACE_HOST:-}" ] && export SAYEYE_PUBLIC_URL="https://$SPACE_HOST"
echo "GPU 서버에서 연결: bash deploy/gpu.sh setup --hub ${SAYEYE_PUBLIC_URL:-https://<space 주소>} --key <SAYEYE_KEY> --token <SAYEYE_WORKER_TOKEN> --weights-gdrive <파일 ID>"
cd code/app && exec python server.py

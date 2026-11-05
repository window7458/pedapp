# SayEye 상시 서버 — Hugging Face Spaces (Docker, 무료 CPU basic) 용.
# 이 파일 하나만 Space 에 올리면 된다. 코드는 깃허브에서 받아 오고, 켤 때마다 최신 코드로 갱신한다.
# 필요한 Secrets (Space → Settings → Variables and secrets):
#   SAYEYE_KEY            접속 키 (폰 주소 뒤 ?k=)
#   SAYEYE_GUARDIAN_PIN   보호자 확인번호 6자리
#   SAYEYE_WORKER_TOKEN   GPU 작업 서버가 붙을 때 쓰는 토큰
#   WEIGHTS_GDRIVE_ID     (선택) 팀 가중치 구글 드라이브 파일 ID — 있으면 GPU 없을 때 CPU 로 인식
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 build-essential curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

ARG REPO=https://github.com/window7458/pedapp.git
ARG BRANCH=wip/phone-only-handoff
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user
RUN git clone --depth 1 -b $BRANCH $REPO pedapp \
    && pip install --no-cache-dir --user -r pedapp/code/app/requirements.txt gdown

ENV SAYEYE_ENGINE=hub SAYEYE_FALLBACK=cpu SAYEYE_PORT=7860 SAYEYE_HOST=0.0.0.0 SAYEYE_DEVICE=cpu \
    SAYEYE_CAM_H=1.4 HB_SURFACE=0 SAYEYE_BRANCH=$BRANCH
EXPOSE 7860
CMD ["bash", "pedapp/deploy/huggingface/start.sh"]

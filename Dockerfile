# SayEye 서버 — Cloud Run 용 (CPU 기본. GPU 는 deploy/deploy_gpu.ps1 이 TORCH_INDEX 를 cu124 로 바꿔 빌드)
FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 YOLO_CONFIG_DIR=/tmp/Ultralytics PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY code/app/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt
COPY code /srv/code
WORKDIR /srv/code/app
# 포트는 Cloud Run 이 PORT 로 준다. 접속 키(SAYEYE_KEY) · 보호자 번호(GUARDIAN_PIN) · DEVICE 는 배포할 때 환경변수로.
CMD ["python", "server.py", "--host", "0.0.0.0"]

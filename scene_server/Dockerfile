# SayEye 풍경 서버 ("주변 설명해줘") — 박상준 SJ 브랜치 "말로 듣는 풍경" (code/voice_demo, fb08855) 를 그대로 Cloud Run GPU 에 올린다.
#   · vLLM 으로 Qwen3.5-9B (SJ serve_qwen.sh 와 같은 설정, 컨테이너 안 127.0.0.1:8000)
#   · SJ server.py (FastAPI, 포트 $PORT) — 앱 서버가 /api/text 로 부른다 (말은 폰이 이미 글로 바꿔서 보냄)
#   · Whisper 는 /api/voice 에서만 쓰므로 가장 작은 모델(tiny)을 CPU 로 — 켜지는 시간을 줄이려고
# 빌드: deploy/deploy_scene.ps1
FROM vllm/vllm-openai:v0.32.0
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_URL=http://127.0.0.1:8000/v1 VLM_MODEL=Qwen/Qwen3.5-9B \
    WHISPER_MODEL=tiny WHISPER_DEVICE=cpu ASR_THREADS=2 LIVE_DIR=/tmp/live
RUN pip install "fastapi" "uvicorn[standard]" "python-multipart" "faster-whisper" "pillow" "hf_transfer"
COPY voice_demo /srv/voice_demo
COPY scene_describe /srv/scene_describe
COPY start.sh /srv/start.sh
WORKDIR /srv/voice_demo
ENTRYPOINT ["/bin/bash", "/srv/start.sh"]

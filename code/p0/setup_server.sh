#!/usr/bin/env bash
# Runyour AI GPU 머신(Ubuntu 22.04 컨테이너)에서 한 번 실행: 환경 + 모델 받기.
#   bash setup_server.sh <저장소 경로> [HF 모델 ...]
#   예) bash setup_server.sh /home/ubuntu/runyourai/vlmcall Qwen/Qwen2.5-VL-7B-Instruct
# 저장소(98 GB)가 빠듯하므로 모델은 실험 단계별로 필요한 것만 받는다 (S0 = Qwen2.5-VL-7B).
set -euo pipefail
WORK=${1:?"저장소 경로를 인자로 주세요"}
shift || true
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(Qwen/Qwen2.5-VL-7B-Instruct)
# 저장소(vlmcall)는 NFS라 작은 파일 쓰기가 매우 느림 → venv·모델은 머신 로컬 디스크(FAST), 로그·결과만 저장소
FAST=${FAST:-/home/ubuntu/fast}
mkdir -p "$WORK"/logs "$FAST"/{hf,venv}

echo "== GPU / 드라이버 =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true

# 시스템 python3는 3.10, 템플릿의 conda python은 3.12 → conda python으로 venv
PY=/opt/conda/bin/python
[ -x "$PY" ] || PY=python3
"$PY" --version

if [ ! -x "$FAST/venv/p0/bin/python" ]; then
  echo "== 가상환경 (저장소 안 → 머신을 반환해도 남음) =="
  "$PY" -m venv "$FAST/venv/p0"
fi
source "$FAST/venv/p0/bin/activate"
pip install -q -U pip
pip install -q "vllm>=0.11" openai opencv-python-headless numpy huggingface_hub

grep -q "HF_HOME" ~/.bashrc || cat >> ~/.bashrc <<EOF
export HF_HOME=$FAST/hf
source $FAST/venv/p0/bin/activate
EOF
export HF_HOME=$FAST/hf

echo "== 모델 받기 =="
for m in "${MODELS[@]}"; do
  hf download "$m" > /dev/null 2>&1 || huggingface-cli download "$m" > /dev/null
  echo "ok  $m"
done

python - <<'EOF'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__)
EOF
du -sh "$FAST"/hf "$FAST"/venv
df -h "$FAST"

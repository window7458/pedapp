#!/bin/sh
# SayEye server on the GPU server (A5000). Start SJ's vLLM first: python code/p0/serve_vllm.py qwen2_5-vl-7b --gpu-util 0.80
cd "$(dirname "$0")"
pip install -r requirements.txt
python server.py --device 0

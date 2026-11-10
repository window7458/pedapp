#!/bin/sh
# SayEye server (laptop). Put final_10class_best.pt in weights/ first.
cd "$(dirname "$0")"
pip install -r requirements.txt
python server.py --device cpu

#!/bin/sh
# SayEye server (laptop). Put final_10class_best.pt in weights/ first.
cd "$(dirname "$0")"
pip install -r requirements.txt
HB_SURFACE=0 python server.py --device cpu --cam-h 1.4

@echo off
rem SayEye server (laptop). Put final_10class_best.pt in weights\ first.
cd /d %~dp0
pip install -r requirements.txt
set HB_SURFACE=0
python server.py --device cpu --cam-h 1.4
pause

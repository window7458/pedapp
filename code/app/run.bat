@echo off
rem SayEye server (laptop). Put final_10class_best.pt in weights\ first.
cd /d %~dp0
pip install -r requirements.txt
python server.py --device cpu
pause

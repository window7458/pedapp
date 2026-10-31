@echo off
rem real model: put final_10class_best.pt in this folder
cd /d %~dp0
pip install -r requirements.txt -r requirements-model.txt
python server.py --https --perception yolo --weights final_10class_best.pt --device 0 --trigger hitbox --cam-h 1.4 --scene none
pause

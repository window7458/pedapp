@echo off
rem 진짜 모델로 실행: final_10class_best.pt 를 이 폴더에 두세요
cd /d %~dp0
python server.py --https --perception yolo --weights final_10class_best.pt --device 0 --trigger hitbox --cam-h 1.4 --scene none
pause

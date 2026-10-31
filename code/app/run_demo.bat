@echo off
rem 모델 없이 화면·흐름만 보기 (가짜 자동차가 다가오며 경고가 뜸)
cd /d %~dp0
pip install -r requirements.txt
python server.py --https --perception demo --trigger hitbox --vlm none --scene mock
pause

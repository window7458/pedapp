@echo off
rem demo: no model, fake car approaches
cd /d %~dp0
pip install -r requirements.txt
python server.py --https --perception demo --trigger hitbox --vlm none --scene mock
pause

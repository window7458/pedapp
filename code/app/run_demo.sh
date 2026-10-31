#!/bin/sh
cd "$(dirname "$0")"
pip install -r requirements.txt
python server.py --https --perception demo --trigger hitbox --vlm none --scene mock

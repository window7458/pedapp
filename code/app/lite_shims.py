"""가벼운 서버판(lite): INFER=onnx 이면 torch · ultralytics 대신 ONNX Runtime 을 쓴다.

팀원 코드(navigation_trigger · tactile_feature)는 고치지 않는다. 그 코드가 `import torch` · `from ultralytics import YOLO` 를
할 때 아래의 작은 대역(stub)을 받도록 sys.modules 에 미리 넣는다. 대역이 하는 일:
  torch.set_num_threads / get_num_threads / cuda.is_available / cuda.synchronize / no_grad / inference_mode
  ultralytics.YOLO → onnx_yolo.YOLO (같은 이름의 .onnx 를 읽음, tools/export_onnx.py 로 만듦)
서버 음성 인식(Whisper, stt.py)은 torch 가 있어야 해서 lite 에서는 끈다 (STT=0, 브라우저 음성 인식이 기본).
"""
from __future__ import annotations

import contextlib
import os
import sys
import types


def active() -> bool:
    return os.environ.get("INFER", "").lower() == "onnx"


def install():
    if not active() or getattr(sys.modules.get("torch"), "_sayeye_stub", False):
        return
    import onnx_yolo
    os.environ.setdefault("STT", "0")

    torch = types.ModuleType("torch")
    torch._sayeye_stub = True
    torch.__version__ = "onnx-stub"
    _n = {"threads": 1}
    torch.set_num_threads = lambda n: _n.__setitem__("threads", int(n))
    torch.get_num_threads = lambda: _n["threads"]
    torch.no_grad = torch.inference_mode = lambda *a, **k: contextlib.nullcontext()
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = onnx_yolo.cuda_ok
    cuda.synchronize = lambda *a, **k: None
    cuda.device_count = lambda: 1 if onnx_yolo.cuda_ok() else 0
    torch.cuda = cuda
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = cuda

    ul = types.ModuleType("ultralytics")
    ul.YOLO = onnx_yolo.YOLO
    ul.__version__ = "onnx-stub"
    sys.modules["ultralytics"] = ul
    print(f"[lite] ONNX Runtime 로 추론: {onnx_yolo.providers()}", flush=True)

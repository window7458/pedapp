"""YOLO 가중치(.pt) → ONNX (가벼운 서버판 'lite' 용, onnx_yolo.py 가 읽는다).

    python tools/export_onnx.py            # weights/ 의 팀 모델 · COCO · 점자블록 세 개
필요: ultralytics, torch, onnx, onnxslim (이 PC 에서 한 번만. 서버판에는 torch · ultralytics 가 없다)

- 입력 크기는 바뀔 수 있게(dynamic): 팀 모델은 960 직사각형 · 640 잘라 보기, 점자블록은 960 정사각형을 같이 쓴다.
- fp32 그대로 (fp16 내보내기는 GPU 가 있어야 함. ONNX Runtime CUDA 에서 fp32 도 충분히 빠르다).
- 클래스 이름 · task · stride 는 ultralytics 가 ONNX 메타데이터로 넣어 준다.
"""
from pathlib import Path
import sys

W = Path(__file__).resolve().parents[1] / "weights"
MODELS = ["final_10class_best.pt", "yolo11n.pt", "yolo11n_tactile.pt"]


def main():
    from ultralytics import YOLO
    for name in (sys.argv[1:] or MODELS):
        pt = W / name
        if not pt.exists():
            print("없음:", pt)
            continue
        out = YOLO(str(pt)).export(format="onnx", dynamic=True, simplify=True, opset=17, imgsz=640)
        print(name, "→", out, f"{Path(out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

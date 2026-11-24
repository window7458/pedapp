"""ONNX 판(onnx_yolo.py) 이 PyTorch 판(ultralytics) 과 같은 결과를 내는지: 박스 · 클래스 · 신뢰도 · 마스크.
이 PC 처럼 torch · ultralytics 가 있는 곳에서 돌린다.   python tests/onnx_parity_test.py [영상.mp4 ...]
기준: 같은 물체끼리 IoU 0.9 넘게 짝지어지는 비율 ≥ 95 %, 신뢰도 차이 < 0.03, 마스크 IoU 평균 ≥ 0.85.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import onnx_yolo  # noqa: E402


def images(argv):
    import ultralytics
    out = [cv2.imread(str(p)) for p in sorted((Path(ultralytics.__file__).parent / "assets").glob("*.jpg"))]
    for v in argv:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for k in np.linspace(0, max(0, n - 1), 6).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            if ok:
                out.append(f)
    return [x for x in out if x is not None]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def compare(name, kw, imgs, use_masks=None):
    from ultralytics import YOLO as TorchYOLO
    pt = APP / "weights" / name
    tm, om = TorchYOLO(str(pt)), onnx_yolo.YOLO(pt)
    matched = total = 0
    dconf, miou, t_t, t_o = [], [], 0.0, 0.0
    for im in imgs:
        t0 = time.perf_counter(); rt = tm.predict(im, verbose=False, device="cpu", **kw)[0]; t_t += time.perf_counter() - t0
        t0 = time.perf_counter(); ro = om.predict(im, **kw)[0]; t_o += time.perf_counter() - t0
        bt, bo = rt.boxes.data.cpu().tolist(), ro.boxes.data.cpu().tolist()
        for i, a in enumerate(bt):
            total += 1
            cand = [(iou(a, b), j) for j, b in enumerate(bo) if int(b[5]) == int(a[5])]
            if cand and max(cand)[0] > 0.9:
                matched += 1
                j = max(cand)[1]
                dconf.append(abs(a[4] - bo[j][4]))
                if use_masks and rt.masks is not None and ro.masks is not None:
                    mt = rt.masks.data.cpu().numpy()[i] > 0.5
                    mo = np.asarray(ro.masks.data)[j] > 0.5
                    if mt.shape == mo.shape:
                        miou.append((mt & mo).sum() / max(1, (mt | mo).sum()))
    rate = matched / max(1, total)
    print(f"{name} {kw}: 박스 {matched}/{total} ({rate:.0%}) · 신뢰도 차이 최대 {max(dconf or [0]):.3f}"
          + (f" · 마스크 IoU 평균 {np.mean(miou):.3f} (n={len(miou)})" if miou else "")
          + f" · CPU 시간 torch {t_t / len(imgs) * 1000:.0f} ms / onnx {t_o / len(imgs) * 1000:.0f} ms")
    ok = rate >= 0.95 and max(dconf or [0]) < 0.03 and (not miou or np.mean(miou) >= 0.85)
    return ok


def main():
    imgs = images(sys.argv[1:])
    print("이미지", len(imgs), "장")
    oks = [
        compare("final_10class_best.pt", dict(imgsz=960, conf=0.08, iou=0.65), imgs, use_masks=True),   # 팀 모델 (트리거)
        compare("final_10class_best.pt", dict(imgsz=640, conf=0.08, iou=0.65), imgs),                   # 잘라 보기
        compare("yolo11n.pt", dict(imgsz=640, conf=0.35), imgs),                                           # 실내 COCO
        compare("yolo11n_tactile.pt", dict(imgsz=960, conf=0.25, iou=0.7, retina_masks=True, classes=[0], max_det=50), imgs, use_masks=True),
    ]
    print("onnx parity", "ok" if all(oks) else "FAIL")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()

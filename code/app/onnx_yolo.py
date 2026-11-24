"""ONNX Runtime 로 YOLO 돌리기 — 가벼운 서버판(lite) 용. PyTorch · ultralytics 없이.

ultralytics.YOLO 에서 SayEye 가 쓰는 것만 같은 모양으로 흉내 낸다 (팀원 코드를 고치지 않으려고):
  model = YOLO("x.pt")            → 옆의 x.onnx 를 읽는다
  model.names · model.task
  r = model.predict(img, imgsz=960, conf=.08, iou=.65, classes=None, max_det=300, retina_masks=False, ...)[0]
  r.boxes.data.cpu().tolist()     → [[x1, y1, x2, y2, conf, cls], ...]  (원본 이미지 좌표)
  r.boxes.conf.detach().cpu().tolist()
  r.masks.xy                      → 물체마다 윤곽 점들 (원본 좌표)        · 분할 모델만
  r.masks.data.detach().cpu().numpy() → (n, H, W) 0/1 (retina_masks=True 면 원본 크기)

전처리 · 후처리는 ultralytics 8.4 와 같게: LetterBox(auto, stride 32, 회색 114) · 클래스별 NMS · 박스 되돌리기 ·
process_mask(입력 크기에서 자르고 키움) / process_mask_native(원본 크기). 이 PC 에서 PyTorch 판과 결과를 비교했다 (tests/onnx_parity_test.py).
"""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

_SESSIONS: dict = {}


class T(np.ndarray):
    """torch 텐서처럼 .cpu() · .detach() · .numpy() 를 부를 수 있는 numpy 배열."""
    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return np.asarray(self)


_PRELOADED = []


def providers():
    import onnxruntime as ort
    if not _PRELOADED:                                  # pip 의 nvidia-* 라이브러리(CUDA · cuDNN)를 먼저 올린다 (ORT 1.21+)
        _PRELOADED.append(True)
        try:
            ort.preload_dlls()
        except Exception:
            pass
    av = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in av]


_CUDA = {}


def cuda_ok() -> bool:
    """정말 GPU 로 도는지: 목록에 CUDA 가 있어도 라이브러리를 못 찾으면 ORT 는 조용히 CPU 로 돈다 → 작은 모델을 열어 확인."""
    if "ok" not in _CUDA:
        try:
            ok = "CUDAExecutionProvider" in providers()
            if ok:
                w = Path(__file__).resolve().parent / "weights" / "yolo11n.onnx"
                ok = (not w.exists()) or _session(str(w)).get_providers()[0] == "CUDAExecutionProvider"
            _CUDA["ok"] = ok
        except Exception:
            _CUDA["ok"] = False
    return _CUDA["ok"]


def _session(path: str):
    if path not in _SESSIONS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[path] = ort.InferenceSession(path, so, providers=providers())
    return _SESSIONS[path]


def letterbox(im, new_shape, stride=32, auto=True):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new_shape[1] - nw, new_shape[0] - nh
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _nms(boxes, scores, iou):
    order = scores.argsort()[::-1]
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0]); yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2]); yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        o = inter / (area[i] + area[r] - inter + 1e-9)
        order = r[o <= iou]
    return np.array(keep, dtype=int)


def _scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
    pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    b = boxes.copy()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b[:, :4] /= gain
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, img0_shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, img0_shape[0])
    return b


def _crop(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = [boxes[:, i][:, None, None] for i in range(4)]
    c = np.arange(w)[None, None, :]
    rr = np.arange(h)[None, :, None]
    return masks * ((c >= x1) & (c < x2) & (rr >= y1) & (rr < y2))


class Boxes:
    def __init__(self, data):
        self.data = T(data)
        self.conf = T(data[:, 4]) if len(data) else T(np.zeros(0, np.float32))


class Masks:
    def __init__(self, data, orig_shape, input_shape=None):
        self.data = T(data)
        self._orig, self._inp = orig_shape, input_shape

    @property
    def xy(self):
        out = []
        for m in np.asarray(self.data):
            cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            seg = np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32) if cs else np.zeros((0, 2), np.float32)
            if self._inp is not None and len(seg):            # 입력(패딩 포함) 크기 마스크 → 원본 좌표
                gain = min(self._inp[0] / self._orig[0], self._inp[1] / self._orig[1])
                px = round((self._inp[1] - self._orig[1] * gain) / 2 - 0.1)
                py = round((self._inp[0] - self._orig[0] * gain) / 2 - 0.1)
                seg[:, 0] = ((seg[:, 0] - px) / gain).clip(0, self._orig[1])
                seg[:, 1] = ((seg[:, 1] - py) / gain).clip(0, self._orig[0])
            out.append(seg)
        return out


class Result:
    def __init__(self, boxes, masks, names, orig_shape):
        self.boxes, self.masks, self.names, self.orig_shape = boxes, masks, names, orig_shape


class YOLO:
    """ultralytics.YOLO 대신. 경로가 .pt 여도 옆의 .onnx 를 쓴다."""
    def __init__(self, path, task=None):
        p = Path(str(path))
        onnx = p if p.suffix == ".onnx" else p.with_suffix(".onnx")
        if not onnx.exists():                                   # 점자블록처럼 다른 폴더의 .pt 면 앱 weights/ 에서 찾는다
            alt = Path(__file__).resolve().parent / "weights" / onnx.name
            onnx = alt if alt.exists() else onnx
        if not onnx.exists():
            raise FileNotFoundError(f"{onnx} 가 없어요 — python tools/export_onnx.py 로 만들어 주세요")
        self.path = str(onnx)
        s = _session(self.path)
        meta = s.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta.get("names", "{}"))
        self.task = meta.get("task", "detect")
        self.stride = int(meta.get("stride", 32))
        self.nc = len(self.names)

    def __call__(self, *a, **k):
        return self.predict(*a, **k)

    def predict(self, source=None, imgsz=640, conf=0.25, iou=0.7, classes=None, max_det=300, retina_masks=False, **_):
        im0 = source
        if not isinstance(im0, np.ndarray):                    # PIL 이미지 (점자블록) → BGR numpy
            im0 = cv2.cvtColor(np.asarray(im0.convert("RGB")), cv2.COLOR_RGB2BGR)
        h0, w0 = im0.shape[:2]
        im = letterbox(im0, imgsz, self.stride, auto=True)
        x = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        outs = _session(self.path).run(None, {"images": x})
        pred = outs[0][0].T                                    # (anchors, 4 + nc + nm)
        nc = self.nc
        scores_all = pred[:, 4:4 + nc]
        cls = scores_all.argmax(1)
        score = scores_all[np.arange(len(pred)), cls]
        keep = score > conf
        if classes is not None:
            keep &= np.isin(cls, classes)
        pred, cls, score = pred[keep], cls[keep], score[keep]
        if len(pred) > 30000:
            top = score.argsort()[::-1][:30000]
            pred, cls, score = pred[top], cls[top], score[top]
        xy = pred[:, :4].copy()
        boxes_in = np.stack([xy[:, 0] - xy[:, 2] / 2, xy[:, 1] - xy[:, 3] / 2, xy[:, 0] + xy[:, 2] / 2, xy[:, 1] + xy[:, 3] / 2], 1)
        k = _nms(boxes_in + cls[:, None] * 7680.0, score, iou)[:max_det] if len(pred) else np.zeros(0, int)
        boxes_in, cls, score, coeffs = boxes_in[k], cls[k], score[k], pred[k, 4 + nc:]
        boxes0 = _scale_boxes(im.shape[:2], boxes_in, (h0, w0))
        data = np.concatenate([boxes0, score[:, None], cls[:, None].astype(np.float32)], 1).astype(np.float32) if len(k) \
            else np.zeros((0, 6), np.float32)
        masks = None
        if self.task == "segment" and len(k):
            protos = outs[1][0]                                # (32, mh, mw)
            c, mh, mw = protos.shape
            m = (coeffs @ protos.reshape(c, -1)).reshape(-1, mh, mw)   # 로짓 (ultralytics 처럼 sigmoid 없이 > 0)
            ih, iw = im.shape[:2]
            if retina_masks:                                   # 원본 크기 (process_mask_native · scale_masks)
                gain = min(mh / h0, mw / w0)
                pw, ph = (mw - round(w0 * gain)) / 2, (mh - round(h0 * gain)) / 2
                top, left = round(ph - 0.1), round(pw - 0.1)
                bottom, right = top + round(h0 * gain), left + round(w0 * gain)
                m = np.stack([cv2.resize(x[top:bottom, left:right], (w0, h0), interpolation=cv2.INTER_LINEAR) for x in m]) > 0
                masks = Masks(_crop(m, boxes0).astype(np.float32), (h0, w0))
            else:                                              # 입력 크기로 키운 뒤 자름 (process_mask upsample)
                m = np.stack([cv2.resize(x, (iw, ih), interpolation=cv2.INTER_LINEAR) for x in m]) > 0
                masks = Masks(_crop(m, boxes_in).astype(np.float32), (h0, w0), (ih, iw))
        return [Result(Boxes(data), masks, self.names, (h0, w0))]

    def track(self, *a, **k):
        raise NotImplementedError("lite 판에서는 track(ByteTrack)을 쓰지 않아요 — 기본 트리거(nav)를 쓰세요")

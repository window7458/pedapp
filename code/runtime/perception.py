"""① 인식 · 추적 공급 + 최근 프레임 버퍼.

ReplayTracks : results/tracks/<model>/<clip>.csv 를 프레임 번호로 꺼냄 (노트북 CPU 시연 · 반복 실험용 — 인식 결과가 매번 같다)
YoloTracker  : E0 YOLO-seg + ByteTrack 을 프레임마다 실행 (서버 · 실제 스트림용). dump_tracks.py 와 같은 설정
FrameBuffer  : VLM 입력용 최근 프레임(0.25 s 간격, 긴 변 960)과 트랙별 박스 이력
"""
from __future__ import annotations

import bisect
import csv
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import core  # noqa: F401  (sys.path 설정)
from common import CLASSES, OUT, WEIGHTS, iter_frames
from core import Det

PROXY = Path.home() / ".cache" / "insight_proxy"       # 수백 MB–GB → OneDrive(결과 폴더) 밖에 둔다


def build_proxy(key, fps=15.0, max_side=1920):
    """재생용 영상: iter_frames가 고르는 프레임만 (긴 변 1920) + 원본 프레임 번호 · 시각 json.
    원본이 4K 60 fps인 클립(c05 등)은 노트북에서 디코딩만으로 실시간을 못 따라감 → 실행 전에 한 번 만들어 둠.
    무손실(libx264rgb crf 0): h264 crf 18로 재압축하면 B0 결과가 10편 중 3편에서 바뀌었다
    (배경 움직임 광류 · 0.2 s 연속 조건이 문턱에 걸려 있음) → 실행 결과 = 오프라인 평가 결과가 되게 화소를 그대로 보존."""
    import imageio_ffmpeg
    PROXY.mkdir(parents=True, exist_ok=True)
    mp4, meta = PROXY / f"{key}_{fps:g}fps_lossless.mkv", PROXY / f"{key}_{fps:g}fps_lossless.json"
    if mp4.exists() and meta.exists():
        return mp4, meta
    ff, idx = None, []
    for fidx, t, im in iter_frames(key, fps, max_side=max_side):
        if ff is None:
            H, W = im.shape[:2]
            ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                                   "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", str(mp4)],
                                  stdin=subprocess.PIPE)
        ff.stdin.write(np.ascontiguousarray(im).tobytes())
        idx.append([fidx, round(t, 6)])
    ff.stdin.close()
    ff.wait()
    json.dump({"key": key, "fps": fps, "frames": idx}, open(meta, "w", encoding="utf-8"))
    return mp4, meta


def needs_proxy(key, max_side=1920):
    from common import clip_info
    i = clip_info(key)
    return max(i["W"], i["H"]) > max_side


def stream_frames(key, fps=15.0):
    """실행용 프레임 공급: (원본 프레임 번호, 영상 시각, BGR) — iter_frames와 같은 프레임.
    원본이 1920 이하면 바로 읽고 (실시간 충분), 4K 등 큰 원본만 무손실 재생용 영상을 쓴다 (디스크 절약)."""
    if not needs_proxy(key):
        yield from iter_frames(key, fps, max_side=1920)
        return
    mp4, meta = build_proxy(key, fps)
    idx = json.load(open(meta, encoding="utf-8"))["frames"]
    cap = cv2.VideoCapture(str(mp4))
    for fidx, t in idx:
        ok, im = cap.read()
        if not ok:
            break
        yield fidx, t, im
    cap.release()


class ReplayTracks:
    name = "replay"

    def __init__(self, model_id: str, key: str):
        self.by = defaultdict(list)
        for r in csv.DictReader(open(OUT / "tracks" / model_id / f"{key}.csv", encoding="utf-8")):
            self.by[int(r["frame"])].append(Det(
                track_id=int(r["track_id"]), cls=r["cls"], conf=float(r["conf"]),
                box=tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")),
                foot=(float(r["foot_u"]), float(r["foot_v"])), edge=int(r["edge_flag"]),
                mask_area=float(r["mask_area"] or 0)))

    def __call__(self, fidx, image):
        return self.by.get(fidx, [])


class YoloTracker:
    name = "yolo"

    def __init__(self, model_id="E0", device="cpu", imgsz=640, conf=0.25):
        from ultralytics import YOLO
        self.m = YOLO(str(WEIGHTS[model_id]))
        self.device, self.imgsz, self.conf = device, imgsz, conf

    def warmup(self, image, n=3):
        """시계를 시작하기 전에 GPU에 모델을 올림. 안 하면 첫 프레임이 약 3 s 걸려 실시간에서 뒤처진다
        (10/8 A5000 실측). 추적기 상태가 생기지 않게 track이 아니라 predict로."""
        for _ in range(n):
            self.m.predict(image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)

    def __call__(self, fidx, image):
        from dump_tracks import cv2_area, foot_point
        H, W = image.shape[:2]
        r = self.m.track(image, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz, conf=self.conf,
                         device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy if r.masks is not None else [None] * len(ids)
        out = []
        for b, tid, c, p, poly in zip(xyxy, ids, cls, conf, polys):
            fu, fv = foot_point(poly, b)
            edge = int(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            area = float(cv2_area(poly)) if poly is not None and len(poly) >= 3 else 0.0
            out.append(Det(int(tid), CLASSES[c], round(float(p), 3), tuple(round(float(v), 1) for v in b),
                           (round(fu, 1), round(fv, 1)), edge, round(area, 1)))
        return out


class FrameBuffer:
    """VLM은 트리거 시각까지의 과거 프레임을 본다 (미래 프레임 없음). F8 = 1 fps × 8 s → 10 s 보관이면 충분."""

    def __init__(self, keep_s=10.0, step_s=0.25, long_side=960):
        self.keep_s, self.step_s, self.long_side = keep_s, step_s, long_side
        self.frames = deque()                    # (t, image_small, scale)
        self.boxes = defaultdict(list)           # tid -> [(t, box_원본px)]

    def _small(self, image):
        s = min(1.0, self.long_side / max(image.shape[:2]))
        im = cv2.resize(image, (round(image.shape[1] * s), round(image.shape[0] * s)),
                        interpolation=cv2.INTER_AREA) if s < 1 else image
        return im, s

    def push(self, frame):
        if frame.image is not None and (not self.frames or frame.t - self.frames[-1][0] >= self.step_s - 1e-3):
            im, s = self._small(frame.image)
            self.frames.append((frame.t, im, s))
        for d in frame.dets:
            self.boxes[d.track_id].append((frame.t, d.box))
        while self.frames and frame.t - self.frames[0][0] > self.keep_s:
            self.frames.popleft()
        if frame.idx % 150 == 0:                 # 오래된 박스 이력 정리
            for tid in list(self.boxes):
                self.boxes[tid] = [x for x in self.boxes[tid] if frame.t - x[0] <= self.keep_s]
                if not self.boxes[tid]:
                    del self.boxes[tid]

    def image_at(self, t):
        """t 이전(포함) 가장 가까운 보관 프레임 → (image_small, scale) 또는 None"""
        best = None
        for ft, im, s in self.frames:
            if ft <= t + 1e-3:
                best = (im, s)
            else:
                break
        return best

    def box_at(self, tid, t, tol=0.25):
        rows = self.boxes.get(tid)
        if not rows:
            return None
        i = bisect.bisect_left(rows, (t,))
        cand = [rows[j] for j in (i - 1, i) if 0 <= j < len(rows)]
        best = min(cand, key=lambda r: abs(r[0] - t))
        return best[1] if abs(best[0] - t) <= tol else None

"""공통: 평가 클립 10편 (VIABench 원본), 입력 정규화, 경로."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parents[1]                      # 인공지능종합설계/
DATA = PROJ / "6주차" / "종설 데이터"
WEIGHTS = {"E0": PROJ / "6주차" / "best_E0.pt", "A5": PROJ / "6주차" / "best_A5.pt"}
OUT = ROOT.parent / "results"          # 종합설계/results

# best_*.pt 는 detection 브랜치의 fm_yolo 모듈을 pickle 안에 참조한다
sys.path.insert(0, str(ROOT / "insight_fm"))

CLASSES = ["person", "bicycle", "scooter", "motorcycle", "car", "bus",
           "other_vehicle", "obstacle", "stairs", "traffic_light"]

# key: 짧은 ID (파일명·그림에 사용) → (영상 파일명, VIABench ID, 회전)
CLIPS = {
    "c01_vehicle_cross":   ("Vehicle 수직.mp4",                          "3_douyin_2_c/0432", None),
    "c02_step":            ("단차.mp4",                                   "3_douyin_2_c/0416", None),
    "c03_offpath_obst":    ("보행경로 아닌 장애물접근.mp4",               "3_douyin_2_c/0438", None),
    "c04_parked_cars":     ("보행경로 옆 주차된 차들.mp4",                "3_douyin_2_c/0464", None),
    "c05_path_obst_recede": ("보행경로 장애물접근 + 멀어지는 vehcile.mp4", "3_douyin_2_c/0443", None),
    "c06_wait_signal":     ("보행자가 멈춰있는경우.mp4",                  "4_bilibili_c/0645", None),
    "c07_light_rg":        ("신호등 변화(빨-초).mp4",                     "3_douyin_2_c/0434", None),
    "c08_light_rgr_night": ("신호등(빨-초-빨).mp4",                       "3_douyin_2_c/0470", None),
    "c09_bike_on_tactile": ("점자위에자전거.mp4",                         "3_douyin_2_c/0503", None),
    "c10_front_obst_night": ("정면 장애물.mp4",                           "3_douyin_2_c/0424", cv2.ROTATE_90_COUNTERCLOCKWISE),
}


def clip_path(key: str) -> Path:
    return DATA / CLIPS[key][0]


def iter_frames(key: str, target_fps: float = 15.0, max_side: int | None = None):
    """시간 기준으로 target_fps에 맞춰 프레임을 뽑는다 (원본 24–60 fps 혼재).
    yield (원본 프레임 번호, t_sec, BGR 이미지[회전 보정 후])"""
    rot = CLIPS[key][2]
    cap = cv2.VideoCapture(str(clip_path(key)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, next_t, i = 1.0 / target_fps, 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = i / fps
        if t + 1e-6 >= next_t:
            ok, im = cap.retrieve()
            if not ok:
                break
            if rot is not None:
                im = cv2.rotate(im, rot)
            if max_side and max(im.shape[:2]) > max_side:
                s = max_side / max(im.shape[:2])
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            yield i, t, im
            next_t += step
        i += 1
    cap.release()


def clip_info(key: str) -> dict:
    cap = cv2.VideoCapture(str(clip_path(key)))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    if CLIPS[key][2] is not None:
        w, h = h, w
    return {"fps": fps, "frames": n, "dur": n / fps, "W": w, "H": h}


os.makedirs(OUT, exist_ok=True)

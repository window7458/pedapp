"""YoloTracker가 쓰는 도우미. 원본 dump_tracks.py가 저장소에 없어 필요한 두 함수만 둔다.
final_10class_best.pt는 검출 모델이라 poly가 없고, 그때 발 위치 = 상자 아래 가운데."""
import cv2
import numpy as np


def foot_point(poly, box):
    """(u, v). 분할 윤곽이 있으면 맨 아래 10 % 점들의 가운데, 없으면 상자 아래 가운데."""
    if poly is not None and len(poly) >= 3:
        p = np.asarray(poly, dtype=np.float32)
        low = p[p[:, 1] >= np.percentile(p[:, 1], 90)]
        return float(low[:, 0].mean()), float(p[:, 1].max())
    return float((box[0] + box[2]) / 2), float(box[3])


def cv2_area(poly):
    return float(cv2.contourArea(np.asarray(poly, dtype=np.float32)))

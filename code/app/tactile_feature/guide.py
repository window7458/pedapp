"""Temporal guidance from segmentation masks; no metric distance or safe-route claim.

Near means sustained image-space overlap with the lower centre of the camera.
Only the user's explicit confirmation completes arrival.
"""
from __future__ import annotations

from collections import deque
import numpy as np
from scipy import ndimage


def components(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        return []
    h, w = mask.shape
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    # Reject very small isolated fragments, without using colour thresholds.
    ids = np.flatnonzero(sizes >= max(32, int(.0007 * w * h)))
    out = []
    for k in ids:
        if not k:
            continue
        selected = labels == k
        ys, xs = np.nonzero(selected)
        bottom = float(np.quantile(ys, .95) / h)
        if bottom < .25:
            continue
        # Aim at a real connected piece of the near end, not the centroid of a
        # branched mask (whose centroid can lie on ordinary pavement).
        band = selected.copy()
        band[:int(np.quantile(ys, .88))] = False
        near_labels, count = ndimage.label(band)
        ends = []
        for j in range(1, count + 1):
            ey, ex = np.nonzero(near_labels == j)
            if len(ex) >= 8:
                ends.append((abs(float(np.median(ex)) / w - .5), -len(ex), ex, ey))
        if not ends:
            continue
        _, _, ex, ey = min(ends, key=lambda z: z[:2])
        # The chosen (x,y) is always an actual predicted pixel.
        distance = (ex - np.median(ex)) ** 2 + (ey - np.median(ey)) ** 2
        q = int(np.argmin(distance))
        target = [float(ex[q] / w), float(ey[q] / h)]
        foot = selected[int(.86*h):int(.98*h), int(.40*w):int(.60*w)]
        near = bottom >= .89 and foot.size > 0 and float(foot.mean()) >= .10
        out.append(dict(mask=selected, target=target, bottom=bottom, near=near,
                        box=np.array([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h]),
                        pixels=len(xs)))
    return out


def box_iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(overlap / max(1e-9, np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - overlap))


class ApproachGuide:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = None
        self.last_seen = -1e9
        self.last_box = None
        self.last_target = None
        self.shape = None
        self.history = deque(maxlen=6)
        self.near_history = deque(maxlen=30)
        self.direction = None
        self.ever_seen = False

    def update(self, mask, now):
        mask = np.asarray(mask, dtype=bool)
        if self.last_time is not None and (now <= self.last_time or now-self.last_time > 3.5 or self.shape != mask.shape):
            self.reset()
        self.last_time, self.shape = now, mask.shape
        choices = components(mask)
        chosen = None
        if self.last_box is not None and now-self.last_seen <= 3.5:
            matched = [(box_iou(c['box'],self.last_box), c) for c in choices
                       if box_iou(c['box'],self.last_box) >= .08
                       or np.linalg.norm(np.array(c['target'])-self.last_target) < .15]
            if matched:
                chosen = max(matched,key=lambda x:x[0])[1]
        elif choices:
            chosen = max(choices, key=lambda c:c['bottom']-.25*abs(c['target'][0]-.5))
        if chosen is None:
            if now-self.last_seen>1.5:self.history.clear()
            self.near_history.clear(); self.direction=None
            if now-self.last_seen > 3.5:
                self.last_box=None; self.last_target=None
            return dict(state='lost' if self.ever_seen else 'searching',
                        text='점자블록을 놓쳤어요. 멈추고 카메라로 다시 찾아 주세요.' if self.ever_seen else '점자블록을 찾고 있어요. 멈춘 상태에서 앞쪽 바닥을 비춰 주세요.',
                        mask=None, target=None, direction=None)
        self.last_box=chosen['box']; self.last_target=np.array(chosen['target']); self.last_seen=now
        while self.history and now-self.history[0]>3.5:self.history.popleft()
        self.history.append(now)
        stable=len(self.history)>=2
        self.ever_seen = self.ever_seen or stable
        x=chosen['target'][0]
        if self.direction=='left' and x<.44: direction='left'
        elif self.direction=='right' and x>.56: direction='right'
        else: direction='left' if x<.38 else 'right' if x>.62 else 'center'
        self.direction=direction
        if chosen['near'] and stable:
            self.near_history.append(now)
        else:
            self.near_history.clear()
        near=len(self.near_history)>=3 and now-self.near_history[0]>=1.2
        if not stable:
            state='acquiring'; text='점자블록 후보를 확인하고 있어요. 잠시 멈춰 주세요.'
        elif near:
            state='near'; text='점자블록이 바로 앞에 있는 것으로 보여요. 멈춘 뒤 지팡이나 발로 확인해 주세요.'
        else:
            state='approach'
            where={'left':'왼쪽','right':'오른쪽','center':'정면'}[direction]
            text=f'점자블록이 {where}에 보여요. 주변을 확인하며 천천히 접근하세요.'
        return dict(state=state,text=text,mask=chosen['mask'],target=chosen['target'],direction=direction)

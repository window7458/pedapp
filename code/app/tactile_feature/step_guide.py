"""One bounded instruction from the current snapshot, without motion estimates.

The camera must face in the same direction as the user's body. A later image
is evaluated afresh: an instruction does not prove that the user moved. Only
stationary near-confirmation observations retain short-lived matching history.
"""
from __future__ import annotations

from collections import deque
import numpy as np
from .guide import components, box_iou


class StepGuide:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = None
        self.shape = None
        self.ever_seen = False
        self._reset_near()

    def _reset_near(self):
        self.near_history = deque(maxlen=30)
        self.near_box = None
        self.near_target = None

    def update(self, mask, now):
        mask = np.asarray(mask, dtype=bool)
        if self.last_time is not None and (
            now <= self.last_time or now - self.last_time > 3.5 or self.shape != mask.shape
        ):
            self._reset_near()
        self.last_time, self.shape = now, mask.shape

        # Select from this image only; do not retain an old position or assume
        # that a previous left/right/forward instruction was followed.
        choices = components(mask)
        if not choices:
            self._reset_near()
            return dict(
                state='lost' if self.ever_seen else 'searching',
                text=(
                    '점자블록을 놓쳤어요. 멈추고 앞쪽 바닥을 다시 비춰 주세요.'
                    if self.ever_seen else '점자블록을 찾고 있어요. 멈춘 상태에서 앞쪽 바닥을 비춰 주세요.'
                ),
                mask=None, target=None, direction=None, action=None,
            )

        chosen = max(choices, key=lambda c: c['bottom'] - .25 * abs(c['target'][0] - .5))
        self.ever_seen = True
        x = chosen['target'][0]
        direction = 'left' if x < .38 else 'right' if x > .62 else 'center'

        if chosen['near']:
            matched = self.near_box is not None and (
                box_iou(chosen['box'], self.near_box) >= .08
                or np.linalg.norm(np.asarray(chosen['target']) - self.near_target) < .15
            )
            if not matched:
                self._reset_near()
            self.near_box = chosen['box'].copy()
            self.near_target = np.asarray(chosen['target']).copy()
            self.near_history.append(now)
            confirmed = len(self.near_history) >= 3 and now - self.near_history[0] >= 1.2
            return dict(
                state='near' if confirmed else 'near_check',
                text='점자블록이 바로 앞에 있는 것으로 보여요. 멈춘 뒤 지팡이나 발로 확인해 주세요.'
                     if confirmed else '멈추세요. 점자블록이 가까운지 다시 확인하고 있어요.',
                mask=chosen['mask'], target=chosen['target'], direction=direction, action=None,
            )

        self._reset_near()
        action, text = {
            'left': ('turn_left', '왼쪽으로 몸을 조금 돌려 주세요.'),
            'right': ('turn_right', '오른쪽으로 몸을 조금 돌려 주세요.'),
            'center': ('forward_step', '앞을 확인하고 정면으로 한 걸음만 천천히 접근하세요.'),
        }[direction]
        return dict(state='approach', text=text, mask=chosen['mask'],
                    target=chosen['target'], direction=direction, action=action)

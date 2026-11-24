"""V3 exploratory detection: one view, confidence >= 0.40, no flip gate.

Fresh detections are displayed directly. Temporal direction/near checks are
separate; detector confidence is not a calibrated correctness probability.
"""
from __future__ import annotations
import numpy as np

CONFIDENCE = 0.40


class LiveDetector:
    def predict(self, model, image):
        result = model._predict_view(image, 960, CONFIDENCE, .70)
        shape = (image.height, image.width)
        scores = [float(s) for s in result['scores']]
        accepted = [m for m, s in zip(result['masks'], scores) if s >= CONFIDENCE]
        mask = np.any(accepted, axis=0) if accepted else np.zeros(shape, dtype=bool)
        return dict(mask=mask, confidence=max((s for s in scores if s >= CONFIDENCE), default=0.0),
                    verified=bool(accepted), views=1)

"""Causal, heuristic parked/moving estimates; does not change hazard decisions.

Speed proxies use a nominal class height, not measured depth or calibrated m/s.
Slow independent motion can look parked. 'parked' means low-motion evidence,
not a semantic claim that the vehicle is legally parked or its engine is off.
"""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MobilityConfig:
    window_s: float = 1.4
    moving_confirm_s: float = .4
    parked_confirm_s: float = .8
    max_gap_s: float = .45
    hold_s: float = .8
    moving_closing_margin: float = 2.0
    moving_lateral_margin: float = .8
    # Only scales image velocity into comparable walking-speed proxy units.
    nominal_heights: tuple = (0., 1.1, 1.1, 1.2, 1.5, 3., 2.)


class MobilityEstimator:
    def __init__(self, config=None):
        self.config = config or MobilityConfig()
        self.states = {}
        self.last = None

    def step(self, frame, decisions):
        t = frame['time_s']; cfg = self.config
        if self.last is not None and t <= self.last:
            raise ValueError('Mobility timestamps must increase')
        self.last = t
        self.states = {k: v for k, v in self.states.items() if t-v['last'] <= 2.}
        result = {}
        for row in decisions:
            cid = row['class_id']
            if cid not in (1, 2, 3, 4, 5, 6):
                continue
            tid = row['track_id']; family = 'wheel' if cid <= 3 else 'vehicle'
            s = self.states.get(tid)
            if s is None or t-s['last'] > cfg.max_gap_s+1e-8 or s['family'] != family:
                s = dict(last=t, family=family, q=deque(), state='unknown', confirmed=-1e9)
                self.states[tid] = s
            s['last'] = t
            candidate, reason, metrics = self._candidate(frame, row)
            hard_reset = reason in ('camera_unreliable', 'rejected_detection', 'height_clipped', 'insufficient_history')
            if hard_reset:
                s['q'].clear(); s['state'] = 'unknown'; s['confirmed'] = -1e9
            s['q'].append((t, candidate))
            while s['q'] and t-s['q'][0][0] > cfg.window_s+1e-8:
                s['q'].popleft()
            # Confirmation uses actual, recent observations, never predicted boxes.
            hits = [ts for ts, c in s['q'] if c == candidate]
            needed = cfg.moving_confirm_s if candidate == 'moving' else cfg.parked_confirm_s
            confirmed = (candidate != 'unknown' and len(hits) >= 3
                         and hits[-1]-hits[0] >= needed-1e-8
                         and len(hits)/len(s['q']) >= .7)
            held = False
            if confirmed:
                s['state'] = candidate; s['confirmed'] = t
            elif t-s['confirmed'] <= cfg.hold_s+1e-8 and not hard_reset:
                held = True
            else:
                s['state'] = 'unknown'
            state = s['state']
            result[tid] = dict(motion_state=state,
                motion_label={'parked': '주차·정차 추정', 'moving': '이동 추정', 'unknown': '판정 보류'}[state],
                motion_reason='held_recent_state' if held else reason if state == 'unknown' or confirmed else 'confirming',
                motion_candidate=candidate, motion_is_estimate=True,
                motion_confirmed_age_s=round(t-s['confirmed'],4) if state!='unknown' else None,
                motion_evidence=metrics)
            if state == 'unknown' and candidate != 'unknown':
                result[tid]['motion_reason'] = 'confirming'
        return result

    def _candidate(self, f, r):
        cfg = self.config; ctx = f['context']; w, h = f['width'], f['height']
        x1, y1, x2, y2 = r['box']; bh = y2-y1; bw = x2-x1
        hg, wg = r['height_growth'], r['width_growth']
        height = cfg.nominal_heights[r['class_id']]
        focal = ctx['focal_px']; scale = height*min(w, h)/max(bh, 1.)
        closing = height*focal/max(bh, 1.)*hg
        lateral = scale*abs(r['vx'])
        residual = scale*r['residual_speed']
        still = bool(ctx.get('stationary') or ctx.get('walk_valid') and ctx['walk_mps'] < .3)
        walk_valid = bool(ctx.get('walk_valid') or still)
        walk = 0. if still else float(ctx['walk_mps'])
        side_clipped = x1 < .005*w or x2 > .995*w
        metrics = dict(closing_speed_proxy=round(closing, 4), lateral_speed_proxy=round(lateral, 4),
            residual_speed_proxy=round(residual, 4), walk_speed_estimate=round(walk, 4),
            walk_valid=walk_valid, wearer_stationary=still, height_growth=round(hg, 4),
            side_clipped=side_clipped, speed_is_metric_measurement=False)
        def out(state, reason):
            return state, reason, metrics
        if ctx.get('camera_shock') or not f.get('camera', {}).get('camera_ok', False):
            return out('unknown', 'camera_unreliable')
        if r.get('reason') in ('wearer_equipment', 'bottom_clipped_two_wheel_unconfirmed', 'overlapping_person_hypothesis'):
            return out('unknown', 'rejected_detection')
        if y1 < .005*h or y2 > .985*h:
            return out('unknown', 'height_clipped')
        if (r.get('observations', 0) < 3 or r.get('history_span_s', 0) < .35-1e-8
                or not r.get('history_ready', False)):
            return out('unknown', 'insufficient_history')
        if r.get('median_confidence', r['confidence']) < .4 or r['confidence'] < .3:
            return out('unknown', 'low_confidence')
        if bh < .045*h or bw < .025*w:
            return out('unknown', 'too_small_for_motion')
        if not all(math.isfinite(v) for v in (closing, lateral, residual, walk, hg, wg)):
            return out('unknown', 'invalid_features')
        if not side_clipped and abs(wg-hg) > 1.5:
            return out('unknown', 'box_shape_unstable')
        if not walk_valid:
            return out('unknown', 'walking_speed_unavailable')
        # Side-clipped boxes grow sideways as they enter view; do not interpret
        # their artificial centre movement as independent lateral travel.
        if not side_clipped and lateral > max(1.5, walk+cfg.moving_lateral_margin):
            return out('moving', 'faster_lateral_motion')
        if closing > max(3.2, walk+cfg.moving_closing_margin) and hg > .12 and wg > .05:
            return out('moving', 'faster_than_walking_approach')
        if still and not side_clipped and abs(r['vx']) > .06 and lateral > .6:
            return out('moving', 'motion_while_wearer_stopped')
        if still and abs(closing) > 1.2 and abs(hg) > .15 and hg*wg > 0:
            return out('moving', 'size_change_while_wearer_stopped')
        # Recession is ambiguous while the camera is walking/turning.
        if (lateral <= .7 and residual <= max(.9, .6*walk)
                and -max(.8, .5*walk) <= closing <= max(1.2, walk+1.)
                and abs(hg) <= (.15 if still else .65)):
            return out('parked', 'low_motion_within_walking_range')
        return out('unknown', 'ambiguous_speed')


def annotate_mobility(estimator, debug, events):
    """Common path for live output and saved-observation evaluation."""
    labels = estimator.step(debug, debug['decisions'])
    for collection in (debug['decisions'], debug.get('detections', []), events):
        for item in collection:
            if item.get('class_id') in (1, 2, 3, 4, 5, 6) and item.get('track_id') in labels:
                item.update(labels[item['track_id']])
    return labels

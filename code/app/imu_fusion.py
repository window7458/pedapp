"""폰 센서(가속도 · 자이로)를 영상 프레임 시각에 맞춰 트리거에 넣는다.

시간 맞추기
- 폰은 프레임마다 찍은 시각(폰 시계, ms)을 붙여 보내고(SEF1 + 시각 + JPEG), 센서 묶음도 같은 폰 시계로 0.1초마다 보낸다.
- 둘 다 폰 시계라 서버 · 폰 시계 차이와 네트워크 지연이 상관없다. 프레임 시각 앞뒤의 센서 표본으로 값을 구한다.
- 프레임 시각보다 0.4초 넘게 오래된 센서만 있으면(센서 지연 · 권한 없음) 이번 프레임엔 넣지 않는다 → 원래 영상 추정 그대로.

계산하는 것 (폰 세로, 뒤 카메라가 앞을 봄)
- 기울기: 중력 방향 → 카메라가 수평보다 아래를 보는 각(pitch) · 옆으로 기운 각(roll). → 지평선 위치 = 화면 가운데 − 초점 × tan(pitch)
- 회전: 자이로를 직전 프레임부터 적분 → 고개 · 몸을 크게 돌리면 "카메라 흔들림"으로 알려 판단을 잠깐 쉬게 한다.
- 걸음: 가속도 크기의 출렁임으로 걸음 수 · 걷는 중 · 멈춤 → 보행 속도 = 걸음 빈도 × 보폭(키 × 0.415).
"""
from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

KEEP_MS = 6000
STALE_MS = 400


class ImuBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.q: deque = deque()            # (t, lin(3)|None, g(3)|None, rot(3)|None)
        self.prev_frame_t = None
        self.height_m = 1.70
        self.flip = None                   # 기기마다 가속도 부호가 반대일 수 있음(아이폰 사파리 등) → 처음에 정함

    def set_height(self, cm):
        try:
            self.height_m = min(2.2, max(0.9, float(cm) / 100))
        except (TypeError, ValueError):
            pass

    def add(self, rows):
        """sensors.js 의 motion 행: [t, ax,ay,az(중력 뺀 가속도), gx,gy,gz(중력 포함), ra,rb,rg(회전 °/s)]"""
        if not rows:
            return
        def v3(r, i):
            x = r[i:i + 3]
            return None if any(a is None for a in x) else np.array(x, float)
        with self.lock:
            for r in rows:
                if not r or len(r) < 10 or r[0] is None:
                    continue
                self.q.append((float(r[0]), v3(r, 1), v3(r, 4), v3(r, 7)))
            if self.q:
                newest = self.q[-1][0]
                while self.q and newest - self.q[0][0] > KEEP_MS:
                    self.q.popleft()

    def estimate(self, t_ms: float | None) -> dict | None:
        if t_ms is None:
            return None
        with self.lock:
            rows = list(self.q)
        if not rows or rows[-1][0] < t_ms - STALE_MS:
            return None
        out = dict(t_ms=t_ms)

        # ── 기울기 (프레임 시각 앞 0.15초 ~ 뒤 0.05초 평균) ──
        g = [r[2] for r in rows if r[2] is not None and t_ms - 150 <= r[0] <= t_ms + 50]
        if len(g) >= 3:
            gm = np.mean(g, axis=0)
            if self.flip is None and abs(gm[1]) > 6:
                self.flip = gm[1] < 0                   # 세로로 든 폰은 y 가 +9.8 이어야 정상
            if self.flip:
                gm = -gm
            n = float(np.linalg.norm(gm))
            pitch = math.atan2(gm[2], gm[1])          # +: 카메라가 수평보다 아래를 봄
            roll = math.atan2(gm[0], gm[1])
            out.update(pitch_rad=pitch, roll_rad=roll,
                       tilt_ok=bool(7.0 <= n <= 12.5 and abs(roll) < math.radians(30) and abs(pitch) < math.radians(70)))

        # ── 회전 (직전 프레임부터 지금 프레임까지) ──
        t0 = self.prev_frame_t if self.prev_frame_t is not None and 0 < t_ms - self.prev_frame_t < 1500 else t_ms - 150
        rot = [(r[0], r[3]) for r in rows if r[3] is not None and t0 < r[0] <= t_ms]
        if len(rot) >= 2:
            ang, peak = 0.0, 0.0
            for (ta, a), (tb, _) in zip(rot, rot[1:]):
                speed = float(np.linalg.norm(a))
                ang += speed * (tb - ta) / 1000
                peak = max(peak, speed)
            out.update(rot_deg=ang, rate_peak_dps=peak)
        self.prev_frame_t = t_ms

        # ── 걸음 (최근 3초) ──
        win = [(r[0], r[1] if r[1] is not None else None, r[2]) for r in rows if t_ms - 3000 <= r[0] <= t_ms + 50]
        if len(win) >= 60:
            ts = np.array([w[0] for w in win]) / 1000
            gs = [w[2] for w in win if w[2] is not None]
            up = np.mean(gs, axis=0) if gs else None
            up = up / np.linalg.norm(up) if up is not None and np.linalg.norm(up) > 1 else None
            # 걸음마다 위아래로 한 번 출렁이는 '중력 방향' 가속도만 본다 (크기를 쓰면 위 · 아래 두 번 세어진다)
            if up is not None and all(w[1] is not None for w in win):
                mag = np.array([w[1] for w in win]) @ up
                if self.flip:
                    mag = -mag
            elif all(w[2] is not None for w in win):
                gg = np.linalg.norm([w[2] for w in win], axis=1); mag = gg - np.mean(gg)
            else:
                mag = None
            if mag is not None:
                mag = np.convolve(mag, np.ones(5) / 5, mode="same")      # 잔떨림 줄이기
                sd = float(np.std(mag))
                thr = float(np.mean(mag) + 0.5 * sd)
                peaks, last = 0, -1e9
                for i in range(1, len(mag) - 1):
                    if mag[i] > thr and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1] and ts[i] - last >= 0.28:
                        peaks += 1; last = ts[i]
                span = float(ts[-1] - ts[0])
                cad = peaks / span if span > 1.5 else 0.0
                recent = mag[ts >= ts[-1] - 1.5]
                still = len(recent) > 20 and float(np.std(recent)) < 0.22
                walking = (not still) and sd > 0.45 and 1.0 <= cad <= 3.0
                out.update(step_hz=round(cad, 2), accel_sd=round(sd, 3), stationary=bool(still), walking=bool(walking),
                           walk_mps=round(min(2.2, max(0.3, cad * 0.415 * self.height_m)), 2) if walking else 0.0)
        return out


IMU = ImuBuffer()


def horizon_from(est: dict, h: int, focal: float) -> float | None:
    """중력으로 구한 지평선 높이(px). 믿을 수 없으면 None."""
    if not est or not est.get("tilt_ok"):
        return None
    y = h / 2 - focal * math.tan(est["pitch_rad"])
    return float(min(0.85 * h, max(-0.3 * h, y)))

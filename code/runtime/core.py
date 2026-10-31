"""실행 시스템 공통: 시계 · 데이터 형식 · 로그.

시각은 모두 '영상 시각(초)'이다. 녹화 영상을 실제 속도로 재생하므로 영상 시각 = 시작 후 경과한 벽시계 시간.
VLM · TTS 지연은 벽시계로 흐르므로 같은 축에서 바로 비교된다 (경고 지연 = 소리 시작 − 트리거 시각).
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "p0"):         # runtime/ · code/(common) · code/p0(config, frames, ko_template)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


class Clock:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def sleep_until(self, t: float):
        d = t - self.now()
        if d > 0:
            time.sleep(d)


@dataclass
class Det:
    track_id: int
    cls: str
    conf: float
    box: tuple                 # x1, y1, x2, y2 (프레임 픽셀)
    foot: tuple = (0.0, 0.0)   # 마스크 접지점
    edge: int = 0
    mask_area: float = 0.0


@dataclass
class Frame:
    idx: int                   # 원본 프레임 번호
    t: float                   # 영상 시각
    image: np.ndarray | None   # BGR (긴 변 1920 이하)
    dets: list
    W: int
    H: int


@dataclass
class Event:
    """트리거 출력. 어떤 트리거든 이 모양으로 낸다 (임태규 최종 코드도 여기에 맞추면 그대로 끼워짐)."""
    t: float
    cls: str
    scenario: str              # S1 이동체 · S2 장애물 · S3 신호 · S4 단차
    track_id: int | None = None
    box: tuple | None = None
    W: int = 0
    H: int = 0
    kind: str = "warn"
    extra: dict = field(default_factory=dict)   # 트리거별 부가 정보 (S3는 state="GO"/"STOP")

    def as_dict(self) -> dict:
        return dict(t=self.t, cls=self.cls, scenario=self.scenario, track_id=self.track_id, kind=self.kind,
                    box=None if self.box is None else [round(float(v), 1) for v in self.box], **self.extra)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Log:
    """모든 단계의 시각을 한 파일(jsonl)에. now = 기록한 순간의 영상 시각."""

    def __init__(self, path: Path, clock: Clock):
        self.f = open(path, "w", encoding="utf-8")
        self.clock = clock
        self.lock = threading.Lock()
        self.rows = []

    def __call__(self, stage: str, **kw):
        rec = {"now": round(self.clock.now(), 3), "stage": stage, **kw}
        with self.lock:
            self.rows.append(rec)
            self.f.write(json.dumps(rec, ensure_ascii=False, default=_default) + "\n")
            self.f.flush()
        return rec

    def close(self):
        self.f.close()

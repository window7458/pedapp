"""폰 센서 + 프레임 시각 기록. Sensor Logger와 같은 파일 이름 · 열 순서로 저장해 기존 분석 코드를 그대로 쓴다.

  sessions/<시작시각>/Gyroscope.csv · Accelerometer.csv · Gravity.csv · Orientation.csv · Compass.csv · Location.csv · Frames.csv · Metadata.csv
  time = ns (폰 시계), seconds_elapsed = 세션 시작 뒤 초. Frames.csv의 server_ms 와 Sensors의 시계 차이는 clock.csv 로 맞춘다.

웹 브라우저로는 걸음 수 · 활동 · 기압계를 읽을 수 없다 (네이티브 앱에서만 가능).
"""
from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path

GAP_NEW_SESSION_S = 120.0
HEADERS = {
    "Gyroscope": ["time", "seconds_elapsed", "z", "y", "x"],                  # rad/s
    "Accelerometer": ["time", "seconds_elapsed", "z", "y", "x"],              # m/s², 중력 제외
    "Gravity": ["time", "seconds_elapsed", "z", "y", "x"],                    # 중력 포함 - 중력 제외
    "Orientation": ["time", "seconds_elapsed", "yaw", "roll", "pitch"],       # rad
    "Compass": ["time", "seconds_elapsed", "magneticBearing"],                # 도
    "Location": ["time", "seconds_elapsed", "altitude", "speedAccuracy", "bearingAccuracy", "latitude", "altitudeAboveMeanSeaLevel",
                 "bearing", "horizontalAccuracy", "verticalAccuracy", "longitude", "speed"],
    "Frames": ["server_ms", "source", "capture_ms", "bytes"],
    "clock": ["server_ms", "client_ms"],
}
_num = lambda v: None if v is None else float(v)


class SensorLog:
    def __init__(self, root: Path, enabled: bool = True):
        self.root, self.enabled = Path(root), enabled
        self.lock = threading.Lock()
        self.dir: Path | None = None
        self.files, self.writers = {}, {}
        self.t0_ms, self.last_data, self.last_flush = None, 0.0, 0.0
        self.count = {k: 0 for k in HEADERS}
        self.rate, self._rate_prev, self._rate_t = {}, {}, time.time()
        self.latest = {}
        self.meta = None

    def _open(self):
        self.close()
        self.dir = self.root / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0_ms = None
        self.count = {k: 0 for k in HEADERS}
        self._rate_prev = {}
        if self.meta:
            self._write_meta()

    def _write_meta(self):
        (self.dir / "Metadata.csv").write_text("key,value\n" + "".join(f"{k},{str(v).replace(',', ' ')}\n" for k, v in self.meta.items()), encoding="utf-8")

    def _w(self, name):
        w = self.writers.get(name)
        if w is None:
            f = open(self.dir / f"{name}.csv", "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(HEADERS[name])
            self.files[name], self.writers[name] = f, w
        return w

    def _ready(self):
        now = time.time()
        if self.dir is None or now - self.last_data > GAP_NEW_SESSION_S:
            self._open()
        self.last_data = now

    def _row(self, name, t_ms, values):
        if self.t0_ms is None:
            self.t0_ms = t_ms
        self._w(name).writerow([int(t_ms * 1e6), round((t_ms - self.t0_ms) / 1000, 4), *values])
        self.count[name] += 1

    def add(self, m: dict):
        """폰이 보낸 묶음: {now, motion:[[t, ax,ay,az, gx,gy,gz, ra,rb,rg]], orient:[[t, a,b,g, heading]], gps:[[t, lat,lon,alt,acc,altAcc,speed,heading]]}"""
        if not self.enabled:
            return
        with self.lock:
            self._ready()
            if m.get("now"):
                self._w("clock").writerow([round(time.time() * 1000, 1), m["now"]])
                self.count["clock"] += 1
            if m.get("meta"):
                self.meta = m["meta"]
                self._write_meta()
            for r in m.get("motion") or []:
                t, ax, ay, az, gx, gy, gz, ra, rb, rg = r
                if ra is not None:      # deg/s → rad/s. rotationRate: alpha = z, beta = x, gamma = y
                    self._row("Gyroscope", t, [math.radians(ra), math.radians(rg or 0), math.radians(rb or 0)])
                    self.latest["gyro"] = [round(math.radians(rb or 0), 3), round(math.radians(rg or 0), 3), round(math.radians(ra), 3)]
                if ax is not None:
                    self._row("Accelerometer", t, [az, ay, ax])
                    if gx is not None:
                        self._row("Gravity", t, [gz - az, gy - ay, gx - ax])
            for t, a, b, g, heading in m.get("orient") or []:
                if b is not None:
                    self._row("Orientation", t, [math.radians(a or 0), math.radians(g or 0), math.radians(b)])
                    self.latest["tilt"] = round(b, 1)
                if heading is not None:
                    self._row("Compass", t, [heading])
                    self.latest["heading"] = round(heading, 1)
            for t, lat, lon, alt, acc, alt_acc, speed, hd in m.get("gps") or []:
                self._row("Location", t, [alt, "", "", lat, alt, hd, acc, alt_acc, lon, speed])
                self.latest["gps"] = dict(lat=round(lat, 6), lon=round(lon, 6), acc=acc, speed=speed)
            self._flush()

    def frame(self, source: str, capture_ms, n: int):
        if not self.enabled:
            return
        with self.lock:
            self._ready()
            self._w("Frames").writerow([round(time.time() * 1000, 1), source, capture_ms if capture_ms is not None else "", n])
            self.count["Frames"] += 1
            self._flush()

    def _flush(self):
        now = time.time()
        if now - self.last_flush > 2.0:
            for f in self.files.values():
                f.flush()
            self.last_flush = now

    def summary(self) -> dict:
        with self.lock:
            now = time.time()
            dt = now - self._rate_t
            if dt >= 2.0:
                self.rate = {k: round((self.count[k] - self._rate_prev.get(k, 0)) / dt, 1) for k in ("Gyroscope", "Accelerometer", "Orientation", "Location", "Frames")}
                self._rate_prev, self._rate_t = dict(self.count), now
            for f in self.files.values():
                f.flush()
            return dict(session=self.dir.name if self.dir else None, hz=self.rate, latest=self.latest,
                        total=sum(v for k, v in self.count.items() if k != "clock"))

    def close(self):
        for f in self.files.values():
            try:
                f.close()
            except OSError:
                pass
        self.files, self.writers = {}, {}

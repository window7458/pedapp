#!/usr/bin/env python3
"""라즈베리파이 카메라 → 서버 전송 + 끝에서 끝까지 지연 측정.

Pi Zero 2 W 는 판단을 돌리지 않는다. 하드웨어 JPEG 로 압축해서 서버로 보내기만 한다.
서버는 처리 뒤 frame_done 을 돌려주고, 여기서 '찍은 순간 → 판단 결과 도착' 시간을 같은 시계로 잰다 (시계 맞춤 불필요).

  파이:  python3 pi_camera.py wss://주소/ws --key 비밀키 --size 960x540 --fps 15 --rotate 90
  시험:  python pi_camera.py ws://localhost:8000/ws --video test_3.mp4 --fps 10 --seconds 30     (파이 없이 영상 파일로)

결과: 5초마다 화면 출력, 끝나면 realtime_summary.json, 프레임별 realtime_log.csv
"""
import argparse
import csv
import io
import json
import ssl
import struct
import threading
import time

import websocket  # pip install websocket-client

MAGIC = b"SEF1"
P50_OK_MS, P95_OK_MS, FPS_OK = 300, 500, 10      # 실시간 판정 기준 (걸어갈 때 0.5 s ≈ 0.7 m)


def pct(v, q):
    return sorted(v)[min(len(v) - 1, int(len(v) * q))] if v else None


class Sender:
    def __init__(self, url, insecure):
        self.url, self.insecure = url, insecure
        self.latest, self.ev = None, threading.Event()
        self.ws, self.stop = None, threading.Event()
        self.sent = self.dropped = self.bytes = 0
        self.pending = {}
        self.rtt, self.proc, self.up = [], [], []
        self.log = open("realtime_log.csv", "w", newline="")
        self.w = csv.writer(self.log)
        self.w.writerow(["capture_ms", "total_ms", "upload_ms", "server_ms", "bytes"])
        self.t0 = time.time()
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def offer(self, jpg, cap_ms):
        if self.latest is not None:
            self.dropped += 1
        self.latest = (jpg, cap_ms)         # 항상 최신 한 장만. 밀리면 옛 프레임을 버린다
        self.ev.set()

    def _connect(self):
        while not self.stop.is_set():
            try:
                opts = {"sslopt": {"cert_reqs": ssl.CERT_NONE}} if self.insecure else {}
                self.ws = websocket.create_connection(self.url, timeout=10, **opts)
                self.ws.settimeout(None)
                print("서버 연결됨:", self.url.split("?")[0], flush=True)
                return
            except Exception as e:
                print("연결 재시도:", e, flush=True)
                self.stop.wait(2)

    def _send_loop(self):
        self._connect()
        while not self.stop.is_set():
            if not self.ev.wait(0.5):
                continue
            self.ev.clear()
            item, self.latest = self.latest, None
            if item is None:
                continue
            jpg, cap = item
            try:
                t = time.time() * 1000
                self.ws.send_binary(MAGIC + struct.pack("!d", cap) + jpg)
                self.pending[cap] = (time.time() * 1000 - t, len(jpg))
                self.sent += 1
                self.bytes += len(jpg)
            except Exception as e:
                print("전송 끊김:", e, flush=True)
                self._connect()

    def _recv_loop(self):
        while not self.stop.is_set():
            ws = self.ws
            if ws is None:
                time.sleep(0.1)
                continue
            try:
                m = json.loads(ws.recv())
            except Exception:
                time.sleep(0.2)
                continue
            if m.get("type") != "frame_done":
                continue
            cap = m["c"]
            total = time.time() * 1000 - cap
            up, n = self.pending.pop(cap, (None, 0))
            self.rtt.append(total)
            self.proc.append(m.get("p_ms", 0))
            self.w.writerow([cap, round(total, 1), None if up is None else round(up, 1), m.get("p_ms"), n])
            for k in [k for k in self.pending if k < cap]:        # 서버가 건너뛴 프레임
                self.pending.pop(k, None)

    def report(self, last=False):
        n = 60
        r = self.rtt[-n:]
        dur = max(time.time() - self.t0, 1e-3)
        fps = len(self.rtt) / dur
        s = dict(done=len(self.rtt), sent=self.sent, dropped_on_device=self.dropped, fps_done=round(fps, 1), total_p50_ms=pct(r, .5), total_p95_ms=pct(r, .95),
                 server_p50_ms=pct(self.proc[-n:], .5), mbps=round(self.bytes * 8 / dur / 1e6, 2))
        s = {k: round(v) if isinstance(v, float) and k.endswith("ms") else v for k, v in s.items()}
        print(s, flush=True)
        if last:
            rr = self.rtt
            s.update(total_p50_ms=pct(rr, .5), total_p95_ms=pct(rr, .95), seconds=round(dur, 1))
            ok = bool(rr) and pct(rr, .5) <= P50_OK_MS and pct(rr, .95) <= P95_OK_MS and fps >= FPS_OK
            s["realtime_ok"] = ok
            s["criteria"] = f"p50<={P50_OK_MS}ms, p95<={P95_OK_MS}ms, >={FPS_OK}fps"
            json.dump(s, open("realtime_summary.json", "w"), ensure_ascii=False, indent=1)
            print("실시간 판정:", "통과" if ok else "미달", s, flush=True)
            self.log.close()


def from_video(a, snd):
    import cv2
    cap = cv2.VideoCapture(a.video)
    W, H = map(int, a.size.split("x"))
    gap, next_t = 1.0 / a.fps, time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        for _ in range(max(0, round(src_fps / a.fps) - 1)):
            cap.grab()
        h, w = f.shape[:2]
        s = max(W, H) / max(h, w)
        f = cv2.resize(f, (round(w * s), round(h * s)))
        _, b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        snd.offer(b.tobytes(), time.time() * 1000)
        next_t += gap
        time.sleep(max(0, next_t - time.time()))
        yield


class _Out(io.BufferedIOBase):
    def __init__(self, snd):
        self.snd = snd

    def write(self, buf):
        self.snd.offer(bytes(buf), time.time() * 1000)
        return len(buf)


def from_camera(a, snd):
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder, Quality
    from picamera2.outputs import FileOutput
    W, H = map(int, a.size.split("x"))
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (W, H), "format": "YUV420"}, buffer_count=4,
                                                 controls={"FrameRate": a.fps}))
    q = {"low": Quality.LOW, "medium": Quality.MEDIUM, "high": Quality.HIGH}[a.pi_quality]
    cam.start_recording(MJPEGEncoder(), FileOutput(_Out(snd)), quality=q)
    print(f"카메라 시작 {W}x{H} {a.fps}fps (하드웨어 JPEG)", flush=True)
    while True:
        time.sleep(1)
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="wss://주소/ws 또는 ws://IP:8000/ws")
    ap.add_argument("--key", default=None, help="서버 접속 키")
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--quality", type=int, default=70, help="--video 시험용 JPEG 품질")
    ap.add_argument("--pi-quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="카메라를 눕혀 달았으면 서버에서 돌림")
    ap.add_argument("--insecure", action="store_true", help="자체 서명 인증서 허용")
    ap.add_argument("--video", default=None, help="파이 없이 영상 파일로 시험")
    ap.add_argument("--seconds", type=float, default=0, help="시험 시간, 0이면 계속")
    a = ap.parse_args()
    sep = "&" if "?" in a.url else "?"
    url = f"{a.url}{sep}role=camera&rotate={a.rotate}&pin={a.pin}" + (f"&k={a.key}" if a.key else "")
    snd = Sender(url, a.insecure)
    gen = from_video(a, snd) if a.video else from_camera(a, snd)
    t0, last = time.time(), time.time()
    try:
        for _ in gen:
            if time.time() - last >= 5:
                snd.report()
                last = time.time()
            if a.seconds and time.time() - t0 >= a.seconds:
                break
    except KeyboardInterrupt:
        pass
    snd.stop.set()
    time.sleep(0.5)
    snd.report(last=True)


if __name__ == "__main__":
    main()

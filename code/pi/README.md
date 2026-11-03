# 라즈베리파이 Zero 2 W 카메라

파이는 **찍어서 보내기만** 해요. 판단은 GPU 서버, 음성 · 화면 · GPS · 자이로는 폰이 맡아요.

```
파이 카메라 ──JPEG(Wi-Fi)──▶ 서버 (YOLO + v5_29) ──경고──▶ 폰 (음성 · 진동)
                                   ▲
                 폰 센서(GPS · 자이로 · 방향) ──┘  sessions/ 에 CSV 로 저장
```

## 준비물
- Raspberry Pi Zero 2 W, 카메라 모듈, **22핀 미니 CSI 케이블**(Zero 전용, 일반 Pi 케이블은 안 맞아요)
- microSD, 보조배터리, 방열판 권장

## 설치 (한 번만)
1. Raspberry Pi Imager로 **Raspberry Pi OS Lite (64-bit)** 를 굽고, 설정에서 SSH와 Wi-Fi(폰 핫스팟)를 넣어요.
2. **아이폰 핫스팟은 설정 → 개인용 핫스팟 → "호환성 최대화"를 켜요.** Zero 2 W는 2.4 GHz만 돼요.
3. SSH로 들어가서:
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-websocket --no-install-recommends
rpicam-hello --list-cameras     # 카메라가 보이면 정상
```
4. `pi_camera.py`를 파이로 복사해요.

## 실행
서버(Colab) 노트북이 알려 주는 명령을 그대로 넣어요.
```bash
python3 pi_camera.py wss://<주소>/ws --key <키> --size 960x540 --fps 15
```
카메라를 눕혀 달았으면 `--rotate 90`(또는 270)을 붙여요. 판단 코드는 세로 영상 기준이에요.

## 실시간 판정
5초마다 지연이 찍히고, Ctrl+C로 끝내면 `realtime_summary.json`이 저장돼요.

- `total` = 찍은 순간 → 서버 판단 결과가 파이로 돌아온 시간 (왕복 포함)
- 통과 기준: 중앙값 300 ms 이하, 95% 500 ms 이하, 초당 10프레임 이상 (걸음 1.4 m/s면 0.5 s에 0.7 m 이동)

느리면 순서대로 낮춰 보세요: `--fps 10` → `--size 640x360` → `--pi-quality low`.

## 켜면 자동 실행 (선택)
```ini
# /etc/systemd/system/sayeye-cam.service
[Unit]
After=network-online.target
[Service]
User=pi
WorkingDirectory=/home/pi
ExecStart=/usr/bin/python3 /home/pi/pi_camera.py wss://<주소>/ws --key <키>
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now sayeye-cam`. Colab 주소는 켤 때마다 바뀌어요.

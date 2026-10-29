# XIAO ESP32S3 Sense를 서버 카메라로 쓰기

목표: XIAO가 와이파이로 영상을 내보내고(`http://<XIAO주소>:81/stream`), 노트북/서버의 Insight 앱이 그 주소를 읽어 같은 판정(v5_29)을 돌린다.

> 아래 순서는 Seeed·Espressif 공식 예제의 일반적인 흐름이에요. 메뉴 이름은 Arduino IDE · 보드 패키지 버전마다 조금 다를 수 있고, **이 문서는 실제 보드로 검증하지 않았어요.** 막히면 Seeed Studio의 "XIAO ESP32S3 Sense Camera" 위키를 같이 보세요.

## 1. 준비
- 보드: XIAO ESP32S3 **Sense** (카메라 확장 보드가 붙은 것), USB-C 데이터 케이블
- Arduino IDE 2.x
- 와이파이: **2.4 GHz** (ESP32는 5 GHz를 못 씁니다). 노트북/서버와 **같은 공유기**에 있어야 해요.
  - 폰 핫스팟을 쓰면 "호환성 극대화(2.4GHz)" 같은 옵션이 필요할 수 있어요 (기종마다 이름이 달라 확인 필요).

## 2. 펌웨어 올리기
1. Arduino IDE → 설정 → "추가 보드 관리자 URL"에 Espressif 패키지 주소(`https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`)를 넣고, 보드 관리자에서 **esp32 (Espressif Systems)** 설치.
2. 도구 → 보드 → **XIAO_ESP32S3** 선택.
3. 도구 → **PSRAM → "OPI PSRAM"** 으로 켜기 (카메라에 필요).
4. 파일 → 예제 → ESP32 → Camera → **CameraWebServer** 열기.
5. 코드 위쪽에서 `#define CAMERA_MODEL_XIAO_ESP32S3` 한 줄만 주석을 풀고 나머지 모델은 주석 처리.
6. `const char *ssid`, `const char *password`를 내 와이파이로 바꾸기.
7. 업로드. 끝나면 시리얼 모니터(115200)에 `Camera Ready! Use 'http://192.168.x.x' to connect` 같은 줄이 나와요 → 그 주소가 XIAO 주소.

## 3. 영상 주소 확인
- 브라우저에서 `http://<XIAO주소>` 를 열면 설정 페이지가 나오고, 스트림은 `http://<XIAO주소>:81/stream` 입니다.
- 이 주소를 주소창에 직접 넣어 영상이 나오면 성공.
- 해상도는 설정 페이지에서 **VGA(640x480) 이하**를 권장해요. 크면 와이파이에서 끊겨요.

## 4. 앱에 연결
```
python server.py --https --source http://<XIAO주소>:81/stream --perception yolo --weights <final_10class_best.pt> --device 0 --trigger hitbox
```
또는 앱 화면에서 "서버 카메라"를 고르고 주소를 입력.
서버(노트북)가 읽은 화면에 박스·경고를 덧입힌 영상은 `https://서버:8443/preview.mjpg` 로 폰에서도 볼 수 있어요.

## 알아 둘 점
- v5_29는 **카메라 높이 1.70 m, 세로 영상**을 가정하고 만들어졌어요. XIAO를 몸에 달 거면 실제 높이를 `--cam-h 1.3` 처럼 넣고, 센서가 가로 방향이면 판정(특히 바닥 분할의 위치 가정)이 원본과 달라질 수 있어요. 실제 장착 방향으로 한 번 시험해야 합니다.
- XIAO 영상은 화질·화각이 폰보다 낮아서 먼 신호등/작은 물체는 덜 잡혀요.
- 영상이 끊기면 앱이 자동으로 다시 연결을 시도합니다 (화면 아래 상태 줄에 표시).
- 다른 카메라: `rtsp://…`(IP 카메라, 앱이 TCP로 읽음), `udp://@:8554`(GoPro 계열 스트림), 숫자 `0`(USB 웹캠 번호). Tapo 등 제품별 RTSP 주소 형식과 GoPro 스트리밍 방법은 **제품 문서로 확인**이 필요해요 (여기서는 검증 안 함).

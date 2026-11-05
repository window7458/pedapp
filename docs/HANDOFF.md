# SayEye 작업 인계 (2026-11-05 기준, 브랜치 wip/phone-only-handoff)

저장소가 공개라서 서버 주소 · 키 · 사람이 찍힌 사진은 이 문서와 저장소에 넣지 않는다.

## 결정된 것
- 스마트폰만 쓴다. 라즈베리파이 · 서버 카메라(URL) 코드는 지웠다.
- **상시 서버 = Google Cloud VM `sayeye`** (서울, e2-standard-4, 무료 체험 크레딧, 12월 중순까지). `deploy/sayeye.sh install` 로 설치 완료, Tailscale Funnel 고정 주소. 안 쓸 땐 콘솔에서 중지, 쓸 땐 시작 (자동으로 다 켜짐). **GPU = 몬드리안(Runyour AI) A5000** 을 필요할 때 켜고 `bash deploy/gpu.sh start` (처음 한 번 `gpu.sh setup --hub …`). 오라클 · Hugging Face(무료 CPU도 PRO 필요) · Google Cloud L4(서울 리전 없음)는 포기.
- Tailscale 은 가입 완료 (구글 로그인).
- 트리거(`hitbox_*`, `runtime/triggers.py`)는 아직 손대지 않는다.

## 11-05 에 한 것
- 가짜 · 시험 코드 제거: `code/app/tests/`, mock VLM · mock 장면, 가짜 인식기 주입, 오프라인 데모 · 평가 코드(`runtime/run_demo.py` 등, `p0/` 평가 스크립트), Pi 코드, 카메라 진단 페이지. 저장소에 올라가 있던 `.pyc` 도 뺐다.
- VLM 자동 연결: 서버를 먼저 켜도 vLLM 이 나중에 올라오면 자동으로 설명 · 특별 장면이 켜짐 (15초마다 확인). 꺼져 있을 땐 부르지 않음.
- 접속 키: 키 없이 열면 키 입력 화면, 쿠키 7일 → 400일 (홈 화면 앱이 1주 뒤 막히던 문제).
- 키만 있으면 보이던 `/preview.mjpg` 제거, `/api/sensors` 는 보호자 확인번호 필요.
- 서버 설정을 `SAYEYE_*` 환경변수로도 받음 (상시 서버용), `--alerts-dir` 추가, 푸시 연락처 설정(`SAYEYE_PUSH_SUBJECT`).
- 상시 서버 / GPU 분리: `engine.py`(상시 서버 쪽 연결 · CPU 인식), `worker.py`(GPU 작업 서버), `SAYEYE_ENGINE=hub`. 경고 사진은 GPU 가 알려 준 프레임 번호로 상시 서버가 찍는다. 보행자 앱이 인식이 완전히 꺼졌을 때만 음성으로 알림 (CPU 인식일 때는 따로 말하지 않음).
- `deploy/sayeye.sh`(상시 서버 설치 · 관리), `deploy/gpu.sh`(GPU 설치 · 켜기 · 끄기). 이 환경에서 둘을 같이 띄워 CPU→GPU→CPU 전환까지 확인.
- 이 환경(클라우드 컨테이너, GPU 없음)에서 공식 yolo11n-seg 로 서버를 실제로 띄워 폰 흉내(웹소켓 영상) · 보호자 · 키 · 확인번호 · VLM 늦게 켜짐 · 감시 루프 재시작을 확인했다. 팀 가중치 · GPU · 실제 폰 · Tailscale 은 이 환경에서 확인하지 못했다.

## 남은 일 (순서대로)
1. **A5000 에서 `bash deploy/gpu.sh setup --hub <상시 서버 주소> --key … --token … --weights-gdrive …`** (값은 상시 서버 `bash deploy/sayeye.sh info`). 반납해도 남는 저장소 폴더에 clone.
2. **사람을 장애물로 말하는 문제** (트리거 쪽이라 보류 중). `alerts/evidence` 가 쌓이면 확인. 유력 경로 `hitbox_core.py:694`, `:977`.
3. 실제 폰에서 음성 명령 속도 확인.

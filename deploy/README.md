# SayEye 상시 서버 (A5000 GPU 서버)

한 번 설치하면 **터미널을 닫아도, 서버를 재부팅해도** 계속 돌아가고, 폰은 어디서든(LTE 포함) 고정 주소로 접속해요.

```
폰 (어디서든) ──HTTPS──▶ https://sayeye.<tailnet>.ts.net  (Tailscale Funnel, 무료 · 도메인 불필요)
                                   │
                         GPU 서버 127.0.0.1:8080  SayEye (YOLO + 히트박스 + 알림)   ← 자동 시작 · 죽으면 재시작
                                   │
                                   └─ 127.0.0.1:8000  vLLM Qwen2.5-VL (설명 · 특별 장면)   ← 자동 시작 · 늦게 켜져도 자동 연결
```

## 준비물 (처음 한 번)
1. **Tailscale 계정** — https://login.tailscale.com 에서 구글 계정으로 가입 (무료).
2. **팀 가중치** `final_10class_best.pt` — 서버에 올려 두거나, 구글 드라이브 파일 ID.

## 설치
GPU 서버에 접속해서:
```bash
git clone https://github.com/window7458/pedapp.git && cd pedapp
bash deploy/install.sh --weights ~/final_10class_best.pt
#   또는 드라이브에서 바로: bash deploy/install.sh --weights-gdrive <파일 ID>   ('링크가 있는 모든 사용자' 공유일 때)
```
중간에 두 번 브라우저를 열 일이 있어요.
- **Tailscale 로그인 링크** → 열어서 로그인.
- 처음이면 **Funnel 허용 링크**가 나와요 → 열어서 허용한 뒤 `bash deploy/install.sh` 를 한 번 더 실행.

끝나면 이렇게 나와요:
```
보행자:  https://sayeye.xxxx.ts.net/?k=접속키
보호자:  https://sayeye.xxxx.ts.net/guardian?k=접속키   확인번호 123456
```
- 보행자 폰에서 주소를 열고 **홈 화면에 추가** (아이폰은 사파리). 키는 기기에 기억돼서 다음부터는 아이콘만 누르면 돼요.
- 키 없이 열면 키 입력 화면이 나와요. 접속 키 · 확인번호는 `deploy/local/sayeye.env` 에 있고 저장소에는 올라가지 않아요.

## 관리
```bash
bash deploy/sayeye.sh info      # 주소 · 키 · 확인번호 · 상태
bash deploy/sayeye.sh status    # 서비스 상태
bash deploy/sayeye.sh logs      # SayEye 로그 (logs vllm: VLM 로그)
bash deploy/sayeye.sh update    # 최신 코드 받고 다시 시작
bash deploy/sayeye.sh restart   # deploy/local/sayeye.env 를 고친 뒤
```

## 알아 둘 것
- **vLLM 첫 실행**은 모델(약 16GB)을 받느라 10분 이상 걸려요. 그동안은 경고만 나가고, VLM이 켜지면 설명 · 특별 장면이 저절로 붙어요.
- GPU 메모리: vLLM 80%, 나머지를 YOLO · 바닥 분할이 써요 (`VLLM_GPU_UTIL`). 메모리 부족이 나면 0.75로 낮추세요.
- **컨테이너형 GPU 서버(Runyour AI 등)**: systemd 가 없어서 감시 스크립트 + `@reboot` 로 돌려요. 컨테이너 자체를 반납 · 정지하면 같이 멈추니, 컨테이너는 켜 둬야 해요.
- 아이폰 푸시 알림을 쓰려면 `SAYEYE_PUSH_SUBJECT=mailto:본인메일` 로 바꿔 주세요.
- 위험 순간 사진 · 영상(`code/app/alerts/`), 센서 기록(`code/app/sessions/`)은 서버에만 남아요. 사람이 찍혀 있으니 공유하지 마세요.

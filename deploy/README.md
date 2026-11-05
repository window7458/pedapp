# SayEye 서버 설치

**상시 서버**(Google Cloud VM)와 **GPU 서버**(A5000, 필요할 때만)를 나눠서 써요.

```
폰 (어디서든) ──HTTPS──▶ 상시 서버: Google Cloud VM   https://sayeye.<tailnet>.ts.net
                          · 화면 · 접속 키 · 보호자 알림 · 위치 · 푸시 · 기록      ← 24시간, 재부팅해도 자동 시작
                          · GPU 가 없을 땐 이 서버 CPU 로 인식 (YOLO + 히트박스, VLM 설명 없이)
                                  ▲
                                  │ GPU 쪽에서 먼저 연결 (GPU 에는 공개 주소 · 테일스케일 필요 없음)
                          GPU 서버: A5000 (켤 때만)
                          · 팀 YOLO + v5_29 히트박스 + VLM 설명                  ← 켜면 몇십 초 안에 자동 연결
```

시연 · 최종 확인은 GPU 를 켜고 하고, 평소 동작 확인은 GPU 없이도 돼요.

---

## 1. 상시 서버: Google Cloud VM (서울)
1. https://console.cloud.google.com 가입 (무료 체험 300달러 크레딧).
2. Compute Engine → VM 인스턴스 → 인스턴스 만들기: 리전 **asia-northeast3(서울)**, 머신 **e2-standard-4**, 부팅 디스크 **Ubuntu 22.04 · 30GB**.
3. 목록의 **SSH** 버튼으로 접속해서:
   ```bash
   sudo apt-get update && sudo apt-get install -y git
   git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
   bash deploy/sayeye.sh install --weights-gdrive <가중치 파일 ID>
   ```
   중간의 Tailscale 로그인 링크는 **Ctrl+클릭**으로 열어요 (터미널에서 Ctrl+C 는 '중지'라 누르면 안 돼요).
4. 끝나면 보행자 · 보호자 주소와 GPU 연결 명령이 나와요. 안 쓸 땐 콘솔에서 VM **중지**, 쓸 땐 **시작** (자동으로 다 켜지고 주소도 그대로).

> 가중치 파일 ID: 구글 드라이브 `final_10class_best.pt` → 공유 → "링크가 있는 모든 사용자" → 링크의 `/d/` 와 `/view` 사이.

## 3. GPU 를 쓸 때 (A5000 에서)
처음 한 번 (상시 서버 `bash deploy/sayeye.sh info` 에 나온 명령 그대로). 반납해도 남는 저장소 폴더에서 clone 하세요:
```bash
git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
bash deploy/gpu.sh setup --hub https://sayeye.<tailnet>.ts.net --key <접속키> --token <작업서버토큰> --weights-gdrive <파일 ID>
```
다음부터 GPU 를 켤 때마다:
```bash
cd pedapp && bash deploy/gpu.sh start
```
다 쓰면 `bash deploy/gpu.sh stop` 하고 GPU 를 반납하세요 (안 해도 상시 서버가 알아서 CPU 인식으로 바꿔요).

- VLM(상황 설명)은 처음 켤 때 모델(약 16GB)을 받느라 10분 이상 걸려요. 그동안에도 위험 경고는 나가요.
- 저장 공간이 유지되는 GPU 대여라면 두 번째부터는 설치 없이 바로 켜져요.

## 관리 명령
| 어디서 | 명령 | 하는 일 |
|---|---|---|
| 상시 서버 | `bash deploy/sayeye.sh info` | 주소 · 키 · 확인번호 · 지금 인식 상태 · GPU 연결 명령 |
| 상시 서버 | `bash deploy/sayeye.sh logs` / `status` | 로그 / 상태 |
| 상시 서버 | `bash deploy/sayeye.sh update` | 최신 코드 받고 재시작 |
| 상시 서버 | `bash deploy/sayeye.sh restart` | `deploy/local/sayeye.env` 고친 뒤 |
| GPU | `bash deploy/gpu.sh start` / `stop` | 켜기 / 끄기 |
| GPU | `bash deploy/gpu.sh logs` (`logs vllm`) | 작업 서버 / VLM 로그 |

## 알아 둘 것
- 접속 키 · 확인번호 · 작업 서버 토큰은 `deploy/local/` 에만 있고 저장소에는 올라가지 않아요.
- 아이폰 푸시 알림을 쓰려면 `deploy/local/sayeye.env` 의 `SAYEYE_PUSH_SUBJECT=mailto:본인메일` 로 바꾸고 `restart`.
- GPU 가 꺼져 있을 때 인식을 아예 쉬게 하려면 `SAYEYE_FALLBACK=none`.
- 위험 순간 사진 · 영상(`code/app/alerts/`), 센서 기록(`code/app/sessions/`)은 상시 서버에만 남아요. 사람이 찍혀 있으니 공유하지 마세요.

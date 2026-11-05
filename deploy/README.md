# SayEye 서버 설치

**상시 서버**(Hugging Face Spaces, 무료)와 **GPU 서버**(필요할 때만)를 나눠서 써요.

```
폰 (어디서든) ──HTTPS──▶ 상시 서버: Hugging Face Space   https://<아이디>-sayeye.hf.space
                          · 화면 · 접속 키 · 보호자 알림 · 위치 · 푸시 · 기록      ← 24시간, 재부팅해도 자동 시작
                          · GPU 가 없을 땐 이 서버 CPU 로 인식 (YOLO + 히트박스, VLM 설명 없이)
                                  ▲
                                  │ GPU 쪽에서 먼저 연결 (GPU 에는 공개 주소 · 테일스케일 필요 없음)
                          GPU 서버: A5000 (켤 때만)
                          · 팀 YOLO + v5_29 히트박스 + VLM 설명                  ← 켜면 몇십 초 안에 자동 연결
```

시연 · 최종 확인은 GPU 를 켜고 하고, 평소 동작 확인은 GPU 없이도 돼요.

---

## 1. 상시 서버: Hugging Face Spaces (무료 · 카드 불필요 · 고정 HTTPS 주소)
1. https://huggingface.co/join 에서 가입 (이메일 인증).
2. https://huggingface.co/new-space
   - Space name: `sayeye` · SDK: **Docker** → **Blank** · Hardware: **CPU basic (Free)** · **Public**
     (공개여도 접속 키가 없으면 키 입력 화면만 보여요.)
3. 만든 Space → **Settings → Variables and secrets → New secret** 으로 넣기:
   | 이름 | 값 |
   |---|---|
   | `SAYEYE_KEY` | 접속 키 (아무 영문·숫자 10자 이상) |
   | `SAYEYE_GUARDIAN_PIN` | 보호자 확인번호 6자리 |
   | `SAYEYE_WORKER_TOKEN` | GPU 연결용 토큰 (아무 영문·숫자 20자 이상) |
   | `WEIGHTS_GDRIVE_ID` | (선택) 가중치 드라이브 파일 ID — 넣으면 GPU 없을 때 CPU 로 인식 |
   | `SAYEYE_PUSH_SUBJECT` | (선택) `mailto:본인메일` — 아이폰 푸시 알림용 |
4. Space → **Files → + Add file → Create a new file** → 이름 `Dockerfile`, 내용은 이 저장소의
   [`deploy/huggingface/Dockerfile`](huggingface/Dockerfile) 을 그대로 복사 → **Commit**.
5. 몇 분 빌드되면 주소가 생겨요: `https://<아이디>-sayeye.hf.space`
   - 보행자: `https://<아이디>-sayeye.hf.space/?k=<SAYEYE_KEY>` → 폰에서 **홈 화면에 추가**
   - 보호자: `https://<아이디>-sayeye.hf.space/guardian?k=<SAYEYE_KEY>` + 확인번호

> 가중치 파일 ID: 구글 드라이브 `final_10class_best.pt` → 공유 → "링크가 있는 모든 사용자" → 링크의 `/d/` 와 `/view` 사이.

알아 둘 것
- 48시간 동안 아무도 안 열면 잠들어요. 누가 주소를 열면 1~2분 뒤 다시 켜져요.
- 켜질 때마다 깃허브 최신 코드를 받아요. 코드를 고친 뒤에는 Space → **Settings → Restart** 만 하면 돼요.
- 무료 Space 는 저장 공간이 유지되지 않아서, 다시 켜지면 보호자 알림 기록 · 사진이 지워져요.

### (다른 방법) 리눅스 서버 · 집 PC 를 상시 서버로
`bash deploy/sayeye.sh install` — Tailscale Funnel 로 `https://sayeye.<tailnet>.ts.net` 주소를 만들어요 (그 컴퓨터를 계속 켜 둬야 해요).

## 3. GPU 를 쓸 때 (A5000 에서)
처음 한 번 (주소 · 키 · 토큰은 1단계에서 정한 값):
```bash
git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
bash deploy/gpu.sh setup --hub https://<아이디>-sayeye.hf.space --key <SAYEYE_KEY> --token <SAYEYE_WORKER_TOKEN> --weights-gdrive <파일 ID>
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

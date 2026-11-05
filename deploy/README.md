# SayEye 서버 설치

**상시 서버**(무료, 24시간)와 **GPU 서버**(필요할 때만)를 나눠서 써요.

```
폰 (어디서든) ──HTTPS──▶ 상시 서버: 오라클 무료 서버   https://sayeye.<tailnet>.ts.net
                          · 화면 · 접속 키 · 보호자 알림 · 위치 · 푸시 · 기록      ← 24시간, 재부팅해도 자동 시작
                          · GPU 가 없을 땐 이 서버 CPU 로 인식 (YOLO + 히트박스, VLM 설명 없이)
                                  ▲
                                  │ GPU 쪽에서 먼저 연결 (GPU 에는 공개 주소 · 테일스케일 필요 없음)
                          GPU 서버: A5000 (켤 때만)
                          · 팀 YOLO + v5_29 히트박스 + VLM 설명                  ← 켜면 몇십 초 안에 자동 연결
```

시연 · 최종 확인은 GPU 를 켜고 하고, 평소 동작 확인은 GPU 없이도 돼요.

---

## 1. 오라클 무료 서버 만들기 (처음 한 번, 약 15분)
1. https://www.oracle.com/kr/cloud/free/ → **무료로 시작하기**.
   - 카드 등록은 본인 확인용이에요. 무료(Always Free) 자원만 쓰면 돈이 나가지 않아요.
   - **홈 리전은 `South Korea Central (Seoul)` 또는 `South Korea North (Chuncheon)`** 으로 고르세요. 나중에 못 바꾸고, 폰과 가까워야 빨라요.
2. 로그인 → **인스턴스 생성 (Create a VM instance)**
   - 이름: `sayeye`
   - 이미지: **Ubuntu 22.04** (또는 24.04)
   - Shape: **Ampere → VM.Standard.A1.Flex, OCPU 4 · 메모리 24GB** ("Always Free 대상" 표시 확인)
   - SSH 키: **개인 키 저장(Save private key)** 을 눌러 파일을 받아 두세요.
   - "용량 부족(Out of capacity)"이 나오면 가용성 도메인을 바꾸거나 OCPU 2 · 12GB 로 해 보세요.
3. 만들어지면 나오는 **공용 IP** 로 접속:
   ```bash
   ssh -i 받은키.key ubuntu@공용IP
   ```
   (윈도우는 PowerShell 에서 같은 명령이 돼요.)

## 2. 상시 서버 설치 (오라클 서버에서, 처음 한 번)
```bash
git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
bash deploy/sayeye.sh install --weights-gdrive <가중치 파일 ID>     # 가중치는 GPU 가 꺼져 있을 때 CPU 인식용 (없어도 됨)
```
- 중간에 **Tailscale 로그인 링크**가 나오면 열어서 가입한 구글 계정으로 로그인하세요.
- **Funnel 허용 링크**가 나오면 열어서 허용하고 `bash deploy/sayeye.sh install` 을 한 번 더 실행하세요.
- 끝나면 보행자 · 보호자 주소와 **GPU 연결 명령**이 나와요. 다시 보려면 `bash deploy/sayeye.sh info`.

> 가중치 파일 ID: 구글 드라이브에서 `final_10class_best.pt` → 공유 → "링크가 있는 모든 사용자" → 링크의 `/d/` 와 `/view` 사이 문자열.

이제 폰에서 보행자 주소를 열고 **홈 화면에 추가**하세요 (아이폰은 사파리). 키는 기기에 기억돼요.

## 3. GPU 를 쓸 때 (A5000 에서)
처음 한 번 — 상시 서버의 `info` 에 나온 명령을 그대로 붙여 넣으면 돼요:
```bash
git clone https://github.com/window7458/pedapp.git && cd pedapp && git checkout wip/phone-only-handoff
bash deploy/gpu.sh setup --hub https://sayeye.xxxx.ts.net --key 접속키 --token 토큰 --weights-gdrive <파일 ID>
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

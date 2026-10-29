# Insight 앱 (웹앱 · 서버는 노트북/GPU, 폰은 브라우저)

시각장애인 보행 보조: **폰 카메라(또는 서버에 연결한 카메라)의 실시간 영상**을 서버가 받아 판정하고, 경고를 **영상이 찍히는 동안 바로** 폰 화면과 음성으로 낸다.

```
 폰 브라우저 ──JPEG 프레임(웹소켓)──▶  서버 (FastAPI)                         ◀── XIAO / GoPro / RTSP / 웹캠
   ▲  카메라 · 음성(TTS) · 화면              │  ① 인식·추적 (YOLO)
   │                                         │  ② 트리거 = 내 v5_29 히트박스 (--trigger hitbox)
   └──── 경고 · 설명 · 특별 장면 (웹소켓) ◀────┤  ③ (선택) VLM 설명
                                              └  ④ 특별 장면 묘사 (불꽃놀이 · 벚꽃길 …, VLM)
```

## 실행

```bash
cd code/app
pip install -r requirements.txt

# A. 모델 없이 화면·흐름만 보기 (가짜 자동차가 커지며 경고가 뜸)
python server.py --https --perception demo --trigger hitbox --vlm none --scene mock

# B. 실제 모델 (GPU 서버 권장, 예: A5000)
python server.py --https --perception yolo --weights /경로/final_10class_best.pt --device 0 \
                 --trigger hitbox --cam-h 1.5 --scene anthropic
```
- 터미널에 뜨는 `https://<노트북IP>:8443` 을 **폰 브라우저**로 열기 (같은 와이파이). 처음 한 번 "안전하지 않음" 경고 → 고급 → 계속 (자체 서명 인증서라서요). 폰 카메라는 HTTPS에서만 켜집니다.
- 서버 카메라: `--source http://<XIAO>:81/stream` (설정은 `XIAO_SETUP.md`) 또는 화면에서 "서버 카메라" 선택.
- 특별 장면 묘사: `--scene mock`(가짜) · `--scene anthropic`(환경변수 `ANTHROPIC_API_KEY`, 모델 기본 claude-sonnet-5-5, `--scene-model`로 변경) · `--scene openai --scene-url http://서버:8000/v1`(vLLM 등 OpenAI 호환).

## v5_29 실시간 이식

판정 코드는 **손으로 다시 쓰지 않았어요.** `hitbox_original_v5_29.py`(내 파일 그대로)를 `hitbox_build.py`가 읽어 `hitbox_core.py`를 만듭니다. 상수·함수·`for fr in frames:` 루프 본문·이벤트 만들기는 줄 단위로 같고, **입력을 공급하는 곳만** 바꿨어요(바꾼 곳은 `hitbox_build.py`의 `PATCHES` 표에 전부 있음; 원본이 바뀌면 변환이 에러로 멈춤).

| 원본 입력 (pass1이 파일로 저장) | 실시간 공급 |
|---|---|
| YOLO conf 0.2 추적 | 앱의 YoloTracker (hitbox면 자동으로 conf 0.2) |
| 배경 광류 쌍 | `FlowTracker` — `pass1_detect.RecordingMotion`과 같은 계산, 매 프레임 |
| 신호등 + 상자 안 색(HSV) | `LightWorker` — 별도 스레드, 3프레임마다 (바쁘면 건너뜀) |
| Mask2Former 바닥 분할 | `SurfaceWorker` — 별도 스레드, 5프레임마다 (GPU 필요) |
| group_speech | `OnlineSpeech` — 같은 규칙을 한 사건씩 |

**검증** (`python tests/test_hitbox_port.py`, 모델·GPU·영상 불필요): 가짜 장면(검출·광류·신호등·바닥)을 원본 `run()`과 실시간 코어에 똑같이 넣어 경고 사건이 **전부 일치**(7개), 말하기 묶기는 무작위 사건열 300개에서 원본과 일치. 즉 *같은 입력이면 같은 판정*이에요. 실제 영상에서의 정확도는 이 테스트로는 알 수 없고, 아래 차이 때문에 오프라인 결과와 100% 같지는 않습니다.

### 원본과 다른 점 (정직하게)
- 바닥 분할·신호등은 **비동기**라 몇 프레임 늦게 반영돼요 (원본은 같은 프레임 번호의 결과). 모델이 없거나 느리면 그 입력만 빠진 채 돌고, 화면 아래 상태줄에 `신호등 off / 바닥 off`로 표시돼요.
- 사람 경고의 "붐빔 걸러내기"(v5_13)는 앞뒤 ±2 s를 보던 걸 **지난 2 s만** 봅니다.
- 폰은 프레임이 ~10 fps이고 간격이 일정하지 않아요. 판정은 원본처럼 실제 시각(dt)으로 계산하지만, 15 fps로 튜닝된 시간 문턱(0.2 s 유지 등)이 조금 다르게 걸릴 수 있어요.
- 폰 영상은 긴 변 960 px로 보냅니다(원본은 최대 1920). `y2 − 지평선 > 15 px` 같은 **픽셀 문턱**은 영상이 작으면 먼 물체에서 달라져요.
- **카메라 높이**: 원본 기본 1.70 m. 폰을 가슴 앞에 들면 `--cam-h 1.3~1.5`처럼 실제 높이를 넣으세요 (거리 추정이 이 값에 비례).
- **세로 영상 가정**: 바닥 분할 위치 가정이 세로(540×960) 기준이에요. 폰은 세로로 들고 쓰세요. 가로 영상(XIAO)은 가로용으로 리사이즈해 돌리지만 원본에서 검증된 적은 없어요.
- 경고 문장은 SJ 실행 시스템 패치와 같은 형식: `정면 자동차가 다가와요. 멈추세요.` / `1시 방향 기둥, 11시 방향 자전거가 있어요.` / 우선순위 멈추세요 > 피하세요 > 위치 안내(폰 음성도 같은 규칙으로 끊고 이어감). 신호등 변경은 `신호등이 초록불로 바뀌었어요. 건너가세요.` — **"건너가세요"처럼 행동을 지시하는 문구는 안전 측면에서 팀이 한 번 더 정해 주세요** (`hitbox_trigger.compose`).
- hitbox일 때는 경고마다 VLM 설명을 덧붙이지 않아요 (문장이 이미 위치·움직임·행동을 말함). 특별 장면 묘사는 그대로 동작.

### 노트북만으로는 어디까지?
LG gram(CPU)에서는 YOLO가 느려 실시간이 어렵고, Mask2Former(바닥 분할)는 사실상 불가능해요. 현실적인 구성: **GPU 서버(A5000)에서 서버를 돌리고 폰은 그 주소로 접속**(폰 카메라는 HTTPS 필요, 서버가 외부에서 보이려면 같은 망/VPN/터널). 노트북에서는 `--perception demo`로 화면·흐름만 확인하거나 `HB_SURFACE=0`로 바닥 분할을 끄고 시험하세요.

## 파일
| 파일 | 역할 |
|---|---|
| `server.py` | FastAPI · 웹소켓 · 자체 서명 HTTPS · QR |
| `pipeline.py` | 프레임 → 인식 → 트리거 → 경고/설명, 상태 |
| `hitbox_trigger.py` | v5_29 실시간 트리거(입력 공급 · 말하기) |
| `hitbox_core.py` | **자동 생성** v5_29 판정 코어 (직접 고치지 않기) |
| `hitbox_original_v5_29.py` / `hitbox_build.py` | 원본 / 변환기 |
| `scene.py` | 특별 장면 묘사 (불꽃놀이 · 벚꽃길 …) |
| `sources.py` | 폰·URL·RTSP·웹캠 소스 |
| `web/` | 폰 화면 (디자인은 Figma로 교체 예정, 아래 참고) |
| `tests/` | `test_hitbox_port.py`(원본 대조) · `hitbox_pipeline_smoke.py` · `ws_inprocess.py` |

## Figma 디자인 입히기
로직과 화면은 분리돼 있어요. `web/index.html`의 요소 id(`#go`, `#mPhone`, `#mServer`, `#url`, `#feed`, `#caption`, `#meta`, `#s-det` …)와 웹소켓 메시지만 지키면 모양은 마음대로 바꿔도 됩니다.

| `type` | 내용 |
|---|---|
| `hello` | 서버 설정 |
| `dets` | 박스 `[id, cls, x1, y1, x2, y2]` (0–1 정규화) |
| `warn` | 즉시 경고 `text`, `prio`(3 멈춰 · 2 피해 · 1 위치), `level`, `scenario` |
| `desc` | VLM 설명 (hitbox에서는 기본 꺼짐) |
| `scene` | 특별 장면 `kind`, `label`, `text` |
| `status` | fps · 지연 · `hb`(걸음 속도, 길 유형, 길 방향, 신호등/바닥 on/off) |
| `source` / `scene_error` | 상태 알림 |

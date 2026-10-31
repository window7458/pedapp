# 실행 시스템 (docs/13 A) — 트리거 → 분배기 → 즉시 경고 / VLM 설명 → TTS → 재생 큐

녹화 영상을 **실제 속도**로 재생하면서 위험할 때 소리를 낸다. 모든 단계의 시각을 로그로 남기고, 시연 영상(소리 · 박스 · 자막)을 만든다.

```
프레임 (15 fps, 실제 속도)
  │
  ├─ ① perception.py   저장된 E0 트랙 재생 (노트북) | YOLO-seg + ByteTrack 실시간 (서버)
  ├─ ② triggers.py     step(frame) → [Event]   ← 트리거 교체 지점 (B0 · 저장된 이벤트 · 임태규 최종 코드)
  ├─ ③ dispatcher.py   같은 트랙 4 s · 같은 문장 3 s 안 중복 금지
  │      ├─▶ ④ 즉시 경고 "왼쪽 오토바이" (미리 합성한 wav, VLM 안 기다림)
  │      └─▶ ⑤ vlm_worker.py  트리거 순간의 과거 8프레임 + 빨간 박스 → VLM (별도 스레드, 밀리면 최신 것만)
  │              → 하이브리드 (대상 · 움직임 · 위험 = VLM, 방향 = 박스, 행동 = 규칙표) → 문장
  ├─ ⑥ speech.py       문장 → wav (캐시 · 앞뒤 무음 제거)
  └─ ⑦ audio.py        재생 큐: 경고가 설명을 끊음 · 늦은 설명 폐기 · 새 경고가 나가면 이전 이벤트 설명 폐기
⑧ log.jsonl · timeline.json · summary.json  →  render.py: demo.mp4
```

## 실행

```bash
cd 종합설계/code/runtime
# 노트북 (서버 없음): 저장된 트랙 + B0 + VLM 지연 1 s 흉내 + Windows 한국어 음성, 스피커 재생
~/egoenv/Scripts/python.exe run_demo.py --clip c09_bike_on_tactile --vlm mock --mock-latency 1.0 --play
# 시연 영상 (소리 + 박스 + 자막) → results/runtime/<run>/<clip>/demo.mp4
~/egoenv/Scripts/python.exe run_demo.py --clip c09_bike_on_tactile --vlm mock --render
# 서버 (A5000): YOLO 실시간 + vLLM + TTS 서버
python run_demo.py --clip c09_bike_on_tactile --perception yolo --device 0 \
    --vlm server --model qwen2_5-vl-7b --base-url http://localhost:8000/v1 \
    --tts server --tts-url http://localhost:7788/v1 --tts-model supertonic --tts-voice F1
# 다른 트리거의 오프라인 결과로 (results/events/<폴더>/<clip>.json)
~/egoenv/Scripts/python.exe run_demo.py --clip c05_path_obst_recede --trigger replay:b0_E0 --vlm pt
```

| 옵션 | 값 |
|---|---|
| `--trigger` | `b0` (스트리밍 B0) · `replay:<results/events 폴더>` |
| `--vlm` | `none` (경고만) · `pt` (VLM 없이 템플릿, 지연 0) · `mock` (pt + `--mock-latency` 대기) · `server` (vLLM) |
| `--tts` | `sapi` (Windows Heami, 임시) · `server` (OpenAI 호환 TTS 서버, 예: `supertonic serve`) |
| `--no-hybrid` · `--vlm-filter` | 하이브리드 끄기 · VLM이 위험 아님이라 답하면 설명 생략 (10/9부터 기본은 항상 설명, 위험 판단은 트리거) |

## 트리거 끼우기 (임태규 최종 코드)

```python
# triggers.py 에 추가하고 make_trigger 에 이름 등록
class TaegyuTrigger:
    name = "taegyu"
    def step(self, frame):            # frame.t, frame.image(BGR), frame.dets = [Det(track_id, cls, conf, box, foot, edge, mask_area)]
        ...                           # 상태는 객체가 들고 있음 (이력 · 배경 움직임 등)
        return [Event(t=frame.t, cls=..., scenario="S1", track_id=..., box=..., W=frame.W, H=frame.H)]
```
S3(신호)는 `extra={"state": "GO"|"STOP"}`를 넣으면 경고가 "초록불로 바뀜 / 빨간불로 바뀜"이 된다.

## 확인한 것 (2026-10-07)

| 확인 | 결과 |
|---|---|
| 스트리밍 B0 = 오프라인 B0 (`test_b0_stream.py`) | 10편 이벤트 시각 · 시나리오 · 트랙 **모두 같음** |
| 재생용 영상 (`--proxy`) | h264 crf 18로 재압축하면 B0가 10편 중 3편에서 달라짐 → **무손실**로 바꾼 뒤 같음. 4K 원본(c05)만 재생용 영상 사용 (`~/.cache/insight_proxy`, OneDrive 밖) |
| c09 끝까지 (B0 + mock 1 s + Heami) | 경고 지연 중앙 0.11 s · 설명 지연 중앙 1.38 s · 설명 6개 중 4개 재생 · 2개는 새 경고에 끊김 |
| c05 (4K) | 원본 직접 디코딩은 10 s 분량에 25.8 s (실시간 불가) → 재생용 영상으로 프레임 지연 중앙 0.001 s |

## 정해 둔 규칙 (바꿀 수 있음, `DispatchConfig` · `AudioQueue`)

| 규칙 | 기본값 | 이유 |
|---|---|---|
| 경고 마감 | 트리거 + 2 s | 늦은 경고는 이미 지나간 위험 |
| 설명 마감 | 트리거 + 6 s | |
| 새 경고가 나가면 이전 이벤트 설명 폐기 | 켬 | c09에서 1.1 s 오토바이 설명이 3.1 s 장애물 경고 뒤에 나가고 장애물 설명은 버려지던 문제 |
| VLM 위험 아님 → 설명 생략 | 켬 | 경고는 이미 나갔음 |
| VLM 대상 ≠ YOLO 클래스 → VLM 쪽 말함 | 켬 (불일치는 로그) | 여러 프레임을 보고 판단한 쪽 |
| TTS wav 앞뒤 무음 제거 | 켬 | Heami는 앞뒤 무음이 커서 경고 하나가 약 2 s → 설명 지연 중앙 1.99 → 1.38 s |

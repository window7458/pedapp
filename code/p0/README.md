# P0 — 트리거 → VLM → TTS

근거: [`docs/11_P0_설계근거.md`](../../docs/11_P0_설계근거.md) · 지표: [`docs/09`](../../docs/09_평가데이터_지표_실험설계.md), [`docs/10`](../../docs/10_공개벤치마크_평가지표.md)

| 파일 | 역할 |
|---|---|
| `config.py` | 모델 후보 5개(S-M), 프레임 설정 F1/F3/F8/F32(S-F, 논문 설정), bbox 방식(S1), 출력 어휘 · JSON 스키마 |
| `frames.py` | 트리거 시각까지의 프레임 창(미래 프레임 없음), 트랙 bbox 표시 · 크롭, 448px 축소 |
| `describe.py` | VLM 경계 함수 `describe()` — vLLM OpenAI 호환 서버, JSON 스키마 강제, 스트리밍으로 TTFT 측정 |
| `ko_template.py` | JSON → 한국어 문장, 즉시 경고 템플릿, 허용 행동 규칙표, VLM 없는 기준선 PT |
| `tts.py` | 로컬 TTS 서버 호출 + wav 캐시 (경고 템플릿 미리 합성) |
| `run_p0.py` | 클립 단위 오프라인 실행 → `results/vlm/<run>/<clip>.jsonl` |
| `serve_vllm.py` | 후보 모델을 같은 인자로 vLLM 서버에 띄움 |

## A5000에서

```bash
pip install "vllm>=0.11" openai opencv-python-headless    # Qwen3-VL은 vLLM 0.11 이상
python serve_vllm.py qwen2_5-vl-7b &                       # 서버 (포트 8000)

# S0 배관 점검
python run_p0.py --clips c09_bike_on_tactile --events b0_E0 --dry      # 입력만 저장해서 눈으로 확인
python run_p0.py --events b0_E0 --model qwen2_5-vl-7b --frames F8 --bbox draw

# S-M: 모델마다 서버를 바꿔 띄우고 같은 명령
for m in internvl3_5-8b internvl3_5-4b qwen2_5-vl-7b qwen2_5-vl-3b qwen3-vl-8b; do
  python serve_vllm.py $m & sleep 240
  python run_p0.py --events oracle --model $m --frames F8 --bbox draw --repeat 3
  kill %1
done

# S-F: S-M 상위 2개 모델 × F1 F3 F8 F32
# PT: python run_p0.py --events oracle --arm pt
```

## 확인이 필요한 것 (서버를 띄운 뒤)

- InternVL3.5의 `max_dynamic_patch`, Qwen3-VL의 `max_pixels` 인자 이름과 프레임당 실제 토큰 수 (`usage.prompt_tokens`로 확인)
- `response_format` json_schema가 각 모델에서 동작하는지 (형식 오류율 0 기대)
- Supertonic `serve`의 모델 · 음성 이름 (`--tts-model`, `--tts-voice`)
- 정답 시각 모드(`--events oracle`)는 라벨에 `target_track`이 들어간 뒤부터 bbox가 표시된다 (labels/v3 전에는 박스 없이 실행됨)

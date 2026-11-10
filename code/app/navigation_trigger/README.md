# 보행 보조 Trigger — 규칙 + TTS 합성

팀 앱에서 받은 프레임을 분석해 경고 이벤트를 반환하는 코드입니다. 위험 판정 규칙과 그 규칙이 사용하는 카메라 보정·추적·경로·신호등·이동 상태 코드 전체를 포함합니다.

## 포함 범위

- `trigger/`: 위험 판단, 이동 상태, 추적, 카메라 보정, 경로 및 신호등 규칙의 전체 의존 코드
- `trigger/models/final_10class_best.pt`: 기존 팀 10클래스 모델
- `run_single.py`: 로컬 영상·카메라로 이벤트 JSONL을 확인하는 실행 예시
- `tts.py`: Microsoft Edge TTS로 한국어 경고를 MP3로 합성하는 독립 모듈
- `requirements-noaux.txt`: 규칙 실행 의존성
- `requirements-tts.txt`: TTS 합성 의존성
- 모델·제3자 코드 라이선스와 출처

영상 파일, 영상 업로드 서버, 모자이크, 리뷰 영상 제작, Colab 노트북, 과거 검증 로그는 포함하지 않습니다. `run_single.py`는 로컬 입력을 처리할 뿐 외부로 업로드하지 않습니다. 경고 문장은 “유의하세요”를 사용합니다. 모델 추론과 규칙의 동작은 기존 v12.1과 동일합니다.

## 설치 및 간단 확인

사용하는 CPU/CUDA 환경에 맞는 PyTorch 및 torchvision을 먼저 준비하세요.

```bash
cd navigation_trigger
python -m pip install -r requirements-noaux.txt
python run_single.py --source your_video.mp4 --device 0 --output events.jsonl
```

CPU 실행은 `--device cpu`를 사용합니다. 이 명령은 이벤트를 터미널과 JSONL 파일에 출력합니다.

## 앱 연결

저장소 루트에서 실행하는 예시입니다. 서버가 받은 이미지는 OpenCV BGR 배열로 전달합니다.

```python
import sys
from pathlib import Path

root = Path('navigation_trigger').resolve()
sys.path.insert(0, str(root / 'trigger'))
from single_model import Pipeline

# 세션 시작 시 한 번 생성합니다.
pipeline = Pipeline(str(root / 'trigger/models/final_10class_best.pt'),
                    device='0', camera_mode='egocentric')

# 각 새 프레임에 대해 실행합니다.
events, debug = pipeline.step(bgr_frame, frame_number, capture_time_seconds)
for event in events:
    print(event['message'])
    # 앱의 기존 VLM/TTS 처리부로 필요한 이벤트를 전달합니다.
```

각 카메라 세션마다 Pipeline을 따로 만들고 프레임 번호와 촬영 시각은 증가시켜야 합니다. 스트림이 바뀌면 `pipeline.reset()`을 호출하세요. 같은 Pipeline을 여러 스트림에서 동시에 호출하지 마세요.

이동수단의 `motion_state`는 `parked`(정차 추정), `moving`(이동 추정), `unknown`(보류)입니다. 이는 단안 영상에 대한 규칙 기반 추정입니다. 신호등 이벤트는 색상 전환을 안내합니다.

팀 앱이 이미 검출을 수행한다면 `single_model.py`의 `SingleModelCore.step_observations()`도 사용할 수 있습니다. 다만 bbox만 전달하는 인터페이스가 아닙니다. 카메라 보정, 보행 방향, 이미지 품질, 신호등 색상 등 동일한 관측 입력이 필요하므로 현재 앱 구조에 맞춘 연결 작업이 필요합니다. 위 `Pipeline.step()` 예시는 그 전처리와 모델 추론을 함께 수행합니다.

## TTS 합성

기존과 같은 Microsoft Edge TTS의 `ko-KR-SunHiNeural`(선희), 속도 `+15%`를 사용합니다. 경고 끝말은 **“유의하세요”**로 유지하고, `12시 방향`은 `열두 시 방향`으로 읽습니다. MP3 출력에는 ffmpeg가 필요하지 않습니다.

`navigation_trigger` 폴더에서:

```bash
python -m pip install -r requirements-tts.txt
python tts.py --text "12시 방향, 이동 중인 차량에 유의하세요."
python tts.py --events events.jsonl
```

MP3는 `tts_cache/`, 경고별 파일 대응은 `tts_manifest.json`에 저장됩니다. 같은 문구·음성·속도는 캐시를 재사용합니다. JSON 목록 또는 한 줄에 이벤트 하나인 JSONL을 입력할 수 있습니다.

앱/서버의 비동기 경고 처리부에서 사용할 예시입니다. 아래 `root`는 위 앱 연결 예시의 `navigation_trigger` 경로입니다.

```python
sys.path.insert(0, str(root))
from tts import SpeechSynthesizer

async def handle_warnings(events):
    tts = SpeechSynthesizer(cache_dir=root / 'tts_cache')
    for event in events:
        mp3_path = await tts.synthesize_event(event)
        # 기존 앱의 오디오 전송·재생 함수에 mp3_path를 전달합니다.
```

지속 실행하는 서버에서는 같은 이벤트 루프 안에서 `SpeechSynthesizer`를 한 번 생성해 재사용하세요. 합성 작업은 영상 추론 루프와 분리하고, 재생은 앱에서 순서대로 처리해야 음성이 겹치지 않습니다. 이 모듈은 MP3 생성까지 담당하며 서버 전송·스피커 재생·영상 합성은 포함하지 않습니다. manifest의 `time_s`는 원래 경고 시각이며 재생 예약 시각이 아닙니다.

합성 시 인터넷 연결이 필요하고 **경고 문장만 Microsoft TTS 서비스로 전송**합니다. 영상·이미지·bbox·경고 시각은 전송하지 않습니다. 연결 실패·시간 초과 시 예외를 반환하며, 실패한 임시 음성은 캐시에 남기지 않습니다. 최초 문구의 합성 지연은 네트워크와 서비스 상태에 따라 달라집니다.

## 확인 범위

배포 폴더를 줄이면서 규칙 소스 및 모델 내용은 변경하지 않았습니다. 기존 관련 규칙 테스트 85개가 통과한 소스를 사용합니다. 정리된 폴더에서 `SingleModelCore` import, 로컬 실행 CLI, 내부 Python 의존 파일, 모델 해시를 확인했습니다. TTS 문구 변환·동일 문구 캐시·동시 호출·실패 파일 정리는 외부 서비스를 호출하지 않는 대체 합성기로 검증했습니다. 이 작업에서는 실제 음성 합성, GPU 추론 및 팀 서버 연결을 실행하지 않았습니다.

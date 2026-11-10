# 사용한 구현과 가중치

- Ultralytics: https://github.com/ultralytics/ultralytics
  설치된 배포본의 라이선스를 ULTRALYTICS_LICENSE.txt에 포함했습니다.
- OpenCV: https://opencv.org/
  영상 처리, optical flow, 카메라 움직임 추정에 사용합니다.
- NumPy / SciPy: 배열 계산과 추적 대상 배정에 사용합니다.
- final_10class_best.pt: 팀이 제공한 가중치 그대로입니다. 추가 학습하지 않았습니다.
  SHA-256: 5e3261f034f84fb3631eefe3e67451a10f8e2b0671cd1114346865fdd1c293d4

이 버전에는 공개 YOLO 가중치, Mask2Former, Grounding DINO 및 해당 다운로드 스크립트를 포함하지 않습니다.
기존 코드에서 재사용한 기하·추적·신호등 모듈은 전체 소스로 포함합니다.

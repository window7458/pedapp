# Third-party notices

## GuideTWSI
- Repository: https://github.com/DARoSLab/GuideTWSI
- Weight: https://github.com/DARoSLab/GuideTWSI/blob/master/model_weights/yolo11n_tactile.pt
- Original file is included without modification at weights/yolo11n_tactile.pt.
- SHA256: 79f0bb39a354dd66d367bb5b97184df626c85f91568456e099e49a8cdbfcd878
- Repository MIT license: licenses/GuideTWSI_MIT.txt
- Checkpoint metadata specifies AGPL-3.0 (https://ultralytics.com/license).
- Paper: GuideTWSI: A Diverse Tactile Walking Surface Indicator Dataset from Synthetic and Real-World Images for Blind and Low-Vision Navigation. Hwang et al., ICRA 2026.

## Ultralytics
Runtime dependency: Ultralytics, used for YOLO11n instance segmentation. Original image tests: 8.3.162. V2 inference checks: 8.4.174. The installer does not install or downgrade dependencies.
Source: https://github.com/ultralytics/ultralytics
License: AGPL-3.0; see https://www.ultralytics.com/license.

## Input images
The original private test package used four user-supplied sample images. Those images are not included in this repository distribution.

## Wrapper
segment_tactile.py is an implementation for this image experiment, not the authors' original inference script. It loads the unmodified public checkpoint via Ultralytics.

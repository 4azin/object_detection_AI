# Divery Vision Pipeline - CLI Integration Guide

이 문서는 `web_app.py`의 파이프라인(Best-shot 추출 및 어종 분류)을 Gradio 웹 브라우저 형태에서 벗어나, 백그라운드나 GPU 서버 등에서 단일 스크립트로 동작할 수 있게 만든 **`integration.py`** 의 사용법과 구조를 설명합니다.

---

## 📌 주요 특징
1. **커맨드 라인 제어 (CLI 지원)**
   Gradio UI에 의존하지 않고 터미널에서 `argparse`를 활용하여 명령어 옵션으로 파라미터를 입력할 수 있습니다.
2. **GPU 서버 / 백그라운드 환경 최적화**
   진행 상황 등을 확인할 수 있는 UI 로직 대신 `tqdm` 등을 활용하고, 콘솔로 상태를 깔끔하게 출력하도록 변환했습니다. 자동화 스크립트나 배치 작업에 적합합니다.
3. **통합 파이프라인 구축**
   Step 1 (Best-shot 추출)의 결과물을 자동으로 타겟 파일로 읽어들인 뒤 바로 Step 2 (분류 모델)로 넘겨, 중간 수동 조작 과정 없이 CSV 포맷의 최종 분류 결과를 도출합니다.

---

## 🚀 사용법 및 실행 옵션

추론을 진행할 가상환경(`conda activate diveary-vision`)이 활성화된 상태여야 합니다. 

### 기본 실행 예시
1. **비디오 파일을 이용해 Open-Domain 분류 진행 시**
   ```bash
   python integration.py --source /경로/비디오.mp4 --mode open-domain
   ```
2. **이미지 파일을 이용해 Zero-shot 분류 진행 시 (커스텀 클래스 지정)**
   ```bash
   python integration.py --source /경로/이미지.jpg --mode zero-shot --classes "Amphiprion ocellaris,Zebrasoma flavescens"
   ```

### 상세 옵션 설명 (Arguments)

CLI 실행 시 사용할 수 있는 옵션은 다음과 같습니다:

| 옵션 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--source` | `str` | **(필수)** | 처리할 입력 비디오(`.mp4`, `.avi` 등) 또는 이미지(`.jpg`, `.png` 등) 파일 경로 |
| `--conf` | `float` | `0.5` | Tracking(비디오) / Detection(이미지) 신뢰도(Confidence) 임계값 |
| `--no-fp16` | `flag` | - | GPU 연산 시 FP16(Half precision) 최적화를 끄고 Full precision 연산을 사용합니다 (메모리 사용량 증가) |
| `--model` | `str` | `bioclip2` | 분류에 사용할 모델 설정 (옵션: `bioclip2`, `bioclip-2.5-vith14`) |
| `--mode` | `str` | `open-domain` | 분류 모드 (옵션: `open-domain`, `zero-shot`, `zero-shot-all`, `two-stage`)<br>**참고:** `bioclip-2.5-vith14` 모델은 현재 Open-domain을 지원하지 않습니다. |
| `--classes` | `str` | `""` | `zero-shot` 모드 활성화 시 사용할 타겟 어종 리스트. 콤마(`,`)로 구분하여 문자열로 입력합니다. (예: `"Fish A,Fish B"`) |

---

## 📁 출력 결과물 구조

스크립트 실행 완료 후, 프로젝트 폴더 내 `results/best_shots/` 디렉토리에 **타임스탬프**가 포함된 폴더가 생성되며 관련 파일이 저장됩니다.

```shell
results/best_shots/
└── {source_name}_{YYYYMMDD_HHMMSS}/
    ├── 1.jpg (추출된 Best-shot 1)
    ├── 2.jpg (추출된 Best-shot 2)
    ├── tracked_{source_name}.mp4 (비디오일 경우, Bounding Box 시각화 영상)
    └── classification_results.csv (최종 분류 결과가 정리된 파일)
```

- **`classification_results.csv`**: 추출된 이미지마다의 파일명, Inference(예측 시간), Top 1/2/3 예측 종 및 Confidence 확률 정보 등이 표 형태로 저장됩니다.

---

## 🛠 코드 동작 흐름 (Pipeline)

`integration.py`는 내부적으로 다음과 같은 2가지 주요 Step을 연속으로 수행합니다.

### Step 1: Best-Shot Extraction (`run_extraction_video` / `run_extraction_image`)
- 입력된 미디어 파일(Video/Image)을 읽습니다.
- YOLO 탐지 및 ByteTrack(비디오)을 이용하여 움직이는 객체를 포착합니다.
- 화질과 Confidence 점수를 평가하여 객체 ID(트랙)별 Best-shot 프레임을 크롭(Crop) 합니다.
- **Safeguard (Deduplication):** 프레임 간 유사도(Embedding)를 비교해 동일한 객체의 중복 컷을 걸러내고, 유의미한 주요 이미지만 저장합니다.

### Step 2: Species Classification (`run_classification`)
- Step 1에서 추출&저장된 Best-shot 이미지 타겟 목록을 읽어옵니다.
- 선택된 분류 모델(`--model`)과 모드(`--mode`)를 사용하여 BioCLIP 추론을 수행합니다.
- 결과를 종합하여 DataFrame을 만들고 `classification_results.csv`로 내보낸 후, 콘솔에 최상위(Top-1) 예측 정보를 요약 출력합니다.

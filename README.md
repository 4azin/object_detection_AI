# Divery Vision Pipeline

> **수중 생물 탐지 · 추적 · 분류를 위한 AI 파이프라인**

CFD(Custom Fish Detector) + ByteTrack 추적 + BioCLIP-2 기반 다중 모드 어종 분류를 통합한 Gradio 기반 웹 인터페이스입니다.

---

## 📁 프로젝트 구조

```
image_pjt_test/
├── CFD_fishial/            # 🐠 핵심 파이프라인 (탐지 + 추적 + 분류)
│   ├── web_app.py          # Gradio 웹 UI (통합 실행 진입점)
│   ├── config.yaml         # 전체 파이프라인 설정
│   ├── best_shot_extractor.py  # ByteTrack 기반 Best-Shot 추출
│   ├── detector_tester.py  # CFD 탐지 테스트
│   ├── classifier_tester.py# Fishial.AI 분류 테스트
│   ├── deduplicator.py     # 임베딩 기반 중복 제거
│   ├── bytetrack_custom.yaml   # ByteTrack 트래커 설정
│   ├── run_best_shot.py    # Best-Shot CLI 실행 스크립트
│   ├── run_detector_test.py    # 탐지 CLI 실행 스크립트
│   ├── run_classifier_test.py  # 분류 CLI 실행 스크립트
│   └── fishial_model/      # Fishial.AI 모델 파일
│       ├── model.ckpt       # 분류 모델 체크포인트 (~330MB)
│       ├── database.pt      # kNN 임베딩 데이터베이스 (~137MB)
│       ├── inference.py     # Fishial 추론 로직
│       └── info.json        # 모델 메타정보 (v10.0, 755종)
│
├── classified_test/        # � BioCLIP-2 종 분류 Gradio Web App
│   └── classify_app.py     # Open-Domain 및 Zero-Shot 분류 UI
│
├── GMS_test/               # 🤖 GMS 통합 이미지 분석 AI
│   ├── logbook_pipeline.py # gms 제공 모델 추가 및 페르소나 설정
│   ├── image_app.py        # Gradio 기반 이미지 특성 분석 LLM 연동
│   ├── test_llm.py         # gpt-5-mini API 연결 테스트
│   └── test_image.py       # base64 변환 테스트
│
├── requirements.txt        # 통합 의존성 패키지 목록
└── .gitignore
```

---

## 🖥️ 시스템 요구사항

| 항목       | 권장 사양                        |
| ---------- | -------------------------------- |
| **OS**     | Windows 10/11 (64-bit)           |
| **GPU**    | NVIDIA RTX 4070 이상 (VRAM 8GB+) |
| **CUDA**   | 12.4                             |
| **Python** | 3.10 ~ 3.11                      |
| **RAM**    | 16GB 이상                        |

---

## 🚀 초기 세팅 (환경 구성)

### 1. Conda 가상환경 생성

```bash
# 가상환경 생성 (Python 3.11 권장)
conda create -n diveary-vision python=3.11 -y

# 가상환경 활성화
conda activate diveary-vision
```

### 2. PyTorch 설치 (CUDA 12.4)

> ⚠️ **반드시 CUDA 버전에 맞는 PyTorch를 먼저 설치**해야 합니다.

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

설치 확인:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 3. 의존성 패키지 설치

```bash
# 프로젝트 루트에서 실행
pip install -r requirements.txt
```

### 4. 모델 가중치 파일 배치

모델 파일은 `.gitignore`에 의해 Git에서 제외됩니다. 팀원 간 공유(Google Drive, NAS 등)를 통해 아래 경로에 배치하세요.

| 파일                   | 경로                         | 크기   | 설명                       |
| ---------------------- | ---------------------------- | ------ | -------------------------- |
| `cfd-yolov12x-1.00.pt` | `CFD_fishial/`               | ~114MB | CFD 탐지 모델 (YOLOv12x)   |
| `yolo11n.pt`           | `CFD_fishial/`               | ~5.4MB | YOLO11n (기본 모델)        |
| `model.ckpt`           | `CFD_fishial/fishial_model/` | ~331MB | Fishial.AI 분류 체크포인트 |
| `database.pt`          | `CFD_fishial/fishial_model/` | ~137MB | Fishial.AI kNN 임베딩 DB   |

---

## ▶️ 실행 방법

### CFD + Fishial.AI 웹 UI (메인 파이프라인)

```bash
conda activate diveary-vision
cd CFD_fishial
python web_app.py
```

실행 후 브라우저에서 **http://localhost:7860** 접속

#### 사용 흐름

1. **Step 1** — 영상 업로드 → ByteTrack 추적 → Best-Shot 자동 추출 (압축 파일 다운로드 지원)
2. **Step 2** — 추출된 이미지 선택 → BioCLIP-2 다중 모드 예측 적용
   - **Open-Domain**: TreeOfLife-200M 임베딩을 이용한 기본 오픈 도메인 분류
   - **Zero-Shot (Custom)**: 사용자가 입력한 종 목록 내에서만 분류
   - **Zero-Shot (All Bioinfo List)**: 내부 JSON(`taxon_data...`)에 등록된 전 종 목록으로 분류
   - **Two-Stage**: 과(Family) 예측 후 해당 과의 종(Species)들을 대상으로 분류
   > 💡 예측 결과는 Top-3 학명과 함께 **한국어 일반명** 조합으로 반환됩니다. (예: `Acanthopagrus schlegelii (Blackhead seabream) [감성돔]`)

#### 입력 제한 사항

| 항목                 | 제한                                                  |
| -------------------- | ----------------------------------------------------- |
| **최대 업로드 크기** | 500MB                                                 |
| **지원 영상 포맷**   | mp4, webm, ogg 등 (브라우저 + OpenCV 지원 포맷)       |
| **지원 코덱**        | H.264, H.265, VP9 등 (FFmpeg 디코딩 가능한 모든 코덱) |
| **해상도 제한**      | 없음 (모델 내부에서 640px로 리사이즈)                 |
| **영상 길이 제한**   | 없음 (단, 긴 영상은 처리 시간 ↑)                      |

---

### CFD CLI 스크립트 (배치 처리)

```bash
cd CFD_fishial

# 탐지 테스트 (단일 영상)
python run_detector_test.py --video test_videos/dive_clip.mp4

# 탐지 테스트 (디렉토리 내 모든 영상)
python run_detector_test.py

# Best-Shot 추출
python run_best_shot.py --video test_videos/dive_clip.mp4

# 분류 테스트
python run_classifier_test.py --dir results/best_shots
```

> CLI 스크립트의 영상 검색 확장자는 `config.yaml`의 `input.extensions` 항목에서 설정합니다 (기본값: `.mp4`, `.mov`).

---

### BioCLIP-2 종 분류 웹 UI

```bash
conda activate diveary-vision
cd classified_test
python classify_app.py
```

실행 후 브라우저에서 **http://localhost:7860** 접속

---

### GMS (LLM & Vision) 통합 분석 파이프라인

> 실행 전 `GMS_test` 폴더에 `.env` 파일을 생성하고 `GMS_KEY=설명_API_KEY` 를 입력하세요.

1. **단일 이미지 분석 App (`image_app.py`)**  
   단일 이미지 1장과 프롬프트를 입력받아 특징을 요약하는 가벼운 분석 도구입니다.

   ```bash
   cd GMS_test
   python image_app.py
   ```

   실행 후 **http://localhost:7860** 접속

2. **🌊 Divery AI 다이빙 로그북 생성 App (`logbook_pipeline.py`)** **[NEW ✨]**  
   동영상에서 추출된 **5개의 주요 프레임**을 시계열로 분석해 서사형 다이빙 로그북을 자동 생성합니다.
   - **4가지 권장 모델 프리셋 지원** (최고 성능 GPT-5 / Claude 하이브리드 / 가성비 모델 등)
   - **4가지 다이버 성향 페르소나 적용**: (기록형, 예술가형, 실속형, 블로거형)
   - ⚠️ _결과 로그(JSON)는 `GMS_test/history/` 폴더에 자동저장됩니다._

   ```bash
   cd GMS_test
   python logbook_pipeline.py
   ```

   실행 후 **http://localhost:7865** 접속

---

## ⚙️ 주요 설정 파일

### `CFD_fishial/config.yaml`

```yaml
model:
  weights: "cfd-yolov12x-1.00.pt"
  confidence_threshold: 0.25 # 탐지 신뢰도 임계값
  iou_threshold: 0.45 # NMS IOU 임계값
  img_size: 640 # 입력 이미지 크기
  half: true # FP16 사용 (RTX 4070)

tracker:
  tracker_type: "bytetrack_custom.yaml"
  confidence_threshold: 0.5 # 추적 신뢰도 임계값

classifier:
  backbone_model_name: "maxvit_base_tf_224"
  use_knn: true # kNN 하이브리드 추론
  topk_results: 3 # Top-K 결과 개수
  device: "cuda:0"

safeguard:
  embedding_similarity_threshold: 0.85 # 임베딩 중복 제거 임계값
  merge_frame_gap: 150 # 프레임 간격 병합 기준
  merge_similarity_threshold: 0.80 # 병합 유사도 임계값
```

---

## 🔧 트러블슈팅

### CUDA 오류 발생 시

```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch CUDA 연결 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### Gradio 포트 충돌 시

여러 개의 웹 UI(Gradio 앱)가 모두 기본 포트 `7860`을 사용합니다. 동시 실행 시 하나의 포트를 `web_app.py` 등의 파일 안에서 `demo.launch(server_port=7861)`과 같이 변경하세요:

```python
# web_app.py 또는 app.py에서 수정
demo.launch(server_port=7861)
```

---

## 📝 기타 참고사항

- **`.gitignore`**: 모델 가중치(`.pt`, `.ckpt`), 영상 파일(`.mp4`, `.mov`, `.avi`), 로그, 결과 디렉토리 등은 Git 추적에서 제외됩니다.
- **테스트 영상**: `CFD_fishial/test_videos/` 디렉토리에 테스트 영상을 넣어 사용합니다 (gitignore 대상).
- **결과 출력**: 탐지/추적 결과는 `CFD_fishial/results/`에 저장됩니다.

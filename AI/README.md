# Divery AI Pipeline

> **수중 생물 탐지 · 추적 · 분류 · 로그북 생성을 위한 AI 파이프라인**

CFD(Custom Fish Detector) + ByteTrack 추적 + BioCLIP-2 분류 + Vision LLM 로그북 생성을 통합한 GPU 기반 워커입니다.

---

## 아키텍처

```
┌─────────────┐   SQS Queue   ┌─────────────┐   S3 Callback   ┌─────────────┐
│  Frontend   │ ─────────────>│  GPU Worker │ ───────────────>│   Backend   │
│  (Upload)   │               │  (AI Infer) │                 │  (Result)   │
└─────────────┘               └─────────────┘                 └─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌──────────┐    ┌──────────┐    ┌──────────┐
             │Highlight │    │   CFD    │    │ Vision   │
             │Selection │    │ByteTrack │    │   LLM    │
             │(YOLO-WL) │    │ BioCLIP  │    │ Logbook  │
             └──────────┘    └──────────┘    └──────────┘
```

---

## 프로젝트 구조

```
AI/
├── worker/                      # SQS 기반 GPU 워커
│   ├── worker.py                # 메인 워커 (통합 파이프라인)
│   ├── requirements.txt         # Python 의존성
│   ├── .env.example             # 환경 변수 템플릿
│   └── scripts/                 # 배포/운영 스크립트
│       ├── setup_gpu_worker.sh  # 초기 환경 설정
│       ├── start_worker.sh      # 워커 시작
│       ├── status_worker.sh     # 상태 확인
│       └── import_model_weights.sh  # 모델 가중치 설치
│
├── CFD_fishial/                 # 물고기 탐지/분류 모듈
│   ├── integration.py           # CFD 통합 스크립트 (worker 호출)
│   ├── best_shot_extractor.py   # ByteTrack 기반 Best-Shot 추출
│   ├── classifier_tester.py     # BioCLIP-2 분류기
│   ├── deduplicator.py          # 임베딩 기반 중복 제거
│   ├── config.yaml              # 파이프라인 설정
│   ├── cfd-yolov12x-1.00.pt     # CFD 탐지 모델 (~114MB)
│   ├── yolo11n.pt               # YOLO11n 모델 (~5.4MB)
│   └── weights/
│       └── yolov8l-world.pt     # YOLO-World Large (하이라이트 선정)
│
├── dist/                        # 빌드 아티팩트 출력
├── Dockerfile                   # 컨테이너 빌드
├── requirements.gpu.txt         # GPU 전용 의존성
└── README.md                    # 이 파일
```

---

## 파이프라인 상세

### 1. 하이라이트 선정 (Highlight Selection)

- **YOLO-World Large** 모델로 각 프레임 점수화
- 탐지 confidence + 객체 크기 + 선명도(Laplacian) 융합
- 시간 간격 고려하여 Top-K 프레임 선정 (기본 5장)

### 2. 물고기 탐지/추적 (CFD + ByteTrack)

- **CFD-YOLOv12x** 커스텀 물고기 탐지 모델
- **ByteTrack** 개체별 ID 추적
- 각 개체 ID별 최고 confidence 크롭 저장
- 임베딩 기반 중복 제거 (threshold: 0.85)

### 3. 어종 분류 (BioCLIP-2)

- **BioCLIP-2** (TreeOfLife-200M) 454,000+ 종 분류
- Open-domain / Zero-shot / Two-stage 모드 지원
- 학명 → Wikidata 한국어명 자동 매핑

### 4. 로그북 생성 (Vision LLM)

두 가지 방식 지원 (우선순위 순):

1. **Vision LLM 파이프라인** (권장)
   - 하이라이트 이미지 → Vision LLM 분석 → 프레임별 텍스트 생성
   - 5개 텍스트 → Synthesis LLM → 최종 로그북 생성

2. **Text-based LLM 파이프라인** (fallback)
   - 탐지/분류 데이터 JSON → LLM → 로그북 생성

**페르소나 스타일**:
- 기록형: 객관적 관찰 기록 스타일
- 예술가형: 감성적이고 시적인 묘사
- 실속형: 핵심 팩트만 간결하게
- 블로거형: SNS 스타일 (이모지, 해시태그)

---

## 시스템 요구사항

| 항목 | 권장 사양 |
|------|-----------|
| **GPU** | NVIDIA RTX 4070+ (VRAM 8GB+) |
| **CUDA** | 12.4 |
| **Python** | 3.10 ~ 3.11 |
| **RAM** | 16GB+ |

---

## 환경 변수

```bash
# AWS / S3
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=divery-media
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx

# SQS Queue
SQS_QUEUE_URL=https://sqs.ap-northeast-2.amazonaws.com/xxx/divery-ai-queue

# Backend Callback
BACKEND_CALLBACK_BASE_URL=https://j14a402.p.ssafy.io
AI_CALLBACK_SIGNING_SECRET=xxx

# GMS LLM API
GMS_KEY=xxx

# Vision LLM Settings
VISION_LLM_ENABLED=true
VISION_LLM_MODEL=gpt-5-mini
SYNTHESIS_LLM_MODEL=o3-mini
```

---

## GPU 서버 배포

### 1. 환경 설정

```bash
cd AI/worker
cp .env.example .env
# .env 파일 편집하여 환경 변수 설정
```

### 2. 모델 가중치 설치

```bash
bash scripts/import_model_weights.sh
```

복사 대상:
- `CFD_fishial/cfd-yolov12x-1.00.pt` (필수)
- `CFD_fishial/yolo11n.pt` (옵션)
- `CFD_fishial/weights/yolov8l-world.pt` (옵션)

### 3. 워커 실행

```bash
bash scripts/setup_gpu_worker.sh  # 초기 설정
bash scripts/smoke_test.sh        # 테스트
bash scripts/start_worker.sh      # 백그라운드 실행
bash scripts/status_worker.sh     # 상태 확인
```

### 4. 배포 번들 생성

```bash
bash scripts/package_gpu_bundle.sh
```

생성 파일: `AI/dist/divery-gpu-worker-bundle-<timestamp>.tar.gz`

---

## 설정 파일

### `CFD_fishial/config.yaml`

```yaml
model:
  weights: "cfd-yolov12x-1.00.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  img_size: 640
  half: true  # FP16

tracker:
  tracker_type: "bytetrack_custom.yaml"
  confidence_threshold: 0.5

classifier:
  backbone_model_name: "maxvit_base_tf_224"
  use_knn: true
  topk_results: 3
  device: "cuda:0"

safeguard:
  embedding_similarity_threshold: 0.85
  merge_frame_gap: 150
```

---

## 트러블슈팅

### CUDA 오류

```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 모델 가중치 누락

```bash
ls -la CFD_fishial/*.pt
ls -la CFD_fishial/weights/
```

### SQS 연결 실패

```bash
# AWS 자격 증명 확인
aws sts get-caller-identity

# SQS 접근 테스트
aws sqs get-queue-attributes --queue-url $SQS_QUEUE_URL
```

---

## 개발 참고

- **worker.py**: SQS 메시지 처리, CFD 파이프라인 호출, Vision LLM 로그북 생성
- **integration.py**: CLI 방식 CFD 실행 (worker에서 subprocess로 호출)
- 모델 파일은 `.gitignore`로 Git 추적 제외
- 결과는 `CFD_fishial/results/`에 저장

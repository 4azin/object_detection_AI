# Divery AI Pipeline 설계서

## 1. 개요

다이빙 영상에서 자동으로 하이라이트를 추출하고, 수중 생물을 식별하여 개인화된 다이빙 로그를 생성하는 AI 파이프라인.

### 1.1 파이프라인 흐름

```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  영상    │ => │  하이라이트  │ => │  CFD 탐지   │ => │  BioCLIP    │ => │  LLM 로그   │
│  입력    │    │  추출 (5장) │    │  (물고기)   │    │  종 식별    │    │  생성       │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                │                  │                  │                  │
     ▼                ▼                  ▼                  ▼                  ▼
  .mp4/mov      상위 5장면         물고기 크롭        학명+한국어명       다이빙 로그북
```

### 1.2 인프라: Modal (서버리스 GPU)

| 항목 | 내용 |
|------|------|
| 선택 이유 | 초단위 과금, 무료 $30/월, Python 배포 간편 |
| GPU | T4 (개발), A10G (프로덕션) |
| 통신 | REST API (Web Endpoint) |

### 1.3 기존 인프라 현황

```
Divery 인프라 (docker-compose.yml)
├── Nginx (:80/443)
├── SpringBoot Backend (:8080)
├── PostgreSQL (:5432)
├── Redis (:6379)
└── S3 (divery-media) - 영상/이미지 저장
```

### 1.4 운영 방침: Warm Pool 미사용

```
Warm Pool 비용: ~$425/월 (T4 24시간 유지) → 비용 대비 효과 낮음

채택 방식: Cold Start 허용 + 비동기 처리
- Cold Start: 10-30초 발생 가능
- 비동기 처리로 사용자 체감 영향 없음
- 무료 크레딧 $30/월로 ~4,000건 처리 가능
```

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Divery AI Architecture                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              EC2 Server                                      │
│  ┌─────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐             │
│  │  Nginx  │───>│  SpringBoot  │───>│ Postgres│    │  Redis  │             │
│  │         │    │   (:8080)    │    │         │    │ (상태)  │             │
│  └─────────┘    └──────┬───────┘    └─────────┘    └─────────┘             │
│                        │                                                    │
└────────────────────────┼────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│     S3      │  │    Modal    │  │  External   │
│ (영상 저장) │  │ (GPU 추론)  │  │    APIs     │
└─────────────┘  └─────────────┘  └─────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  /analyze   │ │/detect-fish │ │/identify-species│
│ (하이라이트)│ │ (CFD 탐지)  │ │ (BioCLIP)   │
│  [T4 GPU]   │ │  [T4 GPU]   │ │  [T4 GPU]   │
└─────────────┘ └─────────────┘ └─────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────────┐
              │  Modal Volume   │
              │   (Weights)     │
              └─────────────────┘
```

### 영상 전달 흐름

```
1. Mobile App → SpringBoot: 영상 업로드
2. SpringBoot → S3: 영상 저장, URL 획득
3. SpringBoot → Modal: S3 URL 전달 (영상 파일 X)
4. Modal: URL로 영상 다운로드 → GPU 처리
5. Modal → SpringBoot: 결과 반환
```

---

## 3. 모듈별 상세 설계

### 3.1 Module A: 하이라이트 추출 (`/analyze`)

**기존 코드**: `divary_ai/HIGHLIGHT.py`

| 항목 | 내용 |
|------|------|
| 입력 | 영상 S3 URL |
| 출력 | 하이라이트 5장 (base64 + 메타데이터) |
| GPU | T4 |
| 예상 시간 | ~30초 (1분 영상) |

**처리 흐름**:
1. S3에서 영상 다운로드
2. 1FPS 프레임 추출 (In-Memory)
3. 흔들림 필터링 (Laplacian 분산 > 15.0)
4. CLAHE 수중 이미지 보정
5. YOLO-World Large로 수중 객체 탐지
6. NIMA로 미적 점수 평가
7. 점수 융합: `0.6 × YOLO + 0.4 × NIMA`
8. K-Means 클러스터링으로 다양성 확보
9. 최종 5장 선정 및 반환

**수중 탐지 클래스**:
```python
["colorful tropical fish", "sea turtle swimming", "manta ray",
 "nudibranch", "large shark", "jellyfish", "octopus",
 "beautiful coral reef", "scuba diver"]
```

**Modal 코드**:
```python
@app.function(image=image, gpu="T4", timeout=600, volumes={"/weights": volume})
@web_endpoint(method="POST")
def analyze(request: dict) -> dict:
    video_url = request["video_url"]
    top_k = request.get("top_k", 5)

    # 1. 영상 다운로드
    video_path = download_from_s3(video_url)

    # 2. 프레임 추출 + 필터링
    frames = extract_frames(video_path, fps=1)

    # 3. YOLO-World + NIMA 점수 계산
    yolo = YOLO("yolov8l-world.pt")
    nima = load_nima("/weights/nima_resnet50.pth")
    scored = score_frames(frames, yolo, nima)

    # 4. K-Means 다양성 선별
    highlights = select_diverse(scored, k=top_k)

    return {"highlights": highlights}
```

---

### 3.2 Module B: CFD 물고기 탐지 (`/detect-fish`)

**기존 코드**: `CFD_fishial/` 모델 활용

| 항목 | 내용 |
|------|------|
| 입력 | 이미지 5장 (base64) |
| 출력 | 물고기 크롭 이미지들 (base64 + bbox) |
| GPU | T4 |
| 예상 시간 | ~5초 |

**처리 흐름**:
1. 하이라이트 5장 수신
2. CFD (cfd-yolov12x) 모델로 물고기 탐지
3. Bounding Box 기반 크롭
4. 중복 제거 (IoU threshold)
5. 크롭 이미지 리스트 반환

**설정값**:
```yaml
detector:
  weights: "cfd-yolov12x.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
```

**참고**: ByteTrack 추적 미사용 (영상이 아닌 이미지 5장 대상)

**Modal 코드**:
```python
@app.function(image=image, gpu="T4", timeout=120, volumes={"/weights": volume})
@web_endpoint(method="POST")
def detect_fish(request: dict) -> dict:
    images_base64 = request["images"]

    model = YOLO("/weights/cfd-yolov12x.pt")

    detections = []
    for idx, img_b64 in enumerate(images_base64):
        image = decode_base64(img_b64)
        results = model.predict(image, conf=0.5)

        crops = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            crops.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf),
                "image_base64": encode_base64(crop)
            })

        detections.append({"source_index": idx, "crops": crops})

    return {"detections": detections}
```

---

### 3.3 Module C: BioCLIP 종 식별 (`/identify-species`)

**기존 코드**: `AI/bioclip_pipeline.py`

| 항목 | 내용 |
|------|------|
| 입력 | 물고기 크롭 이미지들 (base64) |
| 출력 | 종 정보 (학명, 영어명, 한국어명, 신뢰도) |
| GPU | T4 (또는 CPU 가능) |
| 예상 시간 | ~5초 |

**처리 흐름**:
1. 크롭 이미지 수신
2. BioCLIP (HuggingFace Space) API 호출
3. Top-K 예측 결과에서 학명 추출
4. Wikidata SPARQL로 한국어명 조회
5. 결과 반환

**BioCLIP 장점**:
- 454,000+ 종 지원
- 무료, 무제한 API
- Zero-shot 분류 가능

**Modal 코드**:
```python
@app.function(image=image, gpu="T4", timeout=120)
@web_endpoint(method="POST")
def identify_species(request: dict) -> dict:
    images_base64 = request["images"]

    results = []
    for img_b64 in images_base64:
        # BioCLIP API 호출
        prediction = call_bioclip(img_b64)

        # Wikidata 한국어명 조회
        korean_name = query_wikidata(prediction["scientific_name"])

        results.append({
            "scientific_name": prediction["scientific_name"],
            "common_name": prediction["common_name"],
            "korean_name": korean_name,
            "confidence": prediction["confidence"]
        })

    return {"species": results}
```

---

### 3.4 Module D: LLM 로그 생성 (SpringBoot에서 처리)

**기존 코드**: `GMS_test/logbook_pipeline.py`

| 항목 | 내용 |
|------|------|
| 입력 | 하이라이트 5장 + 종 식별 결과 |
| 출력 | 서사적 다이빙 로그북 |
| 실행 위치 | SpringBoot (GMS API 호출) |

**2단계 파이프라인**:

#### Step 1: Image-to-Text
각 하이라이트 이미지 + 종 정보 → 페르소나별 텍스트 묘사

| 페르소나 | 스타일 | 글자수 |
|----------|--------|--------|
| 기록형 (Record-keeper) | 객관적, 관찰 일지 | ~200자 |
| 예술가형 (Artist) | 시적, 감각적 | ~200자 |
| 실속형 (Pragmatist) | 핵심 팩트만 | ~150자 |
| 블로거형 (Blogger) | SNS 말투, 이모지 | ~200자 |

#### Step 2: Sequence-to-Logbook
5개 텍스트 → 시간순 연결 → 통합 로그북

**모델 프로파일**:
| 프로파일 | Vision Model | Synthesis Model | 비용 |
|----------|--------------|-----------------|------|
| 가성비 | gpt-5-mini | o3-mini | 낮음 |
| 울트라 | gpt-5 | gpt-5 | 높음 |
| 프로 | claude-opus-4-5 | gpt-5.2 | 높음 |

---

## 4. SpringBoot 연동

### 4.1 ModalAIClient 서비스

```java
@Service
@Slf4j
public class ModalAIClient {

    @Value("${modal.base-url}")
    private String modalBaseUrl;

    private final RestTemplate restTemplate;

    public ModalAIClient(RestTemplateBuilder builder) {
        this.restTemplate = builder
            .setConnectTimeout(Duration.ofSeconds(60))   // Cold Start 대비
            .setReadTimeout(Duration.ofMinutes(10))      // GPU 작업 대비
            .build();
    }

    public HighlightResponse analyzeVideo(String videoUrl, int topK) {
        String url = modalBaseUrl + "/analyze";
        Map<String, Object> request = Map.of("video_url", videoUrl, "top_k", topK);
        return restTemplate.postForObject(url, request, HighlightResponse.class);
    }

    public DetectionResponse detectFish(List<String> imagesBase64) {
        String url = modalBaseUrl + "/detect-fish";
        Map<String, Object> request = Map.of("images", imagesBase64);
        return restTemplate.postForObject(url, request, DetectionResponse.class);
    }

    public SpeciesResponse identifySpecies(List<String> imagesBase64) {
        String url = modalBaseUrl + "/identify-species";
        Map<String, Object> request = Map.of("images", imagesBase64);
        return restTemplate.postForObject(url, request, SpeciesResponse.class);
    }
}
```

### 4.2 비동기 처리 서비스

```java
@Service
@Slf4j
public class VideoAnalysisService {

    private final ModalAIClient modalClient;
    private final RedisTemplate<String, Object> redisTemplate;

    @Async("aiTaskExecutor")
    public CompletableFuture<AnalysisResult> analyzeVideoAsync(
            String taskId, String videoUrl, String persona) {
        try {
            updateStatus(taskId, "processing");

            // Step 1: 하이라이트 추출 (Modal)
            HighlightResponse highlights = modalClient.analyzeVideo(videoUrl, 5);

            // Step 2: 물고기 탐지 (Modal)
            List<String> images = highlights.getImagesBase64();
            DetectionResponse detections = modalClient.detectFish(images);

            // Step 3: 종 식별 (Modal)
            List<String> crops = detections.getAllCrops();
            SpeciesResponse species = modalClient.identifySpecies(crops);

            // Step 4: LLM 로그 생성 (GMS API)
            String logbook = generateLogbook(highlights, species, persona);

            updateStatus(taskId, "completed");
            return CompletableFuture.completedFuture(buildResult(taskId, highlights, species, logbook));

        } catch (Exception e) {
            log.error("Analysis failed: {}", e.getMessage());
            updateStatus(taskId, "failed");
            throw new AnalysisException("Analysis failed", e);
        }
    }

    private void updateStatus(String taskId, String status) {
        redisTemplate.opsForValue().set("task:" + taskId + ":status", status);
    }
}
```

### 4.3 REST API Controller

```java
@RestController
@RequestMapping("/api/v1/analysis")
@RequiredArgsConstructor
public class VideoAnalysisController {

    private final VideoAnalysisService analysisService;

    @PostMapping("/video")
    public ResponseEntity<TaskResponse> analyzeVideo(@RequestBody VideoAnalysisRequest request) {
        String taskId = UUID.randomUUID().toString();
        analysisService.analyzeVideoAsync(taskId, request.getVideoUrl(), request.getPersona());
        return ResponseEntity.accepted().body(new TaskResponse(taskId, "processing"));
    }

    @GetMapping("/status/{taskId}")
    public ResponseEntity<StatusResponse> getStatus(@PathVariable String taskId) {
        String status = analysisService.getStatus(taskId);
        return ResponseEntity.ok(new StatusResponse(taskId, status));
    }

    @GetMapping("/result/{taskId}")
    public ResponseEntity<AnalysisResult> getResult(@PathVariable String taskId) {
        return analysisService.getResult(taskId)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }
}
```

### 4.4 통신 흐름

```
┌─────────┐     ┌─────────────┐     ┌─────────────────────────────────┐
│  Mobile │     │ SpringBoot  │     │         Modal Cloud             │
│   App   │     │   (:8080)   │     │   (Web Endpoints)               │
└────┬────┘     └──────┬──────┘     └────────────────┬────────────────┘
     │                 │                             │
     │ POST /api/v1/   │                             │
     │ analysis/video  │                             │
     │────────────────>│                             │
     │                 │                             │
     │ {taskId}        │  POST /analyze              │
     │<────────────────│────────────────────────────>│ (GPU)
     │                 │                             │
     │                 │  POST /detect-fish          │
     │                 │────────────────────────────>│ (GPU)
     │                 │                             │
     │                 │  POST /identify-species     │
     │                 │────────────────────────────>│ (GPU)
     │                 │                             │
     │ GET /status/    │                             │
     │ {taskId}        │                             │
     │────────────────>│                             │
     │ {status}        │                             │
     │<────────────────│                             │
     │                 │                             │
     │ GET /result/    │                             │
     │ {taskId}        │                             │
     │────────────────>│                             │
     │ {result}        │                             │
     │<────────────────│                             │
```

### 4.5 설정

**application.yml**:
```yaml
modal:
  base-url: ${MODAL_BASE_URL:https://your-workspace--divery-ai.modal.run}

spring:
  task:
    execution:
      pool:
        core-size: 5
        max-size: 10
        queue-capacity: 25
      thread-name-prefix: ai-task-
```

**docker-compose.yml**:
```yaml
backend:
  environment:
    MODAL_BASE_URL: ${MODAL_BASE_URL:-https://your-workspace--divery-ai.modal.run}
```

---

## 5. Modal 배포

### 5.1 디렉토리 구조

```
AI_pipeline/
├── AI_PIPELINE_DESIGN.md     # 본 문서
├── requirements.txt          # Python 의존성
│
├── modal_app/
│   ├── __init__.py
│   ├── app.py                # Modal App 정의
│   ├── config.py             # 설정
│   ├── analyze.py            # 하이라이트 추출
│   ├── detect.py             # CFD 물고기 탐지
│   ├── identify.py           # BioCLIP 종 식별
│   └── utils/
│       ├── video.py          # 영상 처리
│       ├── image.py          # 이미지 처리
│       └── wikidata.py       # 한국어명 조회
│
├── scripts/
│   └── upload_weights.py     # 가중치 업로드
│
└── tests/
    └── test_endpoints.py
```

### 5.2 Modal Volume (가중치 관리)

```python
# scripts/upload_weights.py
import modal

volume = modal.Volume.from_name("divery-weights", create_if_missing=True)

with volume.batch_upload() as batch:
    batch.put_file("./weights/yolov8l-world.pt", "/yolov8l-world.pt")
    batch.put_file("./weights/cfd-yolov12x.pt", "/cfd-yolov12x.pt")
    batch.put_file("./weights/nima_resnet50.pth", "/nima_resnet50.pth")

print("Weights uploaded!")
```

**가중치 목록**:
| 파일 | 크기 | 용도 |
|------|------|------|
| `yolov8l-world.pt` | ~500MB | YOLO-World Large |
| `cfd-yolov12x.pt` | ~200MB | CFD 물고기 탐지 |
| `nima_resnet50.pth` | ~100MB | NIMA 미적 평가 |

### 5.3 배포 명령

```bash
# 1. Modal CLI 설치 및 로그인
pip install modal
modal token new

# 2. 로컬 테스트
modal run modal_app/app.py

# 3. 배포
modal deploy modal_app/app.py

# 4. 가중치 업로드
python scripts/upload_weights.py
```

---

## 6. 비용 및 소요 시간

### 6.1 GPU 단가 (2025-03 기준 추정치, 변동 가능)

| GPU | 시간당 | 초당 |
|-----|--------|------|
| T4 | $0.59 | $0.000164 |
| A10G | $1.10 | $0.000306 |

### 6.2 작업별 비용

| 작업 | GPU | 예상 시간 | 비용 |
|------|-----|----------|------|
| 하이라이트 추출 | T4 | ~30초 | $0.0049 |
| CFD 탐지 | T4 | ~5초 | $0.0008 |
| 종 식별 | T4 | ~5초 | $0.0008 |
| **총 (영상 1건)** | | ~40초 | **~$0.007** |

### 6.3 월간 비용

| 사용량 | 비용 |
|--------|------|
| 100건 | ~$0.70 |
| 1,000건 | ~$7.00 |
| 5,000건 | ~$35.00 |

**무료 크레딧 $30/월 → ~4,000건 처리 가능**

### 6.4 예상 소요 시간 (1분 영상)

```
[Cold Start 발생시]          [Warm 상태]
─────────────────────────    ─────────────────────────
S3 → Modal 다운로드   5초     S3 → Modal 다운로드   5초
Cold Start          20초     (없음)                0초
GPU 처리            40초     GPU 처리            40초
결과 전송            2초     결과 전송            2초
─────────────────────────    ─────────────────────────
총                 ~67초     총                 ~47초
```

**비동기 처리로 사용자 체감 없음**

---

## 7. 에러 핸들링

### 7.1 Cold Start 대응

```
비동기 처리 + 상태 폴링
┌─────────────────────────────────────────────────────┐
│ 1. 앱에서 분석 요청                                  │
│ 2. SpringBoot: 즉시 taskId 반환                     │
│ 3. 백그라운드에서 Modal 호출 (Cold Start 포함)        │
│ 4. 앱에서 상태 폴링 or 푸시 알림                      │
│ 5. 완료시 결과 조회                                  │
└─────────────────────────────────────────────────────┘
```

### 7.2 Modal 재시도

```python
@app.function(
    gpu="T4",
    retries=3,      # 최대 3회 재시도
    timeout=600,    # 10분 타임아웃
)
def analyze(request: dict):
    ...
```

### 7.3 SpringBoot 예외 처리

```java
@Async("aiTaskExecutor")
public void analyzeVideoAsync(String taskId, String videoUrl) {
    try {
        updateStatus(taskId, "processing");
        // Modal 호출...
        updateStatus(taskId, "completed");
    } catch (ResourceAccessException e) {
        log.error("Modal timeout: {}", e.getMessage());
        updateStatus(taskId, "failed");
    } catch (Exception e) {
        log.error("Analysis failed: {}", e.getMessage());
        updateStatus(taskId, "failed");
    }
}
```

---

## 8. 기술 스택

| 카테고리 | 기술 |
|----------|------|
| GPU 인프라 | Modal (서버리스) |
| 객체 탐지 | YOLO-World (하이라이트), YOLOv12x CFD (물고기) |
| 미적 평가 | NIMA (ResNet50) |
| 종 식별 | BioCLIP (HuggingFace) |
| 한국어 매핑 | Wikidata SPARQL |
| LLM | GPT-5, Claude Opus 4.5, o3-mini (GMS API) |
| 백엔드 | SpringBoot |
| 저장소 | PostgreSQL, Redis, S3 |
| 이미지 처리 | OpenCV, PIL |

---

## 9. 구현 TODO

### Phase 1: Modal 환경 구축
- [ ] Modal 계정 생성 및 CLI 설정
- [ ] 기본 함수 테스트 (Hello World GPU)
- [ ] Volume 생성 및 가중치 업로드

### Phase 2: Modal 함수 구현
- [ ] `/analyze` 엔드포인트 구현
- [ ] `/detect-fish` 엔드포인트 구현
- [ ] `/identify-species` 엔드포인트 구현
- [ ] 각 엔드포인트 개별 테스트

### Phase 3: SpringBoot 연동
- [ ] `ModalAIClient` 서비스 클래스 구현
- [ ] `VideoAnalysisService` 비동기 처리 구현
- [ ] REST API 컨트롤러 구현
- [ ] Redis 상태 관리 연동

### Phase 4: 통합 테스트
- [ ] Modal ↔ SpringBoot 호출 테스트
- [ ] 전체 파이프라인 E2E 테스트
- [ ] 에러 핸들링 검증

### Phase 5: 배포
- [ ] Modal 프로덕션 배포
- [ ] docker-compose 환경변수 추가
- [ ] CI/CD 파이프라인 연동

---

## 10. 참고 자료

- Modal 공식 문서: https://modal.com/docs
- Modal GPU 가이드: https://modal.com/docs/guide/gpu
- YOLO-World: https://github.com/AILab-CVC/YOLO-World
- BioCLIP: https://huggingface.co/spaces/imageomics/bioclip-demo
- NIMA: https://github.com/idealo/image-quality-assessment
- Wikidata SPARQL: https://query.wikidata.org/

---

*문서 작성일: 2025-03-11*
*작성자: AI Pipeline 통합 담당*

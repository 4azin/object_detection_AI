# Divery GPU Worker

SQS 기반 GPU 추론 워커입니다. JupyterHub 등 GPU 서버에서 실행합니다.

## 역할

1. SQS에서 작업 메시지 수신
2. S3/URL에서 영상 또는 프레임 세트 다운로드
3. (프레임 세션) 하이라이트 선별 -> 하이라이트별 CFD 탐지/종분류
4. (영상 모드) CFD 파이프라인 실행 (물고기 탐지 + 종 분류)
5. 결과 JSON을 S3 업로드
6. BE 콜백 호출 (`/processing`, `/complete`, `/fail`)

## 빠른 시작 (Conda)

```bash
# 1. conda 환경 활성화 (torch 설치 필요)
conda activate divery

# 2. 셋업
cd AI/worker
bash scripts/setup.sh

# 3. 환경변수 설정
cp .env.example .env
vi .env

# 4. 연결 테스트
bash scripts/test.sh

# 5. 워커 실행
bash scripts/run.sh        # 포그라운드
bash scripts/run.sh --bg   # 백그라운드

# 6. 상태/중지
bash scripts/status.sh
bash scripts/stop.sh
```

## 스크립트

| 스크립트 | 설명 |
|----------|------|
| `scripts/setup.sh` | 의존성 설치 및 환경 검증 |
| `scripts/test.sh` | AWS/SQS/BE 연결 테스트 |
| `scripts/run.sh` | 워커 실행 (`--bg` 옵션으로 백그라운드) |
| `scripts/stop.sh` | 워커 중지 |
| `scripts/status.sh` | 상태 및 로그 확인 |
| `scripts/package.sh` | 배포용 tar.gz 번들 생성 |
| `scripts/git-sync.sh` | git 기반 코드 동기화 + (옵션) 워커 재시작 |

## Git 전환/업데이트

```bash
# 최초 1회 (tar 배포 환경 -> git 기반 전환)
cd ~/AI/worker
bash scripts/git-sync.sh --repo-url <REPO_URL> --branch develop

# 이후 업데이트
cd ~/AI/worker
bash scripts/git-sync.sh --branch develop
```

- `.env`, `worker.log`, `worker.pid`, `weights/`, `models/`는 보존됩니다.
- 기본 동작은 동기화 후 워커를 재시작합니다.
- 재시작 없이 코드만 갱신하려면 `--no-restart`를 사용하세요.

## 환경변수

### 필수

| 변수 | 설명 |
|------|------|
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 |
| `S3_BUCKET_NAME` | S3 버킷명 |
| `SQS_QUEUE_URL` | SQS 큐 URL |
| `BACKEND_CALLBACK_BASE_URL` | BE 콜백 URL (예: `https://j14a402.p.ssafy.io/stg`) |
| `AI_CALLBACK_SIGNING_SECRET` | HMAC 서명 시크릿 (BE와 동일) |

### 선택

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AWS_REGION` | `ap-northeast-2` | AWS 리전 |
| `CLOUDFRONT_DOMAIN` | `` | CloudFront 호스트(예: `https://...cloudfront.net`)를 S3 버킷 fallback으로 해석 |
| `CLOUDFRONT_DOMAINS` | `` | 추가 CloudFront 호스트 목록(콤마 구분) |
| `GMS_KEY` | `` | GMS 토큰 (로그북 LLM 생성용) |
| `LOGBOOK_LLM_ENABLED` | `auto` | `auto`(GMS_KEY 있으면 활성화), `true/false` 강제 |
| `LOGBOOK_LLM_URL` | `https://gms.ssafy.io/gmsapi/api.openai.com/v1/chat/completions` | 로그북 생성 API URL |
| `LOGBOOK_LLM_MODEL` | `gpt-5-mini` | 로그북 생성 모델 |
| `LOGBOOK_LLM_TIMEOUT_SECONDS` | `20` | 로그북 생성 요청 타임아웃 |
| `LOGBOOK_LLM_RETRY_COUNT` | `2` | 로그북 생성 재시도 횟수 |
| `LOGBOOK_LLM_TEMPERATURE` | `0.4` | 로그북 생성 온도 |
| `LOGBOOK_LLM_MAX_TOKENS` | `300` | 로그북 생성 최대 토큰 |
| `LOGBOOK_LLM_MAX_CHARS` | `320` | 최종 로그북 최대 길이(문자) |
| `WORKER_DUMMY_MODE` | `false` | 더미 모드 (추론 생략) |
| `CFD_MODEL` | `bioclip2` | 분류 모델 |
| `CFD_MODE` | `open-domain` | 분류 모드 |
| `CFD_CONF_THRESHOLD` | `0.5` | 탐지 신뢰도 임계값 |
| `HIGHLIGHT_TOP_K` | `5` | 프레임 세션 하이라이트 최대 개수 |
| `HIGHLIGHT_MIN_GAP_SECONDS` | `2` | 하이라이트 간 최소 시간 간격 |
| `HIGHLIGHT_MODEL_WEIGHTS` | `../CFD_fishial/weights/yolov8l-world.pt` | 하이라이트 선별 모델 경로 |
| `HIGHLIGHT_MODEL_CONF` | `0.1` | 하이라이트 선별 탐지 임계값 |
| `PIPELINE_TIMEOUT_SECONDS` | `7200` | 파이프라인 타임아웃 |
| `SQS_VISIBILITY_TIMEOUT` | `900` | 메시지 가시성 타임아웃 |
| `WORKER_KEEP_WORKDIR` | `false` | 작업 디렉토리 유지 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

## 참고

- 실패 메시지는 삭제하지 않아 SQS 재시도/DLQ로 처리됨
- `WORKER_DUMMY_MODE=true`로 추론 없이 플로우 테스트 가능
- 장시간 추론 중 visibility heartbeat로 메시지 재전달 방지

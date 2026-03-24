#!/usr/bin/env bash
#
# Divery GPU Worker 번들 생성
# 사용법: bash scripts/package.sh [출력디렉토리]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_DIR="$(cd "$WORKER_DIR/.." && pwd)"
OUT_DIR="${1:-$AI_DIR/dist}"
TS="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="divery-gpu-worker-${TS}.tar.gz"

REQUIRED_MODEL="$AI_DIR/CFD_fishial/cfd-yolov12x-1.00.pt"

mkdir -p "$OUT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "========================================"
echo " Divery GPU Worker Bundle"
echo "========================================"
echo

# 필수 모델 확인
echo "[1/3] 모델 가중치 확인"
if [[ ! -f "$REQUIRED_MODEL" ]]; then
    echo "[error] 필수 모델 없음: $REQUIRED_MODEL"
    exit 1
fi
echo "[ok] $REQUIRED_MODEL"
echo

# 파일 복사
echo "[2/3] 파일 복사"
mkdir -p "$TMP_DIR/AI"

# worker
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'worker.pid' \
    --exclude 'worker.log' \
    "$WORKER_DIR/" "$TMP_DIR/AI/worker/"

# CFD_fishial
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results' \
    --exclude 'test_videos' \
    "$AI_DIR/CFD_fishial/" "$TMP_DIR/AI/CFD_fishial/"

# README
cp "$AI_DIR/README.md" "$TMP_DIR/AI/" 2>/dev/null || true

# DEPLOY.md 생성
cat > "$TMP_DIR/AI/DEPLOY.md" <<'DOC'
# Divery GPU Worker 배포 가이드

## 1. 압축 해제

```bash
tar -xzf divery-gpu-worker-*.tar.gz
cd AI/worker
```

## 2. Conda 환경 준비

```bash
# conda 환경 활성화 (torch 설치 필요)
conda activate divery

# 또는 새 환경 생성
conda create -n divery python=3.11
conda activate divery
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 3. 셋업

```bash
bash scripts/setup.sh
```

## 4. 환경 설정

```bash
cp .env.example .env
vi .env
```

필수 값:
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `SQS_QUEUE_URL`
- `BACKEND_CALLBACK_BASE_URL`
- `AI_CALLBACK_SIGNING_SECRET`

## 5. 테스트 & 실행

```bash
# 연결 테스트
bash scripts/test.sh

# 워커 실행 (포그라운드)
bash scripts/run.sh

# 또는 백그라운드
bash scripts/run.sh --bg

# 상태 확인
bash scripts/status.sh

# 중지
bash scripts/stop.sh
```
DOC
echo

# 압축
echo "[3/3] 압축 생성"
( cd "$TMP_DIR" && tar -czf "$OUT_DIR/$BUNDLE_NAME" AI )

BUNDLE_SIZE=$(ls -lh "$OUT_DIR/$BUNDLE_NAME" | awk '{print $5}')
echo
echo "========================================"
echo "[done] $OUT_DIR/$BUNDLE_NAME"
echo "[size] $BUNDLE_SIZE"
echo "========================================"

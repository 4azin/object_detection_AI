#!/usr/bin/env bash
#
# Divery GPU Worker 번들 생성 (모델 가중치 제외 - 경량 버전)
# 이미 가중치가 있는 서버에 코드만 업데이트할 때 사용
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_DIR="$(cd "$WORKER_DIR/.." && pwd)"
OUT_DIR="${1:-$AI_DIR/dist}"
TS="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="divery-gpu-worker-lite-${TS}.tar.gz"

mkdir -p "$OUT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "========================================"
echo " Divery GPU Worker Bundle (Lite)"
echo "========================================"
echo

echo "[1/2] 파일 복사 (가중치 제외)"
mkdir -p "$TMP_DIR/AI"

# worker
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'worker.pid' \
    --exclude 'worker.log' \
    "$WORKER_DIR/" "$TMP_DIR/AI/worker/"

# CFD_fishial (가중치 제외)
rsync -a --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'results' \
    --exclude 'test_videos' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.ckpt' \
    --exclude 'fishial_model' \
    --exclude 'weights' \
    "$AI_DIR/CFD_fishial/" "$TMP_DIR/AI/CFD_fishial/"

# README
cp "$AI_DIR/README.md" "$TMP_DIR/AI/" 2>/dev/null || true
echo

# 압축
echo "[2/2] 압축 생성"
( cd "$TMP_DIR" && tar -czf "$OUT_DIR/$BUNDLE_NAME" AI )

BUNDLE_SIZE=$(ls -lh "$OUT_DIR/$BUNDLE_NAME" | awk '{print $5}')
echo
echo "========================================"
echo "[done] $OUT_DIR/$BUNDLE_NAME"
echo "[size] $BUNDLE_SIZE"
echo ""
echo "주의: 이 번들은 모델 가중치가 없습니다."
echo "가중치 파일을 별도로 복사하세요:"
echo "  - CFD_fishial/cfd-yolov12x-1.00.pt (필수)"
echo "========================================"

#!/usr/bin/env bash
#
# Divery GPU Worker bundle builder for Windows bash environments
# - intended for Git Bash / WSL
# - avoids rsync and uses only common commands: find, cp, tar, mktemp
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

require_cmd() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "[error] required command not found: $name"
        exit 1
    fi
}

copy_filtered_files() {
    local src="$1"
    local dest="$2"
    local mode="$3"

    mkdir -p "$dest"

    (
        cd "$src"

        if [[ "$mode" == "worker" ]]; then
            find . \
                \( -name '__pycache__' \) -prune -o \
                -type f \
                ! -name '*.pyc' \
                ! -name '.env' \
                ! -name 'worker.pid' \
                ! -name 'worker.log' \
                -print
        else
            find . \
                \( -name '__pycache__' -o -name 'results' -o -name 'test_videos' \) -prune -o \
                -type f \
                ! -name '*.pyc' \
                -print
        fi
    ) | while IFS= read -r rel_path; do
        local trimmed="${rel_path#./}"
        local target="$dest/$trimmed"
        mkdir -p "$(dirname "$target")"
        cp "$src/$trimmed" "$target"
    done
}

require_cmd find
require_cmd cp
require_cmd tar
require_cmd mktemp

echo "========================================"
echo " Divery GPU Worker Bundle (Windows Bash)"
echo "========================================"
echo

echo "[1/3] checking required model"
if [[ ! -f "$REQUIRED_MODEL" ]]; then
    echo "[error] required model not found: $REQUIRED_MODEL"
    exit 1
fi
echo "[ok] $REQUIRED_MODEL"
echo

echo "[2/3] copying files"
mkdir -p "$TMP_DIR/AI"

copy_filtered_files "$WORKER_DIR" "$TMP_DIR/AI/worker" "worker"
copy_filtered_files "$AI_DIR/CFD_fishial" "$TMP_DIR/AI/CFD_fishial" "cfd"

cp "$AI_DIR/README.md" "$TMP_DIR/AI/" 2>/dev/null || true

cat > "$TMP_DIR/AI/DEPLOY.md" <<'DOC'
# Divery GPU Worker Deploy Guide

## 1. Extract

```bash
tar -xzf divery-gpu-worker-*.tar.gz
cd AI/worker
```

## 2. Setup

```bash
bash scripts/setup.sh
```

## 3. Configure

```bash
cp .env.example .env
vi .env
```

Required:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `SQS_QUEUE_URL`
- `BACKEND_CALLBACK_BASE_URL`
- `AI_CALLBACK_SIGNING_SECRET`

## 4. Run

```bash
bash scripts/test.sh
bash scripts/run.sh --bg
bash scripts/status.sh
```
DOC
echo

echo "[3/3] creating archive"
( cd "$TMP_DIR" && tar -czf "$OUT_DIR/$BUNDLE_NAME" AI )

if command -v du >/dev/null 2>&1; then
    BUNDLE_SIZE="$(du -h "$OUT_DIR/$BUNDLE_NAME" | awk '{print $1}')"
else
    BUNDLE_SIZE="$(ls -lh "$OUT_DIR/$BUNDLE_NAME" | awk '{print $5}')"
fi

echo
echo "========================================"
echo "[done] $OUT_DIR/$BUNDLE_NAME"
echo "[size] $BUNDLE_SIZE"
echo "========================================"

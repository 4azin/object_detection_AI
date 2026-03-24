#!/usr/bin/env bash
#
# Divery GPU Worker Smoke Test
# AWS 연결, S3, SQS, BE 콜백 연결 테스트
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$WORKER_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[error] .env 파일 없음: $ENV_FILE"
    echo "cp $WORKER_DIR/.env.example $WORKER_DIR/.env"
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "========================================"
echo " Divery GPU Worker Smoke Test"
echo "========================================"
echo

python - <<'PY'
import os
import sys

try:
    import boto3
    import requests
    import torch
except ImportError as e:
    print(f"[error] 의존성 누락: {e}")
    print("bash scripts/setup.sh 를 먼저 실행하세요.")
    sys.exit(1)

print("[gpu]", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[gpu name]", torch.cuda.get_device_name(0))

region = os.environ.get("AWS_REGION", "ap-northeast-2")
queue_url = os.environ.get("SQS_QUEUE_URL", "")
bucket = os.environ.get("S3_BUCKET_NAME", "")
base_url = os.environ.get("BACKEND_CALLBACK_BASE_URL", "").rstrip("/")

if not queue_url:
    print("[error] SQS_QUEUE_URL 환경변수가 없습니다.")
    sys.exit(1)
if not bucket:
    print("[error] S3_BUCKET_NAME 환경변수가 없습니다.")
    sys.exit(1)

# AWS STS
try:
    sts = boto3.client("sts", region_name=region)
    identity = sts.get_caller_identity()
    print("[aws sts]", identity["Arn"])
except Exception as e:
    print(f"[aws sts] FAIL: {e}")
    sys.exit(1)

# S3
try:
    s3 = boto3.client("s3", region_name=region)
    s3.head_bucket(Bucket=bucket)
    print("[s3]", bucket, "OK")
except Exception as e:
    print(f"[s3] FAIL: {e}")
    sys.exit(1)

# SQS
try:
    sqs = boto3.client("sqs", region_name=region)
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
    print("[sqs]", attrs["Attributes"]["QueueArn"], "OK")
except Exception as e:
    print(f"[sqs] FAIL: {e}")
    sys.exit(1)

# BE 연결 (선택적)
if base_url:
    for path in ["/health", "/api/v1/analysis/status/smoke-test"]:
        url = base_url + path
        try:
            r = requests.get(url, timeout=10)
            print(f"[http] {url} -> {r.status_code}")
        except Exception as e:
            print(f"[http] {url} -> FAIL: {str(e)[:50]}")
else:
    print("[http] BACKEND_CALLBACK_BASE_URL 미설정, 건너뜀")

print()
print("[done] 스모크 테스트 완료")
PY

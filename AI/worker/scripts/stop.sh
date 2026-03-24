#!/usr/bin/env bash
#
# Divery GPU Worker 중지
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$WORKER_DIR/worker.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "[info] PID 파일 없음 (실행 중이 아님)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" &>/dev/null; then
    echo "[info] 워커 중지 (PID: $PID)"
    kill "$PID"

    # 종료 대기 (최대 10초)
    for i in {1..10}; do
        if ! ps -p "$PID" &>/dev/null; then
            break
        fi
        sleep 1
    done

    # 강제 종료
    if ps -p "$PID" &>/dev/null; then
        echo "[warn] 강제 종료"
        kill -9 "$PID" 2>/dev/null || true
    fi

    echo "[stopped]"
else
    echo "[info] 프로세스 이미 종료됨"
fi

rm -f "$PID_FILE"

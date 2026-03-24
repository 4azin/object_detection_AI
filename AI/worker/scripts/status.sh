#!/usr/bin/env bash
#
# Divery GPU Worker 상태 확인
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$WORKER_DIR/worker.pid"
LOG_FILE="$WORKER_DIR/worker.log"

echo "========================================"
echo " Divery GPU Worker Status"
echo "========================================"

# PID 확인
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" &>/dev/null; then
        echo "[status] 실행 중 (PID: $PID)"
        ps -p "$PID" -o pid,ppid,user,%cpu,%mem,etime,cmd --no-headers
    else
        echo "[status] 중지됨 (stale PID 파일)"
    fi
else
    echo "[status] 중지됨"
fi

echo

# 로그 파일
if [[ -f "$LOG_FILE" ]]; then
    echo "[log] 최근 10줄:"
    echo "----------------------------------------"
    tail -10 "$LOG_FILE"
    echo "----------------------------------------"
    echo ""
    echo "전체 로그: tail -f $LOG_FILE"
else
    echo "[log] 로그 파일 없음"
fi

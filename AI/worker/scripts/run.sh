#!/usr/bin/env bash
#
# Divery GPU Worker 실행
#
# 포그라운드: bash scripts/run.sh
# 백그라운드: bash scripts/run.sh --bg
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$WORKER_DIR/.env"
PID_FILE="$WORKER_DIR/worker.pid"
LOG_FILE="$WORKER_DIR/worker.log"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[error] .env 파일 없음"
    echo "cp $WORKER_DIR/.env.example $WORKER_DIR/.env"
    exit 1
fi

# 이미 실행 중인지 확인
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" &>/dev/null; then
        echo "[info] 워커 이미 실행 중 (PID: $OLD_PID)"
        echo "중지하려면: bash scripts/stop.sh"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# 환경변수 로드
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# rembg 미설치 시 배경 제거 비활성화
export CFD_REMOVE_BG="${CFD_REMOVE_BG:-false}"

cd "$WORKER_DIR"

if [[ "${1:-}" == "--bg" || "${1:-}" == "-d" ]]; then
    # 백그라운드 실행
    nohup python worker.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "[started] PID=$(cat "$PID_FILE")"
    echo "[log] $LOG_FILE"
    echo ""
    echo "로그 확인: tail -f $LOG_FILE"
    echo "중지: bash scripts/stop.sh"
else
    # 포그라운드 실행
    echo "[info] 워커 시작 (Ctrl+C로 중지)"
    exec python worker.py
fi

#!/usr/bin/env bash
#
# GPU 서버 워커를 git 기반으로 전환/업데이트
# - repo clone/fetch + develop checkout
# - AI/worker 소스만 대상 디렉토리에 동기화
# - .env / 로그 / pid 보존
# - 필요 시 워커 재시작
#
set -euo pipefail

print_usage() {
    cat <<'USAGE'
Usage:
  bash scripts/git-sync.sh [options]

Options:
  --repo-url <url>       Git 원격 URL (repo cache가 없을 때 필수)
  --branch <name>        동기화할 브랜치 (기본: develop)
  --repo-dir <path>      repo cache 경로 (기본: $HOME/AI/divery-repo)
  --worker-dir <path>    워커 배포 경로 (기본: $HOME/AI/worker)
  --no-restart           동기화 후 워커 재시작 생략
  -h, --help             도움말

Examples:
  bash scripts/git-sync.sh --repo-url git@github.com:org/repo.git
  bash scripts/git-sync.sh --branch develop
USAGE
}

require_cmd() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "[error] required command not found: $name"
        exit 1
    fi
}

REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-develop}"
REPO_DIR="${REPO_DIR:-$HOME/AI/divery-repo}"
WORKER_DIR="${WORKER_DIR:-$HOME/AI/worker}"
RESTART=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url)
            REPO_URL="${2:-}"
            shift 2
            ;;
        --branch)
            BRANCH="${2:-}"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="${2:-}"
            shift 2
            ;;
        --worker-dir)
            WORKER_DIR="${2:-}"
            shift 2
            ;;
        --no-restart)
            RESTART=0
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "[error] unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

require_cmd git
require_cmd rsync
require_cmd bash

echo "[info] repo_dir=$REPO_DIR"
echo "[info] worker_dir=$WORKER_DIR"
echo "[info] branch=$BRANCH"

mkdir -p "$(dirname "$REPO_DIR")" "$WORKER_DIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        echo "[error] repo cache가 없어서 --repo-url 이 필요합니다."
        exit 1
    fi
    echo "[info] clone repository"
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[info] fetch + checkout"
git -C "$REPO_DIR" fetch origin "$BRANCH" --prune
git -C "$REPO_DIR" checkout "$BRANCH"
git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"

SOURCE_WORKER_DIR="$REPO_DIR/AI/worker"
if [[ ! -f "$SOURCE_WORKER_DIR/worker.py" ]]; then
    echo "[error] source worker.py not found: $SOURCE_WORKER_DIR/worker.py"
    exit 1
fi

TMP_ENV=""
if [[ -f "$WORKER_DIR/.env" ]]; then
    TMP_ENV="$(mktemp)"
    cp "$WORKER_DIR/.env" "$TMP_ENV"
    echo "[info] existing .env backup: $TMP_ENV"
fi

echo "[info] sync worker files"
rsync -a --delete \
    --exclude ".env" \
    --exclude "worker.log" \
    --exclude "worker.pid" \
    --exclude "weights/" \
    --exclude "models/" \
    --exclude "__pycache__/" \
    "$SOURCE_WORKER_DIR/" "$WORKER_DIR/"

if [[ -n "$TMP_ENV" ]]; then
    cp "$TMP_ENV" "$WORKER_DIR/.env"
    rm -f "$TMP_ENV"
fi

chmod +x "$WORKER_DIR"/scripts/*.sh

GIT_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "[done] synced commit=$GIT_SHA"

if [[ "$RESTART" -eq 1 ]]; then
    echo "[info] restart worker"
    if [[ -x "$WORKER_DIR/scripts/stop.sh" ]]; then
        (cd "$WORKER_DIR" && bash scripts/stop.sh) || true
    fi
    (cd "$WORKER_DIR" && bash scripts/run.sh --bg)
    echo "[info] status"
    (cd "$WORKER_DIR" && bash scripts/status.sh) || true
else
    echo "[info] restart skipped (--no-restart)"
fi

echo "[ok] git sync completed"

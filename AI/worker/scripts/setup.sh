#!/usr/bin/env bash
#
# Divery GPU Worker Setup (Conda)
# 사용법: bash scripts/setup.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_DIR="$(cd "$WORKER_DIR/.." && pwd)"
CFD_DIR="$AI_DIR/CFD_fishial"

echo "========================================"
echo " Divery GPU Worker Setup"
echo "========================================"
echo

# 1. Conda 확인
echo "[1/5] Conda 환경 확인"
if ! command -v conda &>/dev/null; then
    echo "[error] conda not found. conda를 먼저 설치하세요."
    exit 1
fi

# 현재 conda 환경 확인
CONDA_ENV="${CONDA_DEFAULT_ENV:-base}"
echo "[info] 현재 conda 환경: $CONDA_ENV"

if [[ "$CONDA_ENV" == "base" ]]; then
    echo "[warn] base 환경입니다. 전용 환경 사용을 권장합니다:"
    echo "       conda create -n divery python=3.11 && conda activate divery"
fi
echo

# 2. Python & torch 확인
echo "[2/5] Python & PyTorch 확인"
python --version

if ! python -c "import torch" 2>/dev/null; then
    echo "[error] torch가 설치되어 있지 않습니다."
    echo ""
    echo "다음 명령으로 설치하세요:"
    echo "  conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia"
    echo "  또는"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    exit 1
fi

python -c "import torch; print(f'[torch] {torch.__version__}')"
python -c "import torch; print(f'[cuda] {torch.cuda.is_available()}')"

if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[warn] CUDA 사용 불가 - CPU 모드로 실행됩니다 (매우 느림)"
fi
echo

# 3. 의존성 설치
echo "[3/5] 의존성 설치"
pip install --quiet --upgrade pip

# requirements.txt 기반 의존성 설치
if [[ -f "$WORKER_DIR/requirements.txt" ]]; then
    echo "[info] Installing dependencies from requirements.txt"
    pip install -r "$WORKER_DIR/requirements.txt"
else
    echo "[error] requirements.txt not found: $WORKER_DIR/requirements.txt"
    exit 1
fi
echo

# 4. 모델 가중치 확인
echo "[4/5] 모델 가중치 확인"
REQUIRED_MODEL="$CFD_DIR/cfd-yolov12x-1.00.pt"

if [[ ! -f "$REQUIRED_MODEL" ]]; then
    echo "[error] 필수 모델 없음: $REQUIRED_MODEL"
    echo ""
    echo "가중치 파일을 복사하세요:"
    echo "  bash scripts/import_model_weights.sh /path/to/weights.zip"
    exit 1
fi

echo "[ok] $REQUIRED_MODEL"
ls -lh "$CFD_DIR"/*.pt 2>/dev/null || true
echo

# 5. 코드 검증
echo "[5/5] 코드 검증"
python -m py_compile "$WORKER_DIR/worker.py"
python -m py_compile "$CFD_DIR/integration.py"

python - <<'PY'
import torch
print('[gpu]', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[gpu name]', torch.cuda.get_device_name(0))
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'[gpu memory] {mem_gb:.1f} GB')
PY
echo

# 완료
echo "========================================"
if [[ -f "$WORKER_DIR/.env" ]]; then
    echo "[ok] .env 파일 존재"
else
    echo "[warn] .env 파일 없음"
    echo ""
    echo "다음 명령으로 생성하세요:"
    echo "  cp $WORKER_DIR/.env.example $WORKER_DIR/.env"
    echo "  vi $WORKER_DIR/.env"
fi
echo
echo "[done] 셋업 완료"
echo ""
echo "다음 단계:"
echo "  1. .env 파일 설정"
echo "  2. bash scripts/test.sh    # 연결 테스트"
echo "  3. bash scripts/run.sh     # 워커 실행"
echo "========================================"

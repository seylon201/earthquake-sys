#!/bin/bash

# 지진 조기경보 시스템 간단 배포 스크립트 (Python 가상환경)
# Docker 없이 직접 Python 환경에서 실행

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

VENV_NAME="earthquake_env"
PORT=${1:-8501}

log_info "지진 조기경보 시스템 간단 배포 시작..."

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    log_error "Python3가 설치되지 않았습니다."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_info "Python 버전: $PYTHON_VERSION"

# 가상환경 생성 또는 활성화
if [[ ! -d "$VENV_NAME" ]]; then
    log_info "가상환경 생성 중..."
    python3 -m venv $VENV_NAME
fi

log_info "가상환경 활성화..."
source $VENV_NAME/bin/activate

# 패키지 업그레이드
log_info "pip 업그레이드..."
pip install --upgrade pip

# 의존성 설치
if [[ -f "requirements.txt" ]]; then
    log_info "의존성 설치 중..."
    pip install -r requirements.txt
else
    log_info "기본 패키지 설치 중..."
    pip install streamlit pandas numpy tensorflow plotly influxdb-client scikit-learn matplotlib seaborn
fi

# 필수 파일 확인
REQUIRED_FILES=("new2_streamlit_dashboard.py" "dashboard_utils.py" "retraining_manager.py" "chart_renderer.py")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        log_error "필수 파일이 없습니다: $file"
        exit 1
    fi
done

# 모델 파일 확인
MODEL_FILES=("new2_convlstm_3class_best.h5" "new2_memory_optimized_best.h5")
MODEL_FOUND=false
for model in "${MODEL_FILES[@]}"; do
    if [[ -f "$model" ]]; then
        log_success "모델 파일 발견: $model"
        MODEL_FOUND=true
        break
    fi
done

if [[ "$MODEL_FOUND" == false ]]; then
    log_warning "모델 파일을 찾을 수 없습니다."
fi

# 필요한 디렉토리 생성
mkdir -p influxLogs/{base,new2_analysis,alerts}
mkdir -p data
mkdir -p retrained_models

log_success "환경 설정 완료!"

# Streamlit 실행
log_info "대시보드 시작 중..."
log_success "접속 URL: http://localhost:$PORT"
log_info "종료하려면 Ctrl+C를 누르세요."

export PYTHONPATH=$PWD
streamlit run new2_streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0
#!/bin/bash

# 지진 조기경보 시스템 배포 스크립트
# 사용법: ./deploy.sh [dev|prod] [--rebuild]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 설정
ENVIRONMENT=${1:-dev}
REBUILD=${2:-""}
PROJECT_NAME="earthquake-system"
COMPOSE_FILE="docker-compose.yml"

log_info "지진 조기경보 시스템 배포 시작..."
log_info "환경: $ENVIRONMENT"

# 환경 검증
if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    log_error "잘못된 환경입니다. 'dev' 또는 'prod'를 사용하세요."
    exit 1
fi

# Docker 및 Docker Compose 확인
if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose가 설치되지 않았습니다."
    exit 1
fi

# 필수 파일 확인
REQUIRED_FILES=("Dockerfile" "docker-compose.yml" "requirements.txt" "new2_streamlit_dashboard.py")
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
    log_warning "모델 파일을 찾을 수 없습니다. 배포 후 모델을 업로드하세요."
fi

# 필요한 디렉토리 생성
mkdir -p influxLogs/{base,new2_analysis,alerts}
mkdir -p data
mkdir -p retrained_models

log_info "필요한 디렉토리 생성 완료"

# 기존 컨테이너 정리 (rebuild 옵션 시)
if [[ "$REBUILD" == "--rebuild" ]]; then
    log_info "기존 컨테이너 정리 중..."
    docker-compose down --volumes --remove-orphans || true
    docker system prune -f || true
fi

# 환경별 설정
if [[ "$ENVIRONMENT" == "prod" ]]; then
    log_info "운영 환경 배포 중..."
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}-prod"
    export STREAMLIT_PORT=80
else
    log_info "개발 환경 배포 중..."
    export COMPOSE_PROJECT_NAME="${PROJECT_NAME}-dev"
    export STREAMLIT_PORT=8501
fi

# Docker 이미지 빌드 및 실행
log_info "Docker 이미지 빌드 중..."
docker-compose build

log_info "컨테이너 실행 중..."
docker-compose up -d

# 헬스체크 대기
log_info "서비스 시작 대기 중..."
sleep 30

# 서비스 상태 확인
if docker-compose ps | grep -q "Up"; then
    log_success "배포 완료!"
    log_info "대시보드 접속: http://localhost:$STREAMLIT_PORT"
    
    # 서비스 정보 출력
    echo ""
    echo "=== 서비스 정보 ==="
    docker-compose ps
    
    echo ""
    echo "=== 유용한 명령어 ==="
    echo "로그 확인: docker-compose logs -f"
    echo "서비스 중지: docker-compose down"
    echo "서비스 재시작: docker-compose restart"
    echo "컨테이너 접속: docker-compose exec earthquake_dashboard bash"
    
else
    log_error "배포 실패! 로그를 확인하세요."
    docker-compose logs
    exit 1
fi
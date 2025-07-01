#!/bin/bash

# 지진 조기경보 시스템 원격 접속 가능 배포 스크립트

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

log_info "지진 조기경보 시스템 원격 접속 배포 시작..."

# 현재 IP 주소 확인
LOCAL_IP=$(hostname -I | awk '{print $1}')
EXTERNAL_IP=$(curl -s ifconfig.me 2>/dev/null || echo "확인 불가")

log_info "서버 IP 정보:"
echo "  로컬 IP: $LOCAL_IP"
echo "  외부 IP: $EXTERNAL_IP"

# Python 환경 설정
if [[ ! -d "$VENV_NAME" ]]; then
    log_info "가상환경 생성 중..."
    python3 -m venv $VENV_NAME
fi

log_info "가상환경 활성화..."
source $VENV_NAME/bin/activate

# 패키지 설치
if [[ -f "requirements.txt" ]]; then
    log_info "의존성 설치 중..."
    pip install -r requirements.txt
else
    log_info "기본 패키지 설치 중..."
    pip install streamlit pandas numpy tensorflow plotly influxdb-client scikit-learn matplotlib seaborn
fi

# 필요한 디렉토리 생성
mkdir -p influxLogs/{base,new2_analysis,alerts}
mkdir -p data retrained_models

# Streamlit 설정 파일 생성 (원격 접속 허용)
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
port = $PORT
address = "0.0.0.0"

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = $PORT
EOF

log_success "Streamlit 원격 접속 설정 완료!"

# 방화벽 포트 열기 안내
log_warning "방화벽 설정이 필요할 수 있습니다:"
echo "  Ubuntu/Debian: sudo ufw allow $PORT"
echo "  CentOS/RHEL: sudo firewall-cmd --permanent --add-port=$PORT/tcp && sudo firewall-cmd --reload"
echo "  Windows: Windows Defender 방화벽에서 포트 $PORT 허용"

# 대시보드 시작
log_success "환경 설정 완료!"
log_info "대시보드 시작 중..."

echo ""
echo "🌐 접속 방법:"
echo "  로컬 접속: http://localhost:$PORT"
echo "  같은 네트워크: http://$LOCAL_IP:$PORT"
if [[ "$EXTERNAL_IP" != "확인 불가" ]]; then
echo "  인터넷 접속: http://$EXTERNAL_IP:$PORT (포트포워딩 필요)"
fi
echo ""
log_warning "보안 주의: 원격 접속 시 방화벽과 인증 설정을 확인하세요!"
echo ""

export PYTHONPATH=$PWD
streamlit run new2_streamlit_dashboard.py
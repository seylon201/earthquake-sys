# 지진 조기경보 시스템 Docker 이미지
FROM python:3.9-slim

# 시스템 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 모델 파일 경로 확인
RUN ls -la *.h5 || echo "모델 파일이 없습니다. 배포 전 모델 파일을 확인하세요."

# 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit 실행
CMD ["streamlit", "run", "new2_streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
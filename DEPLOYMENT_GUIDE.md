# 🚀 지진 조기경보 시스템 배포 가이드

이 문서는 지진 조기경보 시스템을 다양한 환경에 배포하는 방법을 단계별로 설명합니다.

## 📋 목차

1. [배포 전 준비사항](#배포-전-준비사항)
2. [Docker 배포 (권장)](#docker-배포-권장)
3. [Python 가상환경 배포](#python-가상환경-배포)
4. [Windows 배포](#windows-배포)
5. [클라우드 배포](#클라우드-배포)
6. [문제해결](#문제해결)

---

## 🔧 배포 전 준비사항

### 1. 시스템 요구사항

**최소 사양:**
- CPU: 2코어 이상
- RAM: 4GB 이상 (8GB 권장)
- 디스크: 5GB 이상 여유공간
- OS: Linux, macOS, Windows 10/11

**필수 소프트웨어:**
- Python 3.8+ 
- Git (선택사항)

**권장 소프트웨어:**
- Docker & Docker Compose
- InfluxDB (외부 DB 사용 시)

### 2. 배포 전 검증

배포하기 전에 반드시 사전 검증을 실행하세요:

```bash
# 배포 전 검증 실행
python3 pre_deploy_check.py
```

이 스크립트는 다음을 확인합니다:
- ✅ Python 버전 호환성
- ✅ 필수 파일 존재
- ✅ Python 패키지 설치 상태
- ✅ 디렉토리 구조
- ✅ Docker 환경 (해당하는 경우)
- ✅ 포트 사용 가능성
- ✅ 디스크 공간

---

## 🐳 Docker 배포 (권장)

Docker를 사용한 배포가 가장 안정적이고 일관된 환경을 제공합니다.

### 1. Docker 설치 확인

```bash
# Docker 버전 확인
docker --version
docker-compose --version
```

### 2. 배포 실행

**개발 환경 배포:**
```bash
# 기본 배포
./deploy.sh dev

# 강제 재빌드 배포
./deploy.sh dev --rebuild
```

**운영 환경 배포:**
```bash
# 운영 환경 (포트 80)
./deploy.sh prod

# 강제 재빌드
./deploy.sh prod --rebuild
```

### 3. 서비스 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f

# 대시보드 접속
# 개발: http://localhost:8501
# 운영: http://localhost:80
```

### 4. 서비스 관리

```bash
# 서비스 중지
docker-compose down

# 서비스 재시작  
docker-compose restart

# 데이터 포함 완전 삭제
docker-compose down --volumes
```

---

## 🐍 Python 가상환경 배포

Docker 없이 직접 Python 환경에서 실행하는 방법입니다.

### Linux/macOS

```bash
# 배포 스크립트 실행 (포트 8501)
./deploy_simple.sh

# 사용자 정의 포트
./deploy_simple.sh 8080
```

### 수동 설정

```bash
# 1. 가상환경 생성
python3 -m venv earthquake_env

# 2. 가상환경 활성화
source earthquake_env/bin/activate  # Linux/macOS

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 필요한 디렉토리 생성
mkdir -p influxLogs/{base,new2_analysis,alerts}
mkdir -p data retrained_models

# 5. 대시보드 실행
export PYTHONPATH=$PWD
streamlit run new2_streamlit_dashboard.py --server.port=8501
```

---

## 🪟 Windows 배포

Windows 환경에서의 배포 방법입니다.

### 자동 배포

```cmd
REM 배포 스크립트 실행
deploy_windows.bat
```

### 수동 설정

```cmd
REM 1. 가상환경 생성
python -m venv earthquake_env

REM 2. 가상환경 활성화
earthquake_env\Scripts\activate.bat

REM 3. 패키지 설치
pip install -r requirements.txt

REM 4. 디렉토리 생성
mkdir influxLogs\base influxLogs\new2_analysis influxLogs\alerts
mkdir data retrained_models

REM 5. 환경변수 설정 및 실행
set PYTHONPATH=%CD%
streamlit run new2_streamlit_dashboard.py --server.port=8501
```

---

## ☁️ 클라우드 배포

### AWS EC2

```bash
# 1. EC2 인스턴스 생성 (Ubuntu 20.04+)
# 2. 보안 그룹에서 8501 포트 열기
# 3. 서버 접속 후 Docker 설치

# Docker 설치
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER

# 프로젝트 복사 및 배포
git clone <repository_url>
cd earthquake_project
./deploy.sh prod
```

### Google Cloud Platform

```bash
# 1. Compute Engine 인스턴스 생성
# 2. 방화벽 규칙에서 8501 포트 허용
# 3. SSH 접속 후 Docker 설치 및 배포 (AWS와 동일)
```

### Azure

```bash
# 1. Virtual Machine 생성 (Ubuntu)
# 2. 네트워크 보안 그룹에서 8501 포트 허용  
# 3. SSH 접속 후 Docker 설치 및 배포 (AWS와 동일)
```

---

## 🔧 문제해결

### 일반적인 문제

**1. 포트 이미 사용 중**
```bash
# 사용 중인 프로세스 확인
sudo netstat -tulpn | grep :8501

# 프로세스 종료
sudo kill <PID>

# 다른 포트 사용
streamlit run new2_streamlit_dashboard.py --server.port=8502
```

**2. 모델 파일 없음**
```bash
# 모델 파일 확인
ls -la *.h5

# 모델 파일이 없다면 모델 훈련 실행
python train_new2_convlstm.py
```

**3. 패키지 설치 오류**
```bash
# 캐시 클리어 후 재설치
pip cache purge
pip install --no-cache-dir -r requirements.txt

# 개별 패키지 설치
pip install streamlit pandas numpy tensorflow
```

**4. 메모리 부족**
```bash
# 메모리 최적화 모델 사용
# dashboard_utils.py에서 모델 경로 변경:
# MODEL_PATH = "new2_memory_optimized_best.h5"
```

### Docker 관련 문제

**1. Docker 권한 오류**
```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker
```

**2. 컨테이너 빌드 실패**
```bash
# 시스템 정리
docker system prune -f

# 강제 재빌드
docker-compose build --no-cache
```

**3. 볼륨 마운트 오류**
```bash
# 권한 확인
sudo chown -R $USER:$USER ./influxLogs ./data

# SELinux 비활성화 (CentOS/RHEL)
sudo setsebool -P container_manage_cgroup true
```

### 성능 최적화

**1. 메모리 사용량 줄이기**
```python
# config 설정에서 배치 크기 조정
BATCH_SIZE = 8  # 기본값: 16
```

**2. CPU 사용량 최적화**
```bash
# TensorFlow CPU 스레드 제한
export TF_NUM_INTEROP_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
```

**3. 캐시 설정**
```python
# Streamlit 캐시 TTL 조정
@st.cache_data(ttl=60)  # 60초 캐시
```

---

## 📞 지원

배포 과정에서 문제가 발생하면:

1. **로그 확인**: `docker-compose logs` 또는 터미널 출력
2. **사전 검증**: `python3 pre_deploy_check.py` 실행
3. **문제해결 가이드**: 위의 문제해결 섹션 참고
4. **이슈 리포트**: 상세한 오류 메시지와 환경 정보 포함

---

## 📈 배포 후 확인 사항

배포 완료 후 다음을 확인하세요:

- ✅ 대시보드 접속 가능 (http://localhost:8501)
- ✅ 실시간 이벤트 모니터링 작동
- ✅ 센서 파형 표시 정상
- ✅ NEW2 AI 분석 결과 표시
- ✅ 파일 다운로드 기능 작동
- ✅ 재학습 시스템 상태 확인

성공적인 배포를 위해 이 가이드를 단계별로 따라 진행하세요! 🚀
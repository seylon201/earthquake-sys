@echo off
REM 지진 조기경보 시스템 Windows 배포 스크립트

echo [INFO] 지진 조기경보 시스템 Windows 배포 시작...

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았습니다.
    echo https://www.python.org/downloads/ 에서 Python을 설치하세요.
    pause
    exit /b 1
)

echo [INFO] Python 설치 확인됨

REM 가상환경 생성
if not exist earthquake_env (
    echo [INFO] 가상환경 생성 중...
    python -m venv earthquake_env
)

REM 가상환경 활성화
echo [INFO] 가상환경 활성화...
call earthquake_env\Scripts\activate.bat

REM pip 업그레이드
echo [INFO] pip 업그레이드...
python -m pip install --upgrade pip

REM 의존성 설치
if exist requirements.txt (
    echo [INFO] 의존성 설치 중...
    pip install -r requirements.txt
) else (
    echo [INFO] 기본 패키지 설치 중...
    pip install streamlit pandas numpy tensorflow plotly influxdb-client scikit-learn matplotlib seaborn
)

REM 필수 파일 확인
if not exist new2_streamlit_dashboard.py (
    echo [ERROR] 대시보드 파일이 없습니다: new2_streamlit_dashboard.py
    pause
    exit /b 1
)

REM 필요한 디렉토리 생성
if not exist influxLogs mkdir influxLogs
if not exist influxLogs\base mkdir influxLogs\base
if not exist influxLogs\new2_analysis mkdir influxLogs\new2_analysis
if not exist influxLogs\alerts mkdir influxLogs\alerts
if not exist data mkdir data
if not exist retrained_models mkdir retrained_models

echo [SUCCESS] 환경 설정 완료!
echo [INFO] 대시보드 시작 중...
echo [INFO] 접속 URL: http://localhost:8501
echo [INFO] 종료하려면 Ctrl+C를 누르세요.

REM Streamlit 실행
set PYTHONPATH=%CD%
streamlit run new2_streamlit_dashboard.py --server.port=8501 --server.address=0.0.0.0

pause
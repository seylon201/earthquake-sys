# Streamlit Cloud 배포 가이드

## 🚀 Streamlit Cloud 배포 (무료)

Streamlit Cloud는 Streamlit 앱을 위한 전용 호스팅 서비스입니다.

### 1단계: GitHub 저장소 준비

```bash
# 프로젝트를 GitHub에 업로드
git init
git add .
git commit -m "지진 조기경보 시스템 초기 버전"
git branch -M main
git remote add origin https://github.com/[username]/earthquake-system.git
git push -u origin main
```

### 2단계: Streamlit Cloud 앱 생성

1. https://share.streamlit.io 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소 선택: `earthquake-system`
5. 메인 파일: `new2_streamlit_dashboard.py`
6. "Deploy!" 클릭

### 3단계: 환경 설정

`secrets.toml` 파일 생성 (Streamlit Cloud 대시보드에서):

```toml
# InfluxDB 설정
[influxdb]
host = "your-influxdb-host"
port = 8086
database = "earthquake"
username = "admin"
password = "admin123"

# 기타 설정
[app]
timezone = "Asia/Seoul"
```

### 장점:
- ✅ 완전 무료
- ✅ GitHub 연동 자동 배포
- ✅ HTTPS 기본 제공
- ✅ 도메인 자동 할당
- ✅ Streamlit 최적화

### 단점:
- ❌ InfluxDB 별도 호스팅 필요
- ❌ 파일 업로드 제한
- ❌ 컴퓨팅 리소스 제한
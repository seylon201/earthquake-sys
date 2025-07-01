# GitHub 간단 푸시 가이드

## 방법 1: 브라우저에서 직접 업로드 (가장 쉬움)

### 1단계: 압축 파일 생성
터미널에서 실행:
```bash
# 필요한 파일들만 압축
tar -czf earthquake-system.tar.gz \
  new2_streamlit_dashboard.py \
  dashboard_utils.py \
  retraining_manager.py \
  chart_renderer.py \
  requirements.txt \
  README.md \
  .gitignore \
  packages.txt \
  Procfile \
  runtime.txt \
  DEPLOYMENT_GUIDE.md \
  streamlit_cloud_setup.md
```

### 2단계: GitHub에서 업로드
1. https://github.com/seylon201/earthquake-sys 접속
2. "Upload files" 버튼 클릭
3. 파일들을 드래그&드롭으로 업로드
4. Commit message 입력: "🚀 지진 조기경보 시스템 초기 배포"
5. "Commit changes" 클릭

## 방법 2: Personal Access Token (정석)

### 1단계: 개인 설정으로 이동
- 우측 상단 프로필 사진 클릭 → "Settings"
- 또는 직접 URL: https://github.com/settings/profile

### 2단계: Developer settings 찾기
- 좌측 메뉴 맨 아래 "Developer settings" 클릭

### 3단계: Token 생성
- Personal access tokens → Tokens (classic)
- Generate new token (classic)
- Scopes: repo 체크
- Generate token
- 토큰 복사!

### 4단계: Git 설정
```bash
git remote set-url origin https://[토큰]@github.com/seylon201/earthquake-sys.git
git push -u origin main
```

## 방법 3: Git Credential Manager (Windows)

Windows에서는 Git Credential Manager가 자동으로 처리해줄 수 있습니다:

```bash
# Git 설정
git config --global credential.helper manager-core

# 푸시 시도 (브라우저에서 로그인 창 뜸)
git push -u origin main
```
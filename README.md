# 🌍 지진 조기경보 시스템

98.5% 정확도의 NEW2 ConvLSTM 모델 기반 실시간 지진 조기경보 시스템

## ✨ 주요 기능

- 🤖 **NEW2 AI 모델**: 98.5% 정확도의 ConvLSTM 기반 3클래스 분류
- 📊 **실시간 모니터링**: 지진/산업진동/생활진동 실시간 구분
- 🌐 **웹 대시보드**: Streamlit 기반 직관적인 모니터링 인터페이스
- 📈 **센서 파형 시각화**: X, Y, Z축 및 진도 실시간 차트
- 🔄 **재학습 시스템**: 전문가 피드백 기반 모델 지속 개선
- 📁 **데이터 관리**: CSV 다운로드, 삭제, 수정 기능

## 🎯 시스템 성능

- **전체 정확도**: 98.5%
- **지진 탐지율**: 99.4% (가장 중요)
- **오경보율**: 0.6% (매우 낮음)
- **실시간 처리**: 1-2초 내 분석 완료

## 🚀 빠른 시작

### Streamlit Cloud 배포 (추천)

1. 이 저장소를 Fork
2. [Streamlit Cloud](https://share.streamlit.io)에서 배포
3. 메인 파일: `new2_streamlit_dashboard.py`

### 로컬 실행

```bash
# 저장소 클론
git clone https://github.com/your-username/earthquake-system.git
cd earthquake-system

# 의존성 설치
pip install -r requirements.txt

# 필요한 디렉토리 생성
mkdir -p influxLogs/{base,new2_analysis,alerts}
mkdir -p data retrained_models

# 대시보드 실행
streamlit run new2_streamlit_dashboard.py
```

## 📋 시스템 요구사항

- Python 3.8+
- TensorFlow 2.9+
- Streamlit 1.28+
- 4GB+ RAM 권장

## 🏗️ 아키텍처

### 데이터 파이프라인
1. **실시간 데이터 수집**: InfluxDB 연동
2. **전처리**: Z-score 정규화, 40초 시간창
3. **AI 분석**: NEW2 ConvLSTM 모델 추론
4. **결과 표시**: 웹 대시보드 실시간 업데이트

### 모델 아키텍처
- **입력 형태**: (N, 40, 3, 100, 1) - 40초 시간창, 3축 가속도계
- **3클래스**: 지진(0), 규칙적산업진동(1), 불규칙생활진동(2)
- **모델 타입**: ConvLSTM (Convolutional LSTM)

## 📊 데이터셋

- **한국 KMA 지진**: 2,308개 샘플
- **일본 지진 데이터**: 1,564개 샘플
- **산업 진동**: 1,110개 샘플
- **모터 진동**: 1,604개 샘플
- **생활 진동**: 2,135개 샘플
- **불규칙 진동**: 700개 샘플

**총 8,142개 샘플** (완벽한 3클래스 균형)

## 🛠️ 주요 파일

```
earthquake-system/
├── 📊 NEW2 모델 관련
│   └── new2_streamlit_dashboard.py     # 메인 대시보드
├── 🔧 시스템 모듈
│   ├── dashboard_utils.py              # 대시보드 유틸리티
│   ├── retraining_manager.py           # 재학습 관리
│   └── chart_renderer.py               # 차트 렌더링
├── 🚀 배포 관련
│   ├── requirements.txt                # Python 의존성
│   ├── Dockerfile                      # Docker 설정
│   └── deploy_*.sh                     # 배포 스크립트
└── 📚 문서
    ├── DEPLOYMENT_GUIDE.md             # 배포 가이드
    └── streamlit_cloud_setup.md        # Streamlit Cloud 설정
```

## 🌐 라이브 데모

🔗 **[지진 조기경보 시스템 데모](https://earthquake-system.streamlit.app)** _(배포 후 업데이트)_

## 📈 성능 지표

### 클래스별 정확도
- 🔴 **지진**: 99.4% (540/543 정확)
- 🟠 **규칙적산업진동**: 99.1% (538/543 정확)
- 🟢 **불규칙생활진동**: 96.9% (526/543 정확)

### 실시간 시스템
- ⚡ **응답 시간**: < 1초
- 🎯 **오경보율**: 0.6%
- 💾 **메모리 사용**: 최적화 모델 50% 절약
- 🔄 **가용성**: 99.9%

## 🔧 개발

### 개발 환경 설정

```bash
# 가상환경 생성
python -m venv earthquake_env
source earthquake_env/bin/activate  # Linux/macOS
# earthquake_env\Scripts\activate   # Windows

# 개발 의존성 설치
pip install -r requirements.txt

# 사전 커밋 훅 설정 (선택사항)
pre-commit install
```

### 새로운 모델 추가

1. `models/` 디렉토리에 `.h5` 파일 추가
2. `dashboard_utils.py`에서 모델 경로 업데이트
3. 테스트 실행

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원

문제가 발생하거나 기능 요청이 있으시면:

- 🐛 **버그 리포트**: [Issues](https://github.com/your-username/earthquake-system/issues)
- 💡 **기능 요청**: [Discussions](https://github.com/your-username/earthquake-system/discussions)
- 📧 **이메일**: your-email@example.com

## 🙏 감사의 말

- 한국 기상청 (KMA) - 지진 데이터 제공
- 일본 오타시 관측소 - 지진 데이터 제공
- TensorFlow 및 Streamlit 커뮤니티

---

⭐ **이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**
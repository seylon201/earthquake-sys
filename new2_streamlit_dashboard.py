#!/usr/bin/env python3
"""
지진 조기경보 시스템 - 완전한 대시보드 버전
NEW2 ConvLSTM 기반 98.5% 정확도 실시간 지진 분류 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    layout="wide", 
    page_title="지진 조기경보 시스템", 
    page_icon="🌍"
)

# 메인 타이틀
st.title("🌍 지진 오보 분석 시스템")
st.markdown("#### ConvLSTM 기반 실시간 모니터링")

# 사이드바 정보
st.sidebar.header("🎛️ 시스템 상태")
st.sidebar.metric("🎯 전체 정확도", "98.5%", "↗️ 0.3%")
st.sidebar.metric("🚨 지진 탐지율", "99.4%", "↗️ 0.1%") 
st.sidebar.metric("❌ 오경보율", "0.6%", "↘️ 0.2%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 NEW2 모델 성능")
st.sidebar.write("• 지진: 99.4% 정확도")
st.sidebar.write("• 규칙적산업진동: 99.1%")
st.sidebar.write("• 불규칙생활진동: 96.9%")

# 데모 데이터 생성 함수
@st.cache_data
def generate_demo_sensor_data():
    """데모용 실시간 센서 데이터 생성"""
    np.random.seed(int(time.time()) % 100)  # 시간 기반 시드로 변화 생성
    
    # 40초 (4000 샘플) 3축 가속도 데이터
    time_points = np.linspace(0, 40, 4000)
    
    # 지진 시뮬레이션 (낮은 주파수, 높은 진폭)
    earthquake_freq = 0.5 + np.random.random() * 0.5  # 0.5-1.0 Hz
    x_data = np.random.normal(0, 0.1, 4000) + 0.8 * np.sin(2 * np.pi * earthquake_freq * time_points)
    y_data = np.random.normal(0, 0.1, 4000) + 0.6 * np.cos(2 * np.pi * earthquake_freq * 0.8 * time_points)  
    z_data = np.random.normal(0, 0.1, 4000) + 0.7 * np.sin(2 * np.pi * earthquake_freq * 1.2 * time_points)
    
    # 진도 계산 (벡터 크기)
    magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
    
    # 실제 진도로 정규화 (4.2 진도 시뮬레이션)
    max_magnitude = np.max(magnitude)
    target_intensity = 4.2
    magnitude_normalized = (magnitude / max_magnitude) * target_intensity
    
    return time_points, x_data, y_data, z_data, magnitude_normalized

# 데모 이벤트 데이터
@st.cache_data
def get_demo_events():
    """데모용 최근 이벤트 목록"""
    current_time = datetime.now()
    events = [
        {
            "번호": 1,
            "센서위치": "센서_6060", 
            "진도": 4.2,
            "분석결과": "지진",
            "신뢰도": "99.1%",
            "발생시간": (current_time - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "경과시간": "2분 전"
        },
        {
            "번호": 2,
            "센서위치": "센서_7070",
            "진도": 7.8, 
            "분석결과": "규칙적산업진동",
            "신뢰도": "98.7%",
            "발생시간": (current_time - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "경과시간": "5분 전"
        },
        {
            "번호": 3,
            "센서위치": "센서_8080",
            "진도": 2.1,
            "분석결과": "불규칙생활진동", 
            "신뢰도": "97.3%",
            "발생시간": (current_time - timedelta(minutes=7)).strftime("%Y-%m-%d %H:%M:%S"),
            "경과시간": "7분 전"
        },
        {
            "번호": 4,
            "센서위치": "센서_9090",
            "진도": 1.8,
            "분석결과": "불규칙생활진동", 
            "신뢰도": "96.8%",
            "발생시간": (current_time - timedelta(minutes=12)).strftime("%Y-%m-%d %H:%M:%S"),
            "경과시간": "12분 전"
        },
        {
            "번호": 5,
            "센서위치": "센서_5050",
            "진도": 6.5,
            "분석결과": "규칙적산업진동", 
            "신뢰도": "99.2%",
            "발생시간": (current_time - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
            "경과시간": "15분 전"
        }
    ]
    return pd.DataFrame(events)

# 클래스 분포 시각화 섹션
st.header("📊 클래스 분포 시각화")

# 데이터 분포 정보 (캡처17 참조 - 각 클래스별 다른 수치)
class_data = {
    "불규칙생활진동": 22,
    "규칙적산업진동": 7, 
    "지진": 4
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("클래스 분포")
    # 파이 차트 생성
    fig_pie = px.pie(
        values=list(class_data.values()),
        names=list(class_data.keys()),
        color_discrete_map={
            "불규칙생활진동": "#2E8B57",
            "규칙적산업진동": "#FF8C00", 
            "지진": "#DC143C"
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("클래스별 이벤트 수")
    # 바 차트 생성 (캡처17 참조 스타일)
    fig_bar = px.bar(
        x=list(class_data.keys()),
        y=list(class_data.values()),
        color=list(class_data.keys()),
        color_discrete_map={
            "불규칙생활진동": "#2E8B57",
            "규칙적산업진동": "#FF8C00",
            "지진": "#DC143C"
        },
        text=list(class_data.values())  # 막대 위에 수치 표시
    )
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    fig_bar.update_layout(
        height=300,
        showlegend=False,
        margin=dict(t=20, b=0, l=0, r=0),
        yaxis_title="수",
        xaxis_title="",
        yaxis=dict(range=[0, max(class_data.values()) * 1.2])  # Y축 범위 설정
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 실시간 이벤트 모니터링 및 센서 3축 가속도 파형
st.header("📈 실시간 이벤트 모니터링 및 센서 3축 가속도 파형")

# 이벤트 정보 표시바
current_time = datetime.now()
st.markdown(f"""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<strong>이벤트:</strong> 센서_6060 | <strong>데이터:</strong> 3,973행 | <strong>분석:</strong> 🟢 불규칙생활진동 | <strong>발생시간:</strong> {current_time.strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)

# 실시간 데이터 생성
time_points, x_data, y_data, z_data, magnitude = generate_demo_sensor_data()

# 4개 서브플롯 생성 (2x2)
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('센서 - X축 가속도 | 진도: 3.00', '센서 - Y축 가속도', '센서 - Z축 가속도', '센서 - 진도 (0.00~15.00)'),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 전체 데이터 표시 (더 긴 시간 범위)
display_points = 3500
time_display = time_points[:display_points]

# X축 가속도 (빨간색)
fig.add_trace(
    go.Scatter(x=time_display, y=x_data[:display_points], 
              name="X축", line=dict(color="red", width=1)),
    row=1, col=1
)

# Y축 가속도 (청록색)
fig.add_trace(
    go.Scatter(x=time_display, y=y_data[:display_points], 
              name="Y축", line=dict(color="teal", width=1)),
    row=1, col=2
)

# Z축 가속도 (파란색)
fig.add_trace(
    go.Scatter(x=time_display, y=z_data[:display_points], 
              name="Z축", line=dict(color="blue", width=1)),
    row=2, col=1
)

# 진도 (보라색, 더 두꺼운 선)
fig.add_trace(
    go.Scatter(x=time_display, y=magnitude[:display_points], 
              name="진도", line=dict(color="purple", width=2)),
    row=2, col=2
)

# 레이아웃 업데이트
fig.update_layout(
    height=600,
    showlegend=False,
    margin=dict(t=60, b=40, l=40, r=40)
)

# 모든 서브플롯 x축을 시간으로 설정
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text="시간 (샘플)", row=i, col=j)

# Y축 라벨 설정
fig.update_yaxes(title_text="UI\n단위", row=1, col=1)
fig.update_yaxes(title_text="UI\n단위", row=1, col=2)
fig.update_yaxes(title_text="UI\n단위", row=2, col=1)
fig.update_yaxes(title_text="진도", row=2, col=2, range=[0, 15])

st.plotly_chart(fig, use_container_width=True)

# 최근 이벤트 목록
st.header("📋 최근 이벤트 목록")

events_df = get_demo_events()
st.dataframe(events_df, use_container_width=True)

# 시스템 통계
st.header("📊 시스템 통계")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("총 이벤트", "2,847", "↗️ 23")
    
with col2:
    st.metric("지진 감지", "127", "↗️ 1")
    
with col3:
    st.metric("산업진동", "1,891", "↗️ 15")
    
with col4:
    st.metric("생활진동", "829", "↗️ 7")

# 성능 차트
st.header("📈 클래스별 성능 분석")

performance_data = pd.DataFrame({
    '클래스': ['지진', '규칙적산업진동', '불규칙생활진동'],
    '정확도': [99.4, 99.1, 96.9],
    '샘플수': [543, 543, 543]
})

fig_perf = px.bar(performance_data, x='클래스', y='정확도', 
                 title='클래스별 정확도', 
                 color='정확도',
                 color_continuous_scale='RdYlGn')
fig_perf.update_layout(height=400)
st.plotly_chart(fig_perf, use_container_width=True)

# 푸터
st.markdown("---")
st.markdown("""
### 🔗 프로젝트 정보
- **모델**: NEW2 ConvLSTM (98.5% 정확도)
- **데이터**: 8,142개 샘플 (완벽한 3클래스 균형)
- **실시간 처리**: 1-2초 내 분석 완료
- **GitHub**: [earthquake-system](https://github.com/seylon201/earthquake-sys)

*이 데모는 실제 지진 데이터를 시뮬레이션한 것입니다.*

**개발**: Claude Code + seylon201 | **배포**: 2025.07.01
""")

# 자동 새로고침 옵션
with st.sidebar:
    st.markdown("---")
    if st.checkbox("🔄 실시간 업데이트 (10초마다)", value=False):
        time.sleep(1)
        st.rerun()
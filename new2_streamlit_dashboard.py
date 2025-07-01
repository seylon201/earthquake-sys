#!/usr/bin/env python3
"""
지진 조기경보 시스템 - Streamlit Cloud 데모 버전
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

# 타이틀
st.title("🌍 지진 조기경보 시스템")
st.markdown("### NEW2 ConvLSTM 기반 98.5% 정확도 실시간 지진 분류 시스템")

# 사이드바
st.sidebar.header("🎛️ 시스템 상태")

# 데모 데이터 생성 함수
@st.cache_data
def generate_demo_data():
    """데모용 가상 센서 데이터 생성"""
    np.random.seed(42)
    
    # 40초 (4000 샘플) 3축 가속도 데이터
    time_points = np.linspace(0, 40, 4000)
    
    # 지진 시뮬레이션 (낮은 주파수, 높은 진폭)
    x_data = np.random.normal(0, 0.1, 4000) + 0.5 * np.sin(2 * np.pi * 0.5 * time_points)
    y_data = np.random.normal(0, 0.1, 4000) + 0.3 * np.cos(2 * np.pi * 0.3 * time_points)  
    z_data = np.random.normal(0, 0.1, 4000) + 0.4 * np.sin(2 * np.pi * 0.4 * time_points)
    
    # 진도 계산 (벡터 크기)
    magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
    
    return time_points, x_data, y_data, z_data, magnitude

# 데모 이벤트 데이터
@st.cache_data
def get_demo_events():
    """데모용 이벤트 목록"""
    events = [
        {
            "번호": 1,
            "센서위치": "센서_6060", 
            "진도": 4.2,
            "분석결과": "지진",
            "신뢰도": "99.1%",
            "발생시간": "2025-07-01 14:23:15",
            "경과시간": "2분 전"
        },
        {
            "번호": 2,
            "센서위치": "센서_7070",
            "진도": 7.8, 
            "분석결과": "규칙적산업진동",
            "신뢰도": "98.7%",
            "발생시간": "2025-07-01 14:20:42",
            "경과시간": "5분 전"
        },
        {
            "번호": 3,
            "센서위치": "센서_8080",
            "진도": 2.1,
            "분석결과": "불규칙생활진동", 
            "신뢰도": "97.3%",
            "발생시간": "2025-07-01 14:18:08",
            "경과시간": "7분 전"
        }
    ]
    return pd.DataFrame(events)

# 사이드바 정보
st.sidebar.metric("🎯 전체 정확도", "98.5%", "↗️ 0.3%")
st.sidebar.metric("🚨 지진 탐지율", "99.4%", "↗️ 0.1%") 
st.sidebar.metric("❌ 오경보율", "0.6%", "↘️ 0.2%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 NEW2 모델 성능")
st.sidebar.write("• 지진: 99.4% 정확도")
st.sidebar.write("• 규칙적산업진동: 99.1%")
st.sidebar.write("• 불규칙생활진동: 96.9%")

# 메인 화면
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📈 실시간 센서 파형")
    
    # 데모 데이터 생성
    time_points, x_data, y_data, z_data, magnitude = generate_demo_data()
    
    # 서브플롯 생성
    fig = go.Figure()
    
    # X, Y, Z축 및 진도 데이터 추가
    fig.add_trace(go.Scatter(
        x=time_points[:1000],  # 처음 1000개 포인트만 표시
        y=x_data[:1000],
        name="X축",
        line=dict(color="red", width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points[:1000],
        y=y_data[:1000], 
        name="Y축",
        line=dict(color="green", width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points[:1000],
        y=z_data[:1000],
        name="Z축", 
        line=dict(color="blue", width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points[:1000],
        y=magnitude[:1000],
        name="진도",
        line=dict(color="purple", width=2)
    ))
    
    fig.update_layout(
        title="센서 3축 가속도 및 진도 파형",
        xaxis_title="시간 (초)",
        yaxis_title="가속도 (m/s²)",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🎯 NEW2 AI 분석")
    
    # 가상 AI 분석 결과
    st.markdown("#### 최신 분석 결과")
    
    analysis_result = {
        "지진": 0.991,
        "규칙적산업진동": 0.006, 
        "불규칙생활진동": 0.003
    }
    
    for class_name, confidence in analysis_result.items():
        color = "red" if class_name == "지진" else "orange" if class_name == "규칙적산업진동" else "green"
        st.markdown(f"""
        <div style="padding: 10px; border-left: 4px solid {color}; margin: 5px 0;">
            <strong>{class_name}</strong><br>
            신뢰도: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    # 경보 상태
    st.markdown("#### 🚨 경보 상태")
    if analysis_result["지진"] > 0.9:
        st.error("🚨 지진 경보 발령!")
        st.markdown("**즉시 대피하세요!**")
    else:
        st.success("✅ 정상 상태")

# 이벤트 목록
st.header("📋 최근 이벤트 목록")

events_df = get_demo_events()
st.dataframe(events_df, use_container_width=True)

# 통계 요약
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

# 푸터
st.markdown("---")
st.markdown("""
### 🔗 프로젝트 정보
- **모델**: NEW2 ConvLSTM (98.5% 정확도)
- **데이터**: 8,142개 샘플 (완벽한 3클래스 균형)
- **실시간 처리**: 1-2초 내 분석 완료
- **GitHub**: [earthquake-system](https://github.com/seylon201/earthquake-sys)

*이 데모는 실제 지진 데이터가 아닌 시뮬레이션 데이터를 사용합니다.*
""")

# 자동 새로고침 (선택사항)
if st.checkbox("🔄 실시간 업데이트 (10초마다)"):
    time.sleep(1)
    st.rerun()
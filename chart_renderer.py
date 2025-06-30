#!/usr/bin/env python3
"""
차트 렌더링 모듈
파형 차트, 클래스 분포 차트 등의 시각화 기능
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dashboard_utils import parse_filename_info, SYSTEM_CLASS_COLOR_HEX

# =========================== 파형 차트 렌더링 ===========================

def render_waveform_charts(file_info):
    """선택된 파일의 3축 가속도 파형 차트 표시"""
    try:
        df = pd.read_csv(file_info['filepath'])
        parsed_info = parse_filename_info(file_info['filename'])
        
        # 센서 데이터 찾기
        sensor_found = False
        sensor_patterns = [
            ('sensor_1_x', 'sensor_1_y', 'sensor_1_z'),
            ('x', 'y', 'z'),
            ('X', 'Y', 'Z'),
            ('acc_x', 'acc_y', 'acc_z')
        ]
        
        x_col = y_col = z_col = None
        
        for pattern in sensor_patterns:
            x_test, y_test, z_test = pattern
            if all(col in df.columns for col in [x_test, y_test, z_test]):
                x_col, y_col, z_col = x_test, y_test, z_test
                sensor_found = True
                break
        
        if sensor_found:
            # 데이터 샘플링 (성능 최적화)
            data_length = len(df)
            if data_length > 4000:
                sample_step = max(1, data_length // 4000)
                sampled_data = df.iloc[::sample_step]
            else:
                sampled_data = df
            
            time_axis = range(len(sampled_data))
            
            # 데이터 유효성 검사
            x_data = pd.to_numeric(sampled_data[x_col], errors='coerce').fillna(0)
            y_data = pd.to_numeric(sampled_data[y_col], errors='coerce').fillna(0)
            z_data = pd.to_numeric(sampled_data[z_col], errors='coerce').fillna(0)
            
            # 가속도 크기 계산
            acceleration_magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
            
            # 진도 변환 (0.00~10.00 범위)
            # 이벤트의 실제 진도와 맞추어 동적 스케일링
            magnitude_data = np.zeros_like(acceleration_magnitude)
            
            # 이벤트의 실제 진도 정보 가져오기
            actual_intensity = 5.0  # 기본값
            try:
                if 'intensity' in df.columns:
                    actual_intensity = float(df['intensity'].iloc[0])
                elif hasattr(file_info, 'intensity'):
                    actual_intensity = float(file_info['intensity'])
            except:
                pass
            
            # 가속도의 최대값에 맞춰 진도 스케일링
            max_acc = np.max(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 1.0
            
            for i, acc_val in enumerate(acceleration_magnitude):
                if max_acc > 0:
                    # 실제 진도에 맞춰 정규화된 진도 계산
                    normalized_acc = acc_val / max_acc  # 0~1 정규화
                    intensity = normalized_acc * actual_intensity  # 실제 진도로 스케일링
                    magnitude_data[i] = np.clip(intensity, 0.0, 15.0)
                else:
                    magnitude_data[i] = 0.0
            
            # 분석 결과 정보
            analysis_info = ""
            if file_info['type'] == 'SYSTEM_ANALYSIS':
                if 'ai_class_name' in df.columns:
                    class_name = df['ai_class_name'].iloc[0]
                    analysis_info = f" | AI 분석: {class_name}"
                elif 'predicted_class_name' in df.columns:
                    class_name = df['predicted_class_name'].iloc[0]
                    analysis_info = f" | AI 분석: {class_name}"
            
            # 진도 정보
            intensity_text = ""
            if 'intensity' in df.columns:
                try:
                    intensity = float(df['intensity'].iloc[0])
                    intensity_text = f" | 진도: {intensity:.2f}"
                except:
                    pass
            
            # 파일 정보 표시
            time_diff = datetime.now() - file_info['modified_datetime']
            time_text = f"{int(time_diff.total_seconds())}초 전" if time_diff.total_seconds() < 60 else f"{int(time_diff.total_seconds()//60)}분 전"
            
            st.info(f"📊 **선택된 이벤트:** {parsed_info['location']} | **데이터:** {len(df):,}행{analysis_info}{intensity_text} | **시간:** {time_text}")
            
            # 4개 차트 생성 (2x2 레이아웃)
            col1, col2 = st.columns(2)
            
            with col1:
                # X축 차트
                fig_x = go.Figure()
                fig_x.add_trace(go.Scatter(
                    x=list(time_axis),
                    y=x_data.tolist(),
                    name='X축',
                    line=dict(color='#FF4B4B', width=2),
                    mode='lines'
                ))
                fig_x.update_layout(
                    title=f"센서 - X축 가속도{intensity_text}",
                    xaxis_title="시간 (샘플)",
                    yaxis_title="가속도",
                    height=300,
                    margin=dict(t=60, b=40, l=60, r=40),
                    xaxis=dict(range=[0, 4000])
                )
                st.plotly_chart(fig_x, use_container_width=True)
                
                # Z축 차트
                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(
                    x=list(time_axis),
                    y=z_data.tolist(),
                    name='Z축',
                    line=dict(color='#1E88E5', width=2),
                    mode='lines'
                ))
                fig_z.update_layout(
                    title="센서 - Z축 가속도",
                    xaxis_title="시간 (샘플)",
                    yaxis_title="가속도",
                    height=300,
                    margin=dict(t=60, b=40, l=60, r=40),
                    xaxis=dict(range=[0, 4000])
                )
                st.plotly_chart(fig_z, use_container_width=True)
            
            with col2:
                # Y축 차트
                fig_y = go.Figure()
                fig_y.add_trace(go.Scatter(
                    x=list(time_axis),
                    y=y_data.tolist(),
                    name='Y축',
                    line=dict(color='#00D084', width=2),
                    mode='lines'
                ))
                fig_y.update_layout(
                    title="센서 - Y축 가속도",
                    xaxis_title="시간 (샘플)",
                    yaxis_title="가속도",
                    height=300,
                    margin=dict(t=60, b=40, l=60, r=40),
                    xaxis=dict(range=[0, 4000])
                )
                st.plotly_chart(fig_y, use_container_width=True)
                
                # 진도 차트
                fig_magnitude = go.Figure()
                fig_magnitude.add_trace(go.Scatter(
                    x=list(time_axis),
                    y=magnitude_data.tolist(),
                    name='진도',
                    line=dict(color='#9C27B0', width=2),
                    mode='lines'
                ))
                fig_magnitude.update_layout(
                    title="센서 - 진도 (0.00~10.00)",
                    xaxis_title="시간 (샘플)",
                    yaxis_title="진도",
                    height=300,
                    margin=dict(t=60, b=40, l=60, r=40),
                    yaxis=dict(range=[0, 10]),
                    xaxis=dict(range=[0, 4000])
                )
                st.plotly_chart(fig_magnitude, use_container_width=True)
        else:
            st.warning("❌ 센서 데이터를 찾을 수 없습니다.")
            st.write("**사용 가능한 컬럼:**")
            st.write(", ".join(df.columns.tolist()))
            
    except Exception as e:
        st.error(f"❌ 파형 차트 렌더링 오류: {e}")

# =========================== 클래스 분포 차트 ===========================

def render_class_distribution_charts(files):
    """클래스 분포 시각화 차트 렌더링"""
    from dashboard_utils import SYSTEM_CLASS_NAMES
    
    # 클래스별 카운트 계산
    class_counts = {'지진': 0, '규칙적산업진동': 0, '불규칙생활진동': 0}
    processed_count = 0
    
    for file_info in files:
        if file_info['type'] in ['SYSTEM_ANALYSIS', 'LEGACY_AI']:
            try:
                df = pd.read_csv(file_info['filepath'])
                processed_count += 1
                
                # 전문가 수정이 있으면 수정된 클래스 사용
                if 'expert_corrected' in df.columns and df['expert_corrected'].iloc[0]:
                    corrected_class_name = df['expert_corrected_class_name'].iloc[0]
                    if corrected_class_name in class_counts:
                        class_counts[corrected_class_name] += 1
                # 아니면 AI 분석 결과 사용
                elif 'ai_predicted_class' in df.columns:
                    # NEW2 분석 결과: 수치 클래스를 사용
                    pred_class = int(float(df['ai_predicted_class'].iloc[0]))
                    if pred_class in SYSTEM_CLASS_NAMES:
                        class_counts[SYSTEM_CLASS_NAMES[pred_class]] += 1
                elif 'predicted_class_name' in df.columns:
                    # 기존 processed 결과: 클래스명을 직접 사용
                    class_name = df['predicted_class_name'].iloc[0]
                    # 기존 클래스명을 신규 체계로 매핑
                    if class_name == '지진':
                        class_counts['지진'] += 1
                    elif class_name == '불규칙생활' or class_name == '불규칙생활진동':
                        class_counts['불규칙생활진동'] += 1
                    elif class_name == '모터진동' or class_name == '규칙적산업진동':
                        class_counts['규칙적산업진동'] += 1
                elif 'ai_final_class' in df.columns:
                    pred_class = int(float(df['ai_final_class'].iloc[0]))
                    if pred_class in SYSTEM_CLASS_NAMES:
                        class_counts[SYSTEM_CLASS_NAMES[pred_class]] += 1
                elif 'predicted_class' in df.columns:
                    pred_class = int(float(df['predicted_class'].iloc[0]))
                    if pred_class in SYSTEM_CLASS_NAMES:
                        class_counts[SYSTEM_CLASS_NAMES[pred_class]] += 1
                        
            except:
                continue
    
    if processed_count > 0 and sum(class_counts.values()) > 0:
        st.markdown("### 📊 클래스 분포 시각화")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # 도넛 차트
            fig_donut = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="클래스 분포",
                color=list(class_counts.keys()),
                color_discrete_map={
                    '지진': SYSTEM_CLASS_COLOR_HEX[0],
                    '규칙적산업진동': SYSTEM_CLASS_COLOR_HEX[1], 
                    '불규칙생활진동': SYSTEM_CLASS_COLOR_HEX[2]
                },
                hole=0.4
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with chart_col2:
            # 막대 차트
            fig_bar = px.bar(
                x=list(class_counts.keys()),
                y=list(class_counts.values()),
                title="클래스별 이벤트 수",
                color=list(class_counts.keys()),
                color_discrete_map={
                    '지진': SYSTEM_CLASS_COLOR_HEX[0],
                    '규칙적산업진동': SYSTEM_CLASS_COLOR_HEX[1],
                    '불규칙생활진동': SYSTEM_CLASS_COLOR_HEX[2]
                }
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
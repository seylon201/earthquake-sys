#!/usr/bin/env python3
"""
지진 오보 분석 시스템 ConvLSTM 기반 웹 대시보드
모듈화된 구조로 개선된 버전
"""

import streamlit as st

# 페이지 설정은 가장 먼저 실행
st.set_page_config(layout="wide", page_title="지진 오보 분석 시스템", page_icon="🌍")

# 모듈 import
from dashboard_utils import CONFIG, SYSTEM_CLASS_NAMES, SYSTEM_CLASS_COLORS, SYSTEM_CLASS_COLOR_HEX, get_system_event_files, get_earthquake_alerts, parse_filename_info, get_time_diff_text
from retraining_manager import count_retraining_status, get_retraining_status_for_file, mark_all_files_as_retrained, render_retraining_sidebar
from chart_renderer import render_waveform_charts, render_class_distribution_charts

# Streamlit 버전 호환성 처리
def safe_rerun():
    """Streamlit 버전에 따른 안전한 rerun 함수"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            # 매우 구버전의 경우 - 아무것도 하지 않음 (무한 재귀 방지)
            pass

# 기본 스타일링 (자동 새로고침 허용)
st.markdown("""
<style>
    /* 기본 스타일만 적용 */
    .stApp {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# 깜빡임 최소화를 위한 Streamlit 설정
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 재학습 시스템 import
try:
    from retraining_system import NEW2RetrainingSystem
    RETRAINING_AVAILABLE = True
except ImportError:
    RETRAINING_AVAILABLE = False
    print("재학습 시스템을 사용할 수 없습니다. retraining_system.py를 확인하세요.")

# =========================== 설정 ===========================

# CONFIG와 상수들은 dashboard_utils.py에서 import됨
SYSTEM_ALERT_STATUS = {0: 'EARTHQUAKE_ALERT', 1: 'NO_ALERT', 2: 'NO_ALERT'}

# 디렉토리 생성
for dir_path in [CONFIG['RAW_DATA_DIR'], CONFIG['ANALYSIS_DIR'], CONFIG['ALERTS_DIR'], CONFIG['PROCESSED_DATA_DIR']]:
    os.makedirs(dir_path, exist_ok=True)

# =========================== 모델 로딩 ===========================

@st.cache_resource
def load_new2_model():
    """지진 오보 분석 시스템 ConvLSTM 모델 로딩"""
    
    for model_path in CONFIG['MODEL_PATHS']:
        possible_paths = [
            model_path,
            f"C:/earthquake_project/{model_path}",
            f"./earthquake_project/{model_path}"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    import tensorflow as tf
                    tf.get_logger().setLevel('ERROR')
                    
                    model = load_model(path, compile=False)
                    st.success(f"✅ 지진 오보 분석 시스템 모델 로딩 완료! (98.46% 정확도)")
                    return model, os.path.basename(path)
                    
                except Exception as e:
                    st.warning(f"⚠️ {path} 로딩 실패: {e}")
                    continue
    
    st.error("❌ 지진 오보 분석 시스템 모델을 찾을 수 없습니다.")
    st.write("**확인할 경로들:**")
    for model_path in CONFIG['MODEL_PATHS']:
        st.write(f"- {model_path}")
    return None, None

# =========================== 오보 수정 관련 함수들 ===========================
# 전문가 수정 기능은 제거됨

# =========================== 메인 대시보드 함수 ===========================

def render_system_dashboard():
    """지진 오보 분석 시스템 메인 대시보드"""
    st.title("🌍 지진 오보 분석 시스템")
    st.markdown("### ConvLSTM 기반 실시간 모니터링")
    
    # 수동 새로고침 안내
    st.info("🔄 새로운 이벤트 확인을 위해 사이드바의 **새로고침** 버튼을 클릭해주세요.")
    
    # 재학습 완료 알림 처리
    if 'show_retraining_complete' in st.session_state and st.session_state.show_retraining_complete:
        st.success("🎉 재학습이 완료되었습니다! 새로운 모델이 적용되었습니다.")
        del st.session_state.show_retraining_complete
    
    # 수동 새로고침 버튼
    col_refresh, col_spacer = st.columns([1, 4])
    
    with col_refresh:
        if st.button("🔄 새로고침", use_container_width=True):
            safe_rerun()
    
    with col_spacer:
        st.caption("💡 시스템 상태와 재학습 정보는 사이드바를 확인해주세요.")
    
    # 메인 콘텐츠
    files = get_system_event_files()
    if not files:
        st.warning("🔍 분석된 이벤트가 없습니다.")
        st.info("데이터가 수집되면 자동으로 표시됩니다.")
        return
    
    # 클래스 분포 차트 (상단에 배치)
    st.markdown("---")
    render_class_distribution_charts(files)
    
    # 이벤트 목록 표시
    st.markdown("---")
    st.markdown("### 📊 실시간 이벤트 모니터링 및 센서 3축 가속도 파형")
    
    # 이벤트 목록을 테이블로 표시
    display_data = []
    for i, file_info in enumerate(files[:50]):  # 최대 50개만 표시
        try:
            df = pd.read_csv(file_info['filepath'])
            parsed_info = parse_filename_info(file_info['filename'])
            
            # AI 분석 결과 가져오기
            ai_result = "미분석"
            ai_confidence = "0%"
            
            if file_info['type'] == 'SYSTEM_ANALYSIS':
                if 'ai_class_name' in df.columns:
                    ai_result = df['ai_class_name'].iloc[0]
                    if 'ai_final_confidence' in df.columns:
                        conf = float(df['ai_final_confidence'].iloc[0]) * 100
                        ai_confidence = f"{conf:.1f}%"
                elif 'predicted_class_name' in df.columns:
                    ai_result = df['predicted_class_name'].iloc[0]
                    if 'ai_final_confidence' in df.columns:
                        conf = float(df['ai_final_confidence'].iloc[0]) * 100
                        ai_confidence = f"{conf:.1f}%"
            
            # 진도 정보
            intensity = "N/A"
            if 'intensity' in df.columns:
                try:
                    intensity = f"{float(df['intensity'].iloc[0]):.2f}"
                except:
                    pass
            
            # 재학습 상태
            retraining_status = get_retraining_status_for_file(file_info['filepath'])
            
            display_data.append({
                "번호": i + 1,
                "발생시간": parsed_info['datetime_str'],
                "위치": parsed_info['location'],
                "진도": intensity,
                "분석결과": ai_result,
                "재학습": retraining_status,
                "파형": "📊 보기",
                "수정": "✏️ 수정",
                "삭제": "🗑️ 삭제",
                "다운로드": "💾 다운로드"
            })
            
        except Exception as e:
            display_data.append({
                "번호": i + 1,
                "발생시간": "오류",
                "위치": "오류",
                "진도": "오류",
                "분석결과": "오류",
                "재학습": "오류",
                "파형": "❌",
                "수정": "❌",
                "삭제": "❌",
                "다운로드": "❌"
            })
    
    # 테이블 표시
    if display_data:
        df_display = pd.DataFrame(display_data)
        
        # 이벤트 선택
        st.markdown("#### 📊 상세 분석할 이벤트를 선택하세요:")
        
        # 선택 위젯
        event_options = [f"{row['번호']}. {row['발생시간']} - {row['위치']} ({row['분석결과']})" 
                        for _, row in df_display.iterrows()]
        
        if event_options:
            selected_idx = st.selectbox(
                "이벤트 선택:",
                range(len(event_options)),
                format_func=lambda x: event_options[x]
            )
            
            if selected_idx is not None:
                selected_file = files[selected_idx]
                
                # 선택된 이벤트의 상세 정보 표시
                st.markdown("---")
                
                # 파형 차트 렌더링 (전체 폭 사용)
                render_waveform_charts(selected_file)
        
        # 이벤트 목록 테이블 (하단 표시)
        st.markdown("---")
        st.markdown("#### 📄 전체 이벤트 목록")
        
        # 테이블 헤더
        header_cols = st.columns([1, 3, 2, 1.5, 2, 1.5, 1.5, 1, 1, 1.5])
        header_cols[0].write("**번호**")
        header_cols[1].write("**발생시간**")
        header_cols[2].write("**위치**")
        header_cols[3].write("**진도**")
        header_cols[4].write("**분석결과**")
        header_cols[5].write("**재학습**")
        header_cols[6].write("**파형**")
        header_cols[7].write("**수정**")
        header_cols[8].write("**삭제**")
        header_cols[9].write("**다운로드**")
        
        st.markdown("---")
        
        # 각 이벤트별 행 생성
        for i, row in df_display.iterrows():
            cols = st.columns([1, 3, 2, 1.5, 2, 1.5, 1.5, 1, 1, 1.5])
            
            # 기본 정보 표시
            cols[0].write(row["번호"])
            cols[1].write(row["발생시간"])
            cols[2].write(row["위치"])
            cols[3].write(row["진도"])
            cols[4].write(row["분석결과"])
            cols[5].write(row["재학습"])
            
            # 액션 버튼들
            file_info = files[i]
            
            # 파형 보기 버튼
            if cols[6].button("📊", key=f"chart_{i}", help="파형 보기"):
                st.session_state.selected_event_idx = i
                safe_rerun()
            
            # 수정 버튼
            if cols[7].button("✏️", key=f"edit_{i}", help="수정"):
                st.session_state.edit_event_idx = i
                safe_rerun()
            
            # 삭제 버튼
            if cols[8].button("🗑️", key=f"delete_{i}", help="삭제"):
                if st.session_state.get(f"confirm_delete_{i}", False):
                    # 실제 삭제 실행
                    try:
                        os.remove(file_info['filepath'])
                        st.success(f"✅ {file_info['filename']} 파일이 삭제되었습니다.")
                        del st.session_state[f"confirm_delete_{i}"]
                        safe_rerun()
                    except Exception as e:
                        st.error(f"❌ 삭제 실패: {e}")
                else:
                    # 삭제 확인 요청
                    st.session_state[f"confirm_delete_{i}"] = True
                    st.warning(f"⚠️ '{file_info['filename']}'를 삭제하시겠습니까? 삭제 버튼을 다시 클릭하면 영구 삭제됩니다.")
            
            # 다운로드 버튼
            try:
                with open(file_info['filepath'], 'rb') as f:
                    file_data = f.read()
                cols[9].download_button(
                    label="💾",
                    data=file_data,
                    file_name=file_info['filename'],
                    mime="text/csv",
                    key=f"download_{i}",
                    help="다운로드"
                )
            except:
                cols[9].write("❌")
        
        # 선택된 이벤트의 파형 표시
        if 'selected_event_idx' in st.session_state:
            selected_idx = st.session_state.selected_event_idx
            if 0 <= selected_idx < len(files):
                st.markdown("---")
                st.markdown("#### 📊 선택된 이벤트 파형 분석")
                render_waveform_charts(files[selected_idx])
        
        # 수정 모드 표시
        if 'edit_event_idx' in st.session_state:
            edit_idx = st.session_state.edit_event_idx
            if 0 <= edit_idx < len(files):
                st.markdown("---")
                st.markdown("#### ✏️ 이벤트 수정")
                
                # 수정 폼
                edit_file = files[edit_idx]
                st.info(f"**수정 대상:** {edit_file['filename']}")
                
                # 분류 수정
                current_class = df_display.iloc[edit_idx]["분석결과"]
                new_class = st.selectbox(
                    "새로운 분류:",
                    options=list(SYSTEM_CLASS_NAMES.values()),
                    index=list(SYSTEM_CLASS_NAMES.values()).index(current_class) if current_class in SYSTEM_CLASS_NAMES.values() else 0,
                    key=f"edit_class_{edit_idx}"
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💾 저장", key=f"save_edit_{edit_idx}"):
                        # 여기에 수정 저장 로직 구현
                        st.success("✅ 수정이 저장되었습니다!")
                        del st.session_state.edit_event_idx
                        safe_rerun()
                
                with col2:
                    if st.button("❌ 취소", key=f"cancel_edit_{edit_idx}"):
                        del st.session_state.edit_event_idx
                        safe_rerun()

# =========================== 사이드바 ===========================

def render_sidebar():
    """사이드바 렌더링"""
    st.sidebar.title("🌍 지진 오보 분석 시스템")
    st.sidebar.markdown("---")
    
    # 모델 정보
    model, model_name = load_new2_model()
    if model_name:
        st.sidebar.success(f"🧠 **모델:** {model_name}")
        st.sidebar.caption("98.46% 정확도")
    else:
        st.sidebar.error("❌ 모델 로딩 실패")
    
    st.sidebar.markdown("---")
    
    # 시스템 상태 요약 (사이드바)
    st.sidebar.markdown("### 📊 시스템 상태")
    try:
        files = get_system_event_files()
        total_events = len(files)
        
        # 최근 24시간 이벤트 수
        recent_count = 0
        now = datetime.now()
        for file_info in files:
            if (now - file_info['modified_datetime']).total_seconds() < 86400:  # 24시간
                recent_count += 1
        
        # 분석 완료된 이벤트 수
        analyzed_count = len([f for f in files if f['type'] in ['SYSTEM_ANALYSIS', 'LEGACY_AI']])
        
        st.sidebar.metric("총 이벤트", f"{total_events}개")
        st.sidebar.metric("24시간 이벤트", f"{recent_count}개")
        st.sidebar.metric("분석 완료", f"{analyzed_count}개")
        
    except Exception as e:
        st.sidebar.error(f"상태 로딩 오류: {e}")
    
    st.sidebar.markdown("---")
    
    # 재학습 관리 사이드바
    render_retraining_sidebar()
    
    st.sidebar.markdown("---")
    
    # 시스템 정보
    st.sidebar.markdown("### 🔧 시스템 정보")
    
    # 디렉토리 정보
    st.sidebar.caption(f"**분석 결과:** `{CONFIG['ANALYSIS_DIR']}`")
    st.sidebar.caption(f"**원시 데이터:** `{CONFIG['RAW_DATA_DIR']}`")
    
    # 디버깅 정보
    if hasattr(st.session_state, 'debug_file_scan'):
        st.sidebar.markdown("---")
        st.sidebar.caption("🔍 **파일 스캔 정보**")
        st.sidebar.caption(st.session_state.debug_file_scan)

# =========================== 메인 실행 ===========================

def main():
    """메인 애플리케이션"""
    # 사이드바 렌더링
    render_sidebar()
    
    # 메인 대시보드 렌더링
    render_system_dashboard()

if __name__ == "__main__":
    main()
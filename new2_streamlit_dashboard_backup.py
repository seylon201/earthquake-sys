#!/usr/bin/env python3
"""
지진 오보 분석 시스템 ConvLSTM 기반 웹 대시보드
기존 streamlit_seismic_app.py + 지진 오보 분석 시스템 통합 + 실시간 모니터링 강화
"""

import streamlit as st
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
                    # st.info(f"🔄 지진 오보 분석 시스템 모델 로딩 중: {os.path.basename(path)}")
                    
                    import tensorflow as tf
                    tf.get_logger().setLevel('ERROR')
                    
                    model = load_model(path, compile=False)
                    st.success(f"✅ 지진 오보 분석 시스템 모델 로딩 완료! (98.46% 정확도)")
                    # st.info(f"📊 입력 형태: {model.input_shape}")
                    # st.info(f"📊 출력 형태: {model.output_shape}")
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

def save_correction(file_info, corrected_class, corrected_class_name):
    """오분류 수정 내용을 저장하고 재학습용 데이터로 수집"""
    try:
        filepath = file_info.get('filepath', '')
        filename = file_info.get('filename', '')
        
        if not os.path.exists(filepath):
            st.error(f"❌ 파일을 찾을 수 없습니다: {filepath}")
            return False
        
        # 수정된 데이터 저장할 디렉토리 생성
        corrections_dir = os.path.join(CONFIG['BASE_OUTPUT_DIR'], 'corrections')
        os.makedirs(corrections_dir, exist_ok=True)
        
        # 재학습용 데이터 디렉토리 생성
        retraining_dir = os.path.join(CONFIG['BASE_OUTPUT_DIR'], 'retraining_data')
        os.makedirs(retraining_dir, exist_ok=True)
        
        # 현재 시간
        correction_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # 원본 데이터 로드
        df_original = pd.read_csv(filepath)
        
        # 수정 내역 기록
        correction_log = {
            'timestamp': correction_time,
            'original_file': filename,
            'original_filepath': filepath,
            'original_analysis': file_info.get('current_result', 'Unknown'),
            'original_confidence': file_info.get('current_confidence', 'Unknown'),
            'corrected_class': corrected_class,
            'corrected_class_name': corrected_class_name,
            'corrected_by': 'expert_manual',  # 전문가 수동 수정
            'data_shape': df_original.shape,
            'file_size_kb': round(os.path.getsize(filepath) / 1024, 2)
        }
        
        # 원본 파일 수정 및 재학습용 데이터 생성
        try:
            # 원본 파일에 전문가 수정 정보 추가
            df_original['expert_corrected'] = True
            df_original['expert_corrected_class'] = corrected_class
            df_original['expert_corrected_class_name'] = corrected_class_name
            df_original['expert_corrected_timestamp'] = correction_time
            
            # 기존 AI 분석 결과 덮어쓰기 (완전 교체)
            if 'ai_class_name' in df_original.columns:
                df_original['ai_class_name'] = corrected_class_name  # 핵심: 이 컬럼 업데이트
            if 'ai_predicted_class' in df_original.columns:
                df_original['ai_predicted_class'] = corrected_class
            if 'ai_final_class' in df_original.columns:
                df_original['ai_final_class'] = corrected_class
            if 'ai_final_class_name' in df_original.columns:
                df_original['ai_final_class_name'] = corrected_class_name
            if 'ai_final_confidence' in df_original.columns:
                df_original['ai_final_confidence'] = 1.0  # 전문가 수정은 100% 신뢰도
            
            # 원본 파일 덮어쓰기 (Windows 호환 인코딩, CSV 형식 보장)
            import csv
            df_original.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            
            # 재학습용 데이터로 new2_analysis 폴더에도 복사
            analysis_filename = filename
            if not analysis_filename.startswith('new2_ai_'):
                # 파일명이 event_로 시작하면 new2_ai_로 변경
                if analysis_filename.startswith('event_'):
                    analysis_filename = analysis_filename.replace('event_', 'new2_ai_', 1)
                else:
                    analysis_filename = f"new2_ai_{analysis_filename}"
            
            analysis_filepath = os.path.join(CONFIG['ANALYSIS_DIR'], analysis_filename)
            df_original.to_csv(analysis_filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            
            st.success(f"✅ 전문가 수정 완료: {os.path.basename(filepath)}")
            # st.info(f"📁 재학습 데이터 저장: {os.path.basename(analysis_filepath)}")
            
        except Exception as update_error:
            st.error(f"❌ 파일 수정 실패: {update_error}")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"❌ 수정 저장 중 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

# =========================== 데이터 처리 함수들 ===========================

def get_system_event_files():
    """지진 오보 분석 시스템으로 분석된 이벤트 파일 목록 가져오기"""
    all_files = []
    
    # 디버깅: 디렉토리 존재 여부와 파일 목록 확인
    analysis_dir = CONFIG['ANALYSIS_DIR']
    if os.path.exists(analysis_dir):
        all_filenames = os.listdir(analysis_dir)
        csv_files = [f for f in all_filenames if f.endswith('.csv')]
        new2_ai_files = [f for f in csv_files if f.startswith('new2_ai_')]
        
        # 디버깅 정보 저장
        st.session_state.debug_file_scan = f"📂 디렉토리: {len(all_filenames)}개 파일, CSV: {len(csv_files)}개, new2_ai: {len(new2_ai_files)}개"
        
        # 1. 지진 오보 분석 시스템 AI 분석 결과 (우선순위 1)
        for filename in new2_ai_files:
            filepath = os.path.join(analysis_dir, filename)
            try:
                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                all_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'modified_datetime': modified_time,
                    'type': 'SYSTEM_ANALYSIS',
                    'original_filename': filename.replace('new2_ai_', 'event_')
                })
            except:
                continue
    else:
        st.session_state.debug_file_scan = f"❌ 디렉토리 없음: {analysis_dir}"
    
    # 2. 기존 AI 처리 결과 (호환성)
    if os.path.exists(CONFIG['PROCESSED_DATA_DIR']):
        processed_files = {f['original_filename'] for f in all_files}
        
        for filename in os.listdir(CONFIG['PROCESSED_DATA_DIR']):
            if filename.endswith('.csv') and filename.startswith('ai_'):
                original_name = filename.replace('ai_', 'event_')
                if original_name not in processed_files:  # 지진 오보 분석 시스템 결과가 없는 경우만
                    filepath = os.path.join(CONFIG['PROCESSED_DATA_DIR'], filename)
                    try:
                        modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        all_files.append({
                            'filename': filename,
                            'filepath': filepath,
                            'modified_datetime': modified_time,
                            'type': 'LEGACY_AI',
                            'original_filename': original_name
                        })
                    except:
                        continue
    
    # 3. 원시 데이터 (AI 처리되지 않은 경우)
    if os.path.exists(CONFIG['RAW_DATA_DIR']):
        processed_files = {f['original_filename'] for f in all_files}
        
        for filename in os.listdir(CONFIG['RAW_DATA_DIR']):
            if filename.endswith('.csv') and filename.startswith('event_'):
                if filename not in processed_files:
                    filepath = os.path.join(CONFIG['RAW_DATA_DIR'], filename)
                    try:
                        modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        all_files.append({
                            'filename': filename,
                            'filepath': filepath,
                            'modified_datetime': modified_time,
                            'type': 'RAW_DATA',
                            'original_filename': filename
                        })
                    except:
                        continue
    
    # 중복 제거 - 같은 original_filename을 가진 파일들 중 최신 것만 유지
    unique_files = {}
    for file_info in all_files:
        original_name = file_info['original_filename']
        if original_name not in unique_files:
            unique_files[original_name] = file_info
        else:
            # 더 최신 파일로 교체 (수정 시간 기준)
            if file_info['modified_datetime'] > unique_files[original_name]['modified_datetime']:
                unique_files[original_name] = file_info
    
    # 발생시간 기준으로 정렬 (최신 이벤트가 위로)
    def get_event_datetime(file_info):
        try:
            parsed_info = parse_filename_info(file_info['filename'])
            # 발생시간을 datetime 객체로 변환
            return datetime.strptime(parsed_info['datetime_str'], '%Y-%m-%d %H:%M:%S')
        except:
            # 파싱 실패시 수정 시간으로 대체
            return file_info['modified_datetime']
    
    final_files = list(unique_files.values())
    final_files.sort(key=get_event_datetime, reverse=True)
    return final_files

@st.cache_data(ttl=10)  # 10초 캐시
def get_earthquake_alerts():
    """지진 경보 로그 가져오기"""
    alerts = []
    
    if os.path.exists(CONFIG['ALERTS_DIR']):
        for filename in os.listdir(CONFIG['ALERTS_DIR']):
            if filename.endswith('.json') and filename.startswith('earthquake_alert_'):
                filepath = os.path.join(CONFIG['ALERTS_DIR'], filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        alert_data = json.load(f)
                    
                    alert_data['filename'] = filename
                    alert_data['filepath'] = filepath
                    alerts.append(alert_data)
                except:
                    continue
    
    # 시간순 정렬 (최신 순)
    alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return alerts

def parse_filename_info(filename):
    """지진 오보 분석 시스템 파일명에서 정보 추출"""
    try:
        # new2_ai_6060_2025-06-28_20-49-01.csv 또는 event_6060_2025-06-28_20-49-01.csv
        parts = filename.replace('.csv', '').split('_')
        
        if filename.startswith('new2_ai_'):
            port = parts[2]
            date_str = parts[3]
            time_str = parts[4]
        elif filename.startswith('ai_'):
            port = parts[1]
            date_str = parts[2]
            time_str = parts[3]
        else:  # event_
            port = parts[1]
            date_str = parts[2]
            time_str = parts[3]
        
        datetime_str = f"{date_str} {time_str.replace('-', ':')}"
        
        return {
            'port': port,
            'datetime_str': datetime_str,
            'location': f"센서_{port}"
        }
    except:
        return {
            'port': 'Unknown',
            'datetime_str': 'Unknown',
            'location': 'Unknown'
        }



# =========================== 사이드바 ===========================

def render_system_sidebar():
    """지진 오보 분석 시스템 사이드바"""
    st.sidebar.title("🌍 지진 오보 분석")
    st.sidebar.markdown("*지진 오보 분석 시스템*")
    
    # 삭제 확인 팝업 처리 (사이드바에서)
    delete_modal_active = False
    for idx in range(100):  # 최대 100개 확인
        delete_modal_key = f'show_delete_modal_{idx}'
        if st.session_state.get(delete_modal_key, False):
            delete_modal_active = True
            delete_file_info_key = f'delete_file_info_{idx}'
            file_info = st.session_state.get(delete_file_info_key, {})
            
            st.sidebar.markdown("---")
            st.sidebar.error("🚨 **삭제 확인 필요**")
            
            # 삭제 확인 팝업을 사이드바에서 처리
            with st.sidebar.container():
                st.markdown("### 🗑️ 파일 삭제")
                st.error("⚠️ **이 파일을 삭제하시겠습니까?**")
                
                # 파일 정보 표시
                st.info(f"**📁 파일:** `{file_info.get('filename', 'Unknown')}`")
                st.info(f"**📍 위치:** {file_info.get('location', 'Unknown')}")
                st.info(f"**🔍 결과:** {file_info.get('analysis_result', 'Unknown')}")
                
                filepath = file_info.get('filepath', '')
                if os.path.exists(filepath):
                    try:
                        file_size_kb = os.path.getsize(filepath) / 1024
                        st.info(f"**📊 크기:** {file_size_kb:.1f}KB")
                    except:
                        st.info(f"**📊 크기:** Unknown")
                else:
                    st.warning("**⚠️ 파일이 존재하지 않음**")
                
                st.error("**⚠️ 되돌릴 수 없습니다!**")
                
                # 삭제 확인 버튼들
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ 삭제", key=f"sidebar_confirm_delete_{idx}", use_container_width=True, type="primary"):
                        if os.path.exists(filepath):
                            try:
                                # 파일 삭제 실행
                                os.remove(filepath)
                                st.session_state[delete_modal_key] = False
                                st.session_state[f'delete_success_{idx}'] = f"✅ {file_info.get('filename', '')} 파일이 삭제되었습니다."
                                st.cache_data.clear()
                                safe_rerun()
                            except Exception as e:
                                st.error(f"❌ 삭제 실패: {str(e)}")
                                st.session_state[delete_modal_key] = False
                                safe_rerun()
                        else:
                            st.warning("⚠️ 파일이 존재하지 않습니다.")
                            st.session_state[delete_modal_key] = False
                            safe_rerun()
                
                with col2:
                    if st.button("❌ 취소", key=f"sidebar_cancel_delete_{idx}", use_container_width=True):
                        st.session_state[delete_modal_key] = False
                        safe_rerun()
            
            st.sidebar.markdown("---")
            break  # 하나의 삭제 팝업만 처리
    
    # 수정 확인 팝업 처리 (사이드바에서)
    modify_modal_active = False
    if not delete_modal_active:  # 삭제 팝업이 없을 때만 수정 팝업 처리
        for idx in range(100):  # 최대 100개 확인
            modify_modal_key = f'show_modify_modal_{idx}'
            if st.session_state.get(modify_modal_key, False):
                modify_modal_active = True
                modify_file_info_key = f'modify_file_info_{idx}'
                file_info = st.session_state.get(modify_file_info_key, {})
                
                st.sidebar.markdown("---")
                st.sidebar.warning("✏️ **분석 결과 수정**")
                
                # 수정 팝업을 사이드바에서 처리
                with st.sidebar.container():
                    st.markdown("### ✏️ 결과 수정")
                    st.info("**올바른 분류를 선택하세요**")
                    
                    # 파일 정보 표시
                    st.info(f"**📁 파일:** `{file_info.get('filename', 'Unknown')}`")
                    current_result = file_info.get('current_result', 'Unknown')
                    st.info(f"**현재:** {current_result}")
                    
                    # 현재 선택된 클래스 확인 (기본값 설정)
                    if '🔴' in current_result or '지진' in current_result:
                        default_selection = 0
                    elif '🟠' in current_result or '산업진동' in current_result:
                        default_selection = 1
                    elif '🟢' in current_result or '생활진동' in current_result:
                        default_selection = 2
                    else:
                        default_selection = 0
                    
                    # 라디오 버튼으로 클래스 선택 (사이드바용 간단 버전)
                    selected_class = st.radio(
                        "분류 선택:",
                        options=[0, 1, 2],
                        format_func=lambda x: "🔴 지진" if x == 0 else "🟠 규칙적산업진동" if x == 1 else "🟢 불규칙생활진동",
                        index=default_selection,
                        key=f"sidebar_radio_class_{idx}"
                    )
                    
                    st.warning("**전문가 수정으로 기록됩니다**")
                    
                    # 저장 및 취소 버튼
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 저장", key=f"sidebar_save_correction_{idx}", use_container_width=True, type="primary"):
                            class_names = {0: "지진", 1: "규칙적산업진동", 2: "불규칙생활진동"}
                            class_name = class_names[selected_class]
                            
                            # 수정 사항 저장
                            if save_correction(file_info, selected_class, class_name):
                                # 모달 닫기
                                st.session_state[modify_modal_key] = False
                                # 성공 메시지 설정
                                st.session_state[f'correction_success_{idx}'] = f"✅ {class_name}으로 수정 완료!"
                                # 캐시 클리어하고 즉시 새로고침
                                st.cache_data.clear()
                                safe_rerun()
                    
                    with col2:
                        if st.button("❌ 취소", key=f"sidebar_cancel_modify_{idx}", use_container_width=True):
                            st.session_state[modify_modal_key] = False
                            safe_rerun()
                
                st.sidebar.markdown("---")
                break  # 하나의 수정 팝업만 처리
    
    # 삭제 또는 수정 팝업이 활성화되어 있으면 다른 사이드바 내용은 숨김
    if delete_modal_active or modify_modal_active:
        return
    
    # 새로고침 버튼
    st.sidebar.subheader("🔄 데이터 새로고침")
    if st.sidebar.button("🔄 새로고침", use_container_width=True, type="primary"):
        st.cache_data.clear()
        safe_rerun()
    
    # 현재 상태 표시
    try:
        analysis_dir = CONFIG['ANALYSIS_DIR']
        if os.path.exists(analysis_dir):
            total_files = len([f for f in os.listdir(analysis_dir) if f.endswith('.csv') and f.startswith('new2_ai_')])
            st.sidebar.success(f"📊 총 이벤트: {total_files}개")
        else:
            st.sidebar.warning("📂 분석 폴더가 없습니다")
    except Exception as e:
        st.sidebar.error(f"❌ 상태 확인 오류: {e}")
    
    # 마지막 새로고침 시간
    current_time = datetime.now().strftime('%H:%M:%S')
    st.sidebar.caption(f"마지막 새로고침: {current_time}")
    
    st.sidebar.markdown("---")
    
    # 시스템 통계
    st.sidebar.subheader("📊 시스템 통계")
    
    current_files = get_system_event_files()
    current_alerts = get_earthquake_alerts()
    
    # 클래스별 카운트 계산
    class_counts = {'지진': 0, '규칙적산업진동': 0, '불규칙생활진동': 0}
    
    for file_info in current_files:
        if file_info['type'] in ['SYSTEM_ANALYSIS', 'LEGACY_AI']:
            try:
                df = pd.read_csv(file_info['filepath'])
                
                # 전문가 수정이 있으면 수정된 클래스 사용
                if 'expert_corrected' in df.columns and df['expert_corrected'].iloc[0]:
                    corrected_class_name = df['expert_corrected_class_name'].iloc[0]
                    if corrected_class_name in class_counts:
                        class_counts[corrected_class_name] += 1
                # 아니면 AI 분석 결과 사용
                else:
                    # AI 분석 결과 사용 (컴럼명에 따라 다른 처리)
                    if 'ai_predicted_class' in df.columns:
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
            except Exception as e:
                # 디버깅을 위해 오류 정보 추가 (필요시 사용)
                # st.sidebar.error(f"데이터 읽기 오류: {file_info['filename']} - {str(e)}")
                continue
    
    st.sidebar.metric("총 이벤트", len(current_files))
    st.sidebar.metric("🔴 지진", class_counts['지진'])
    st.sidebar.metric("🟠 규칙적 산업진동", class_counts['규칙적산업진동'])
    st.sidebar.metric("🟢 불규칙 생활진동", class_counts['불규칙생활진동'])
    
    # 최근 이벤트
    if current_files:
        latest_file = current_files[0]
        time_diff = datetime.now() - latest_file['modified_datetime']
        
        if time_diff.total_seconds() < 60:
            st.sidebar.metric("최근 이벤트", f"{int(time_diff.total_seconds())}초 전")
        elif time_diff.total_seconds() < 3600:
            st.sidebar.metric("최근 이벤트", f"{int(time_diff.total_seconds()//60)}분 전")
        else:
            st.sidebar.metric("최근 이벤트", f"{int(time_diff.total_seconds()//3600)}시간 전")
    
    # 재학습 관리 섹션
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 AI 모델 재학습")
    
    if RETRAINING_AVAILABLE:
        render_retraining_sidebar()
    else:
        st.sidebar.warning("⚠️ 재학습 시스템을 사용할 수 없습니다")

# =========================== 재학습 관리 ===========================

@st.cache_resource
def get_retraining_system():
    """재학습 시스템 인스턴스 가져오기"""
    if not RETRAINING_AVAILABLE:
        return None
    
    # Windows 환경용 경로 정규화
    def normalize_path_for_retraining(path):
        """Windows 환경에서 재학습 시스템용 경로 정규화"""
        import platform
        
        if platform.system() == "Windows":
            # Windows 환경에서는 Windows 경로 스타일 유지
            if path.startswith('/mnt/c/'):
                # WSL 경로를 Windows 경로로 변환
                normalized = path.replace('/mnt/c/', 'C:/')
                normalized = normalized.replace('/', '\\')
                return normalized
            elif path.startswith('C:/'):
                # 슬래시를 백슬래시로 변환
                return path.replace('/', '\\')
            return path
        else:
            # Linux/WSL 환경에서는 원래 로직
            if path.startswith('C:\\') or path.startswith('C:/'):
                normalized = path.replace('C:\\', '/mnt/c/').replace('C:/', '/mnt/c/')
                normalized = normalized.replace('\\', '/')
                return normalized
            return path
    
    config = {
        'base_model_path': 'new2_convlstm_3class_best.h5',
        'retraining_data_dir': 'retraining_data',
        'retrained_models_dir': 'retrained_models',
        'analysis_dirs': [
            normalize_path_for_retraining(CONFIG['ANALYSIS_DIR']),
            normalize_path_for_retraining(CONFIG['RAW_DATA_DIR']),
            normalize_path_for_retraining(CONFIG['PROCESSED_DATA_DIR']),
            normalize_path_for_retraining("/mnt/c/earthquake_project"),  # 현재 프로젝트 폴더도 포함
            # Windows 환경에서 직접 경로도 추가
            "C:\\earthquake_project\\influxLogs\\new2_analysis",
            "C:\\earthquake_project\\influxLogs\\base"
        ]
    }
    
    return NEW2RetrainingSystem(config)

def count_retraining_status():
    """재학습 완료/미완료 파일 카운팅"""
    completed_count = 0
    pending_count = 0
    
    try:
        analysis_dir = CONFIG['ANALYSIS_DIR']
        if os.path.exists(analysis_dir):
            for filename in os.listdir(analysis_dir):
                if filename.endswith('.csv') and filename.startswith('new2_ai_'):
                    filepath = os.path.join(analysis_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                        
                        # 재학습 완료 여부 확인 (retraining_completed 컬럼 기준)
                        if 'retraining_completed' in df.columns:
                            # 다양한 형태의 True 값 처리 (불린, 문자열, 숫자)
                            retraining_value = df['retraining_completed'].iloc[0]
                            
                            # 디버깅용 출력
                            # print(f"파일 {filename}: retraining_value = {repr(retraining_value)} (타입: {type(retraining_value)})")
                            
                            # True로 간주할 조건들
                            is_completed = (
                                retraining_value is True or  # 불린 True
                                retraining_value == 'True' or  # 문자열 'True'
                                str(retraining_value).lower() == 'true' or  # 대소문자 무관 'true'
                                (isinstance(retraining_value, (int, float)) and retraining_value == 1)  # 숫자 1
                            )
                            
                            if is_completed:
                                completed_count += 1
                            else:
                                pending_count += 1
                                # 잘못된 값이 있는 파일 확인용
                                print(f"미완료로 분류: {filename} - 값: {repr(retraining_value)}")
                        else:
                            pending_count += 1
                            print(f"컬럼 없음: {filename}")
                            
                    except Exception as e:
                        print(f"파일 처리 오류 {filename}: {e}")
                        pending_count += 1  # 읽기 실패한 파일은 미완료로 간주
                        
    except Exception as e:
        print(f"재학습 상태 카운팅 오류: {e}")
    
    return completed_count, pending_count

def get_retraining_status_for_file(filepath):
    """개별 파일의 재학습 상태 확인"""
    try:
        df = pd.read_csv(filepath)
        
        # 재학습 완료 여부 확인
        if 'retraining_completed' in df.columns:
            # 다양한 형태의 True 값 처리 (불린, 문자열, 숫자)
            retraining_value = df['retraining_completed'].iloc[0]
            
            # True로 간주할 조건들
            is_completed = (
                retraining_value is True or  # 불린 True
                retraining_value == 'True' or  # 문자열 'True'
                str(retraining_value).lower() == 'true' or  # 대소문자 무관 'true'
                (isinstance(retraining_value, (int, float)) and retraining_value == 1)  # 숫자 1
            )
            
            return "완료" if is_completed else "미완료"
        else:
            return "미완료"
            
    except Exception:
        return "미완료"

def mark_all_files_as_retrained():
    """모든 분석 파일을 재학습 완료 상태로 업데이트"""
    try:
        analysis_dir = CONFIG['ANALYSIS_DIR']
        if os.path.exists(analysis_dir):
            updated_count = 0
            
            for filename in os.listdir(analysis_dir):
                if filename.endswith('.csv') and filename.startswith('new2_ai_'):
                    filepath = os.path.join(analysis_dir, filename)
                    try:
                        df = pd.read_csv(filepath)
                        
                        # retraining_completed 컬럼 강제로 True로 설정 (기존 잘못된 값 덮어쓰기)
                        df['retraining_completed'] = True
                        
                        # 재학습 완료 시간 기록
                        df['retraining_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # 파일 저장 (UTF-8 인코딩, CSV 포맷 보장)
                        df.to_csv(filepath, index=False, encoding='utf-8-sig', 
                                quoting=1, escapechar='\\')  # QUOTE_ALL 사용
                        updated_count += 1
                        print(f"업데이트 완료: {filename}")
                        
                    except Exception as e:
                        print(f"파일 업데이트 실패 {filename}: {e}")
                        continue
            
            print(f"재학습 완료 상태 업데이트: {updated_count}개 파일")
            return updated_count
            
    except Exception as e:
        print(f"재학습 상태 업데이트 오류: {e}")
        return 0

def render_retraining_sidebar():
    """재학습 관리 사이드바"""
    # 재학습 완료/미완료 카운팅
    completed_count, pending_count = count_retraining_status()
    
    # 상태 표시
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("완료", f"{completed_count}개", delta="수정됨")
    with col2:
        st.metric("미완료", f"{pending_count}개", delta="대기중")
    
    # 재학습 버튼
    st.sidebar.markdown("---")
    
    # 재학습 진행 상태 확인
    if 'retraining_in_progress' not in st.session_state:
        st.session_state.retraining_in_progress = False
    
    if st.session_state.retraining_in_progress:
        st.sidebar.warning("🧠 재학습 진행 중...")
        st.sidebar.info("재학습이 완료될 때까지 기다려주세요.")
        
        # 재학습 완료 체크 (실제 구현에서는 백그라운드 작업 상태 확인)
        if st.sidebar.button("✅ 재학습 완료", use_container_width=True):
            st.session_state.retraining_in_progress = False
            st.sidebar.success("🎉 재학습이 완료되었습니다!")
            safe_rerun()
    
    else:
        if pending_count > 0:
            if st.sidebar.button(f"🧠 재학습 시작 ({pending_count}개)", 
                               type="primary", 
                               use_container_width=True,
                               help=f"미완료 {pending_count}개 파일로 모델 재학습을 시작합니다"):
                
                # 재학습 시작
                st.session_state.retraining_in_progress = True
                st.sidebar.success("🚀 재학습을 시작합니다!")
                
                # 실제 재학습 로직 호출 (백그라운드에서 실행)
                try:
                    retraining_system = get_retraining_system()
                    if retraining_system:
                        # 여기서 실제 재학습 실행
                        result = retraining_system.run_full_retraining_pipeline(min_corrections=1)
                        
                        if result['success']:
                            st.sidebar.success(f"✅ {result['message']}")
                            if result.get('new_model_path'):
                                st.sidebar.info(f"📁 새 모델: {os.path.basename(result['new_model_path'])}")
                            # 재학습 시스템에서 자동으로 파일 상태 업데이트됨
                            
                        else:
                            st.sidebar.error(f"❌ {result['message']}")
                        
                        st.session_state.retraining_in_progress = False
                    
                except Exception as e:
                    st.sidebar.error(f"❌ 재학습 오류: {str(e)}")
                    st.session_state.retraining_in_progress = False
                
                safe_rerun()
        else:
            st.sidebar.info("📋 재학습할 파일이 없습니다")
            st.sidebar.caption("전문가 수정이 필요합니다")
    
    # 재학습 시스템 정보
    retraining_system = get_retraining_system()
    if retraining_system:
        try:
            status = retraining_system.get_retraining_status()
            
            st.sidebar.markdown("---")
            st.sidebar.caption("📊 재학습 이력")
            st.sidebar.caption(f"• 총 재학습 횟수: {status.get('retraining_count', 0)}회")
            
            if status.get('last_retraining'):
                try:
                    last_time = datetime.strptime(status['last_retraining'], '%Y%m%d_%H%M%S')
                    time_diff = datetime.now() - last_time
                    if time_diff.days > 0:
                        time_text = f"{time_diff.days}일 전"
                    else:
                        time_text = f"{int(time_diff.total_seconds()//3600)}시간 전"
                    st.sidebar.caption(f"• 마지막 재학습: {time_text}")
                except:
                    st.sidebar.caption("• 마지막 재학습: 오류")
            else:
                st.sidebar.caption("• 마지막 재학습: 없음")
                
        except Exception as e:
            st.sidebar.caption(f"재학습 정보 오류: {str(e)}")
    
    # 디버깅 정보 추가
    st.sidebar.markdown("---")
    st.sidebar.caption("🔍 디버깅 정보")
    st.sidebar.caption(f"• 완료 파일: {completed_count}개")
    st.sidebar.caption(f"• 미완료 파일: {pending_count}개")
    if pending_count > 0:
        st.sidebar.caption(f"• 재학습 가능: ✅ 예")
    else:
        st.sidebar.caption(f"• 재학습 가능: ❌ 아니오")
    
    # 수동 상태 업데이트 버튼 (재학습은 했지만 상태가 업데이트 안된 경우)
    if pending_count > 0:
        st.sidebar.markdown("---")
        st.sidebar.warning("⚠️ 일부 파일의 재학습 상태가 올바르지 않습니다.")
        if st.sidebar.button("🔄 재학습 완료 상태로 업데이트", 
                           use_container_width=True, 
                           type="secondary",
                           help="모든 파일을 재학습 완료 상태로 수동 업데이트"):
            updated_count = mark_all_files_as_retrained()
            if updated_count > 0:
                st.sidebar.success(f"✅ {updated_count}개 파일이 완료 상태로 업데이트되었습니다!")
                safe_rerun()
            else:
                st.sidebar.error("❌ 파일 업데이트에 실패했습니다.")

# =========================== 경보 대시보드 ===========================

def render_alert_dashboard():
    """지진 경보 전용 대시보드"""
    alerts = get_earthquake_alerts()
    
    if not alerts:
        st.info("🟢 현재 지진 경보가 없습니다. 시스템이 정상 작동 중입니다.")
        return
    
    st.subheader(f"🚨 지진 경보 현황 ({len(alerts)}건)")
    
    # 최근 경보 하이라이트
    latest_alert = alerts[0]
    alert_time = datetime.fromisoformat(latest_alert['timestamp'])
    time_since = datetime.now() - alert_time
    
    if time_since.total_seconds() < 300:  # 5분 이내
        st.error(f"🚨 **최신 지진 경보!** {time_since.seconds//60}분 {time_since.seconds%60}초 전")
    
    # 경보 목록
    for i, alert in enumerate(alerts[:5]):  # 최근 5개만 표시
        with st.expander(f"🚨 경보 #{i+1} - {alert.get('event_time', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**경보 시간:** {alert.get('timestamp', 'Unknown')}")
                st.write(f"**이벤트 시간:** {alert.get('event_time', 'Unknown')}")
                st.write(f"**센서 포트:** {alert.get('port', 'Unknown')}")
                
            with col2:
                if 'ai_analysis' in alert:
                    ai_data = alert['ai_analysis']
                    st.write(f"**지진 확률:** {ai_data.get('earthquake_prob', 0)*100:.1f}%")
                    st.write(f"**최종 신뢰도:** {ai_data.get('final_confidence', 0)*100:.1f}%")
                    st.write(f"**분류:** {ai_data.get('final_class_name', 'Unknown')}")
                
            with col3:
                if 'model_info' in alert:
                    model_info = alert['model_info']
                    st.write(f"**모델:** {model_info.get('name', 'Unknown')}")
                    st.write(f"**모델 정확도:** {model_info.get('accuracy', 0)*100:.2f}%")
                
                st.write(f"**진도:** {alert.get('intensity', 'Unknown')}")

# =========================== 메인 대시보드 ===========================

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
                fig_mag = go.Figure()
                fig_mag.add_trace(go.Scatter(
                    x=list(time_axis),
                    y=magnitude_data.tolist(),
                    name='진도',
                    line=dict(color='#9C27B0', width=2),
                    mode='lines'
                ))
                fig_mag.update_layout(
                    title="센서 - 진도 (0.00~10.00)",
                    xaxis_title="시간 (샘플)",
                    yaxis_title="진도",
                    yaxis=dict(range=[0, 10]),  # Y축 범위 0~10 고정
                    xaxis=dict(range=[0, 4000]),  # X축 범위 0~4000 고정
                    height=300,
                    margin=dict(t=60, b=40, l=60, r=40)
                )
                st.plotly_chart(fig_mag, use_container_width=True)
            
            return True
        else:
            # 센서 데이터를 찾지 못한 경우
            st.warning("📊 표시할 센서 데이터가 없습니다.")
            
            # 사용 가능한 칼럼들 일부 표시
            numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] or col.lower() in ['x', 'y', 'z']]
            if numeric_columns:
                st.info(f"사용 가능한 숫자 칼럼들: {', '.join(numeric_columns[:10])}")
            return False
            
    except Exception as e:
        st.error(f"❌ 파형 데이터 로딩 오류: {str(e)}")
        return False

def render_system_dashboard():
    """지진 오보 분석 시스템 메인 대시보드"""
    st.title("🌍 지진 오보 분석 시스템")
    st.markdown("### ConvLSTM 기반 실시간 모니터링")
    
    # 수동 새로고침 안내
    st.info("🔄 새로운 이벤트 확인을 위해 사이드바의 **새로고침** 버튼을 클릭해주세요.")
    
    # 재학습 완료 알림 처리
    if st.session_state.get('retraining_success', False):
        result = st.session_state.get('retraining_result', {})
        st.success(f"🧠 **AI 모델 재학습 완료!** {result.get('message', '')}")
        if result.get('new_model_path'):
            model_name = os.path.basename(result['new_model_path'])
            performance = result.get('performance', {})
            # 재학습 완료 정보는 사이드바에서만 표시하므로 메인 화면에서는 숨김
            # col_success1, col_success2 = st.columns(2)
            # with col_success1:
            #     st.info(f"📁 **새 모델:** {model_name}")
            # with col_success2:
            #     if performance:
            #         accuracy = performance.get('val_accuracy', 0)
            #         samples = performance.get('samples_count', 0)
            #         st.info(f"📊 **성능:** 정확도 {accuracy:.1%}, 샘플 {samples}개")
        
        # 알림 표시 후 제거
        del st.session_state['retraining_success']
        if 'retraining_result' in st.session_state:
            del st.session_state['retraining_result']
    
    # 재학습 오류 알림 처리
    if st.session_state.get('retraining_error'):
        st.error(f"❌ **재학습 실패:** {st.session_state['retraining_error']}")
        del st.session_state['retraining_error']
    
    # 선택된 파일 상태 초기화
    if 'selected_file_for_waveform' not in st.session_state:
        st.session_state.selected_file_for_waveform = None
    
    # 경보 상태 표시
    alerts = get_earthquake_alerts()
    if alerts:
        latest_alert = alerts[0]
        alert_time = datetime.fromisoformat(latest_alert['timestamp'])
        time_since = datetime.now() - alert_time
        
        if time_since.total_seconds() < 600:  # 10분 이내
            st.error(f"🚨 **활성 지진 경보** - {time_since.seconds//60}분 {time_since.seconds%60}초 전 발령")
            render_alert_dashboard()
            st.markdown("---")
    
    # 이벤트 파일 가져오기
    files = get_system_event_files()
    
    if not files:
        st.info("📁 아직 저장된 이벤트 데이터가 없습니다.")
        st.info("influx_new2_realtime.py가 실행되면 자동으로 데이터가 수집됩니다.")
        
        # 디버깅 정보 추가
        with st.expander("🔍 디버깅 정보 (경로 확인)", expanded=False):
            import platform
            st.write(f"**실행 환경:** {platform.system()}")
            st.write(f"**현재 작업 디렉토리:** {os.getcwd()}")
            st.write(f"**스크립트 위치:** {os.path.dirname(os.path.abspath(__file__))}")
            st.write(f"**BASE_PATH:** {BASE_PATH}")
            st.write(f"**ANALYSIS_DIR:** {CONFIG['ANALYSIS_DIR']}")
            st.write(f"**ANALYSIS_DIR 존재:** {os.path.exists(CONFIG['ANALYSIS_DIR'])}")
            
            if os.path.exists(CONFIG['ANALYSIS_DIR']):
                analysis_files = [f for f in os.listdir(CONFIG['ANALYSIS_DIR']) if f.endswith('.csv')]
                st.write(f"**분석 디렉토리 파일 수:** {len(analysis_files)}개")
                new2_files = [f for f in analysis_files if f.startswith('new2_ai_')]
                st.write(f"**new2_ai_ 파일 수:** {len(new2_files)}개")
                if new2_files:
                    st.write("**new2_ai_ 파일들:**")
                    for f in new2_files[:5]:  # 최대 5개만 표시
                        st.write(f"  - {f}")
        
        # 폴더 상태 확인
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📂 원시 데이터:**")
            if os.path.exists(CONFIG['RAW_DATA_DIR']):
                raw_files = [f for f in os.listdir(CONFIG['RAW_DATA_DIR']) if f.endswith('.csv')]
                st.write(f"✅ {len(raw_files)}개 파일")
            else:
                st.write("❌ 폴더 없음")
        
        with col2:
            st.write("**🧠 지진 오보 분석:**")
            if os.path.exists(CONFIG['ANALYSIS_DIR']):
                system_files = [f for f in os.listdir(CONFIG['ANALYSIS_DIR']) if f.endswith('.csv')]
                st.write(f"✅ {len(system_files)}개 파일")
            else:
                st.write("❌ 폴더 없음")
        
        with col3:
            st.write("**🚨 지진 경보:**")
            if os.path.exists(CONFIG['ALERTS_DIR']):
                alert_files = [f for f in os.listdir(CONFIG['ALERTS_DIR']) if f.endswith('.json')]
                st.write(f"✅ {len(alert_files)}개 파일")
            else:
                st.write("❌ 폴더 없음")
        
        # 이벤트 기반 자동 새로고침
        if should_auto_refresh():
            st.cache_data.clear()
            safe_rerun()
        elif st.session_state.get('auto_refresh', True):
            # 변경이 없으면 짧은 대기 후 다시 확인
            time.sleep(2)
            safe_rerun()
        return
    
    # =================== 성능 통계 삭제됨 ==================="
    
    # 통계 계산 - 삭제됨
    # total_events = len(files)
#     new2_processed = len([f for f in files if f['type'] == 'NEW2_ANALYSIS'])
#     legacy_processed = len([f for f in files if f['type'] == 'LEGACY_AI'])
#     raw_only = len([f for f in files if f['type'] == 'RAW_DATA'])
#     
#     class_counts = {name: 0 for name in NEW2_CLASS_NAMES.values()}
#     earthquake_alerts = 0
#     total_confidence = 0
#     confidence_count = 0
#     
#     for file_info in files:
#         if file_info['type'] in ['NEW2_ANALYSIS', 'LEGACY_AI']:
#             try:
#                 df = pd.read_csv(file_info['filepath'])
#                 
#                 # NEW2 결과 확인
#                 if file_info['type'] == 'NEW2_ANALYSIS':
#                     if 'ai_predicted_class' in df.columns:
#                         pred_class = int(df['ai_predicted_class'].iloc[0])
#                         if pred_class in NEW2_CLASS_NAMES:
#                             class_counts[NEW2_CLASS_NAMES[pred_class]] += 1
#                         
#                         if 'ai_final_confidence' in df.columns:
#                             confidence = float(df['ai_final_confidence'].iloc[0])
#                             total_confidence += confidence
#                             confidence_count += 1
#                         
#                         if 'ai_is_earthquake' in df.columns:
#                             is_earthquake = df['ai_is_earthquake'].iloc[0]
#                             if str(is_earthquake).lower() == 'true':
#                                 earthquake_alerts += 1
#                 
#                 # 레거시 결과 확인 (호환성)
#                 elif 'predicted_class' in df.columns:
#                     pred_class = int(df['predicted_class'].iloc[0])
#                     if pred_class in NEW2_CLASS_NAMES:
#                         class_counts[NEW2_CLASS_NAMES[pred_class]] += 1
#                     
#                     if pred_class == 0:  # 지진
#                         earthquake_alerts += 1
#                         
#             except:
#                 continue
#     
#     # 성능 지표 표시
#     col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
#     
#     with col_perf1:
#         st.metric(
#             label="📊 총 이벤트",
#             value=f"{total_events:,}",
#             delta=f"NEW2: {new2_processed}"
#         )
#     
#     with col_perf2:
#         processing_rate = (new2_processed / total_events * 100) if total_events > 0 else 0
#         st.metric(
#             label="🧠 NEW2 처리율",
#             value=f"{processing_rate:.1f}%",
#             delta=f"{new2_processed}/{total_events}"
#         )
#     
#     with col_perf3:
#         avg_confidence = (total_confidence / confidence_count * 100) if confidence_count > 0 else 0
#         st.metric(
#             label="📈 평균 신뢰도",
#             value=f"{avg_confidence:.1f}%",
#             delta="NEW2 AI 분석"
#         )
#     
#     with col_perf4:
#         alert_rate = (earthquake_alerts / (new2_processed + legacy_processed) * 100) if (new2_processed + legacy_processed) > 0 else 0
#         st.metric(
#             label="🚨 지진 경보율",
#             value=f"{alert_rate:.1f}%",
#             delta=f"{earthquake_alerts}건 발령"
#         )
#     
    # =================== 클래스 분포 시각화 ===================
    
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
    
    # =================== 실시간 이벤트 모니터링 및 센서 3축 가속도 파형 ===================
    st.markdown("---")
    st.subheader("📈 실시간 이벤트 모니터링 및 센서 3축 가속도 파형")
    
    # 최근 이벤트 분석 및 파형 표시
    if files:
        # 선택된 파일이 없으면 최신 파일 선택
        if st.session_state.selected_file_for_waveform is None:
            st.session_state.selected_file_for_waveform = files[0]
        
        # 현재 표시할 파일 결정
        display_file = st.session_state.selected_file_for_waveform
        
        try:
            df_display = pd.read_csv(display_file['filepath'])
            parsed_info = parse_filename_info(display_file['filename'])
            
            # 파일 정보 요약
            time_diff = datetime.now() - display_file['modified_datetime']
            time_text = f"{int(time_diff.total_seconds())}초 전" if time_diff.total_seconds() < 60 else f"{int(time_diff.total_seconds()//60)}분 전"
            
            # 분석 결과 추출
            analysis_result = "처리 대기"
            confidence_text = ""
            result_color = "info"
            
            if display_file['type'] == 'SYSTEM_ANALYSIS':
                # 지진 오보 분석 시스템 결과
                possible_class_cols = ['ai_final_class', 'ai_predicted_class', 'predicted_class']
                possible_conf_cols = ['ai_final_confidence', 'ai_confidence', 'confidence']
                
                pred_class = None
                confidence = 0.0
                
                # 클래스 찾기
                for col in possible_class_cols:
                    if col in df_display.columns:
                        pred_class = int(df_display[col].iloc[0])
                        break
                
                # 신뢰도 찾기
                for col in possible_conf_cols:
                    if col in df_display.columns:
                        confidence = float(df_display[col].iloc[0])
                        break
                
                if pred_class is not None:
                    class_name = SYSTEM_CLASS_NAMES.get(pred_class, 'Unknown')
                    class_icon = SYSTEM_CLASS_COLORS.get(pred_class, '⚪')
                    analysis_result = f"{class_icon} {class_name}"
                    confidence_text = f"신뢰도: {confidence:.1%}"
                    result_color = "error" if pred_class == 0 else "warning" if pred_class == 1 else "success"
            
            # 파일 정보 표시 (신뢰도 제거, 간격 넓게)
            summary_text = f"**이벤트:** {parsed_info['location']}　　　　**데이터:** {len(df_display):,}행　　　　**분석:** {analysis_result}　　　　**발생시간:** {parsed_info['datetime_str']}"
            
            if result_color == "error":
                st.error(summary_text)
            elif result_color == "warning":
                st.warning(summary_text)
            elif result_color == "success":
                st.success(summary_text)
            else:
                st.info(summary_text)
            
            # 센서 3축 가속도 파형 표시
            try:
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
                    if all(col in df_display.columns for col in [x_test, y_test, z_test]):
                        x_col, y_col, z_col = x_test, y_test, z_test
                        sensor_found = True
                        break
                
                if sensor_found:
                    # 데이터 샘플링 (성능 최적화)
                    data_length = len(df_display)
                    if data_length > 4000:
                        sample_step = max(1, data_length // 4000)
                        sampled_data = df_display.iloc[::sample_step]
                    else:
                        sampled_data = df_display
                    
                    time_axis = range(len(sampled_data))
                    
                    # 데이터 유효성 검사
                    x_data = pd.to_numeric(sampled_data[x_col], errors='coerce').fillna(0)
                    y_data = pd.to_numeric(sampled_data[y_col], errors='coerce').fillna(0)
                    z_data = pd.to_numeric(sampled_data[z_col], errors='coerce').fillna(0)
                    
                    # Magnitude 계산
                    # 가속도 크기 계산
                    acceleration_magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
                    
                    # 진도 변환 (0.00~10.00 범위)
                    # 이벤트의 실제 진도와 맞추어 동적 스케일링
                    magnitude_data = np.zeros_like(acceleration_magnitude)
                    
                    # 이벤트의 실제 진도 정보 가져오기
                    actual_intensity = 5.0  # 기본값
                    try:
                        if 'intensity' in df_display.columns:
                            actual_intensity = float(df_display['intensity'].iloc[0])
                    except:
                        pass
                    
                    # 가속도의 최대값에 맞춰 진도 스케일링
                    max_acc = np.max(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 1.0
                    
                    for i, acc_val in enumerate(acceleration_magnitude):
                        if max_acc > 0:
                            # 실제 진도에 맞춰 정규화된 진도 계산
                            normalized_acc = acc_val / max_acc  # 0~1 정규화
                            intensity = normalized_acc * actual_intensity  # 실제 진도로 스케일링
                            magnitude_data[i] = np.clip(intensity, 0.0, 10.0)
                        else:
                            magnitude_data[i] = 0.0
                    
                    # 진도 정보
                    intensity_text = ""
                    if 'intensity' in df_display.columns:
                        try:
                            intensity = float(df_display['intensity'].iloc[0])
                            intensity_text = f" | 진도: {intensity:.2f}"
                        except:
                            pass
                    
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
                        fig_mag = go.Figure()
                        fig_mag.add_trace(go.Scatter(
                            x=list(time_axis),
                            y=magnitude_data.tolist(),
                            name='진도',
                            line=dict(color='#9C27B0', width=2),
                            mode='lines'
                        ))
                        fig_mag.update_layout(
                            title="센서 - 진도 (0.00~10.00)",
                            xaxis_title="시간 (샘플)",
                            yaxis_title="진도",
                            yaxis=dict(range=[0, 10]),  # Y축 범위 0~10 고정
                            xaxis=dict(range=[0, 4000]),  # X축 범위 0~4000 고정
                            height=300,
                            margin=dict(t=60, b=40, l=60, r=40)
                        )
                        st.plotly_chart(fig_mag, use_container_width=True)
                else:
                    st.warning("📊 표시할 센서 데이터가 없습니다.")
                        
            except Exception as sensor_error:
                st.error(f"센서 파형 표시 오류: {sensor_error}")
        
        except Exception as e:
            st.error(f"❌ 파일 처리 오류: {e}")
        
        # =================== 이벤트 목록 (테이블 형태 + 페이지네이션) ===================
        st.markdown("---")
        st.subheader("📋 최근 이벤트 목록")
        
        # 페이지네이션 설정
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0
        
        items_per_page = st.selectbox("페이지당 표시 개수", [10, 20, 50, 100], index=0)
        total_pages = max(1, (len(files) + items_per_page - 1) // items_per_page)
        
        # 현재 페이지 데이터 추출
        start_idx = st.session_state.page_number * items_per_page
        end_idx = min(start_idx + items_per_page, len(files))
        current_page_files = files[start_idx:end_idx]
        
        
        # 테이블 데이터 준비
        table_data = []
        
        for i, file_info in enumerate(current_page_files):
            try:
                df = pd.read_csv(file_info['filepath'])
                parsed_info = parse_filename_info(file_info['filename'])
                
                # 이벤트 타입별 아이콘
                type_icons = {
                    'SYSTEM_ANALYSIS': '🧠 시스템',
                    'LEGACY_AI': '🔄 Legacy', 
                    'RAW_DATA': '📁 Raw'
                }
                
                # 분석 결과 확인 (파일의 전문가 수정 우선 적용)
                analysis_result = "⏳ 처리 대기"
                confidence_value = 0.0
                is_earthquake = False
                is_expert_corrected = False
                
                # 전문가 수정 여부 확인 (파일 기반)
                if 'expert_corrected' in df.columns and df['expert_corrected'].iloc[0]:
                    try:
                        corrected_class = int(df['expert_corrected_class'].iloc[0])
                        corrected_class_name = df['expert_corrected_class_name'].iloc[0]
                        
                        class_icon = SYSTEM_CLASS_COLORS.get(corrected_class, '⚪')
                        analysis_result = f"{class_icon} {corrected_class_name} ✏️"  # 수정 표시
                        is_earthquake = (corrected_class == 0)
                        is_expert_corrected = True
                        confidence_value = 1.0  # 전문가 수정은 100% 신뢰도
                    except:
                        pass
                
                # 전문가 수정이 없으면 기존 AI 분석 결과 사용
                if not is_expert_corrected and file_info['type'] == 'SYSTEM_ANALYSIS':
                    # 지진 오보 분석 시스템 결과 확인 (다양한 칼럼명 지원)
                    possible_class_cols = ['ai_final_class', 'ai_predicted_class', 'predicted_class']
                    possible_conf_cols = ['ai_final_confidence', 'ai_confidence', 'confidence']
                    possible_name_cols = ['ai_final_class_name', 'ai_class_name', 'class_name']
                    
                    pred_class = None
                    confidence_value = 0.0
                    class_name = "Unknown"
                    
                    # 클래스 찾기
                    for col in possible_class_cols:
                        if col in df.columns:
                            pred_class = int(df[col].iloc[0])
                            break
                    
                    # 신뢰도 찾기
                    for col in possible_conf_cols:
                        if col in df.columns:
                            confidence_value = float(df[col].iloc[0])
                            break
                    
                    # 클래스명 찾기
                    for col in possible_name_cols:
                        if col in df.columns:
                            class_name = df[col].iloc[0]
                            break
                    
                    if pred_class is not None:
                        if class_name == "Unknown" and pred_class in SYSTEM_CLASS_NAMES:
                            class_name = SYSTEM_CLASS_NAMES[pred_class]
                        
                        class_icon = SYSTEM_CLASS_COLORS.get(pred_class, '⚪')
                        analysis_result = f"{class_icon} {class_name}"
                        is_earthquake = (pred_class == 0)
                    
                elif file_info['type'] == 'LEGACY_AI' and 'predicted_class' in df.columns:
                    pred_class = int(df['predicted_class'].iloc[0])
                    class_name = SYSTEM_CLASS_NAMES.get(pred_class, 'Unknown')
                    analysis_result = f"🔄 {class_name}"
                    is_earthquake = (pred_class == 0)
                    if 'confidence' in df.columns:
                        confidence_value = float(df['confidence'].iloc[0])
                
                # 시간 정보
                time_diff = datetime.now() - file_info['modified_datetime']
                if time_diff.total_seconds() < 60:
                    time_text = f"{int(time_diff.total_seconds())}초 전"
                elif time_diff.total_seconds() < 3600:
                    time_text = f"{int(time_diff.total_seconds()//60)}분 전"
                else:
                    time_text = f"{int(time_diff.total_seconds()//3600)}시간 전"
                
                # 파일 크기
                try:
                    file_size_kb = os.path.getsize(file_info['filepath']) / 1024
                    size_text = f"{file_size_kb:.1f}KB"
                except:
                    size_text = "Unknown"
                
                # 진도 (intensity) 정보 추출 - influx_new2_realtime.py에서 저장된 값 사용
                magnitude_text = "-"
                try:
                    # 1순위: influx_new2_realtime.py에서 저장한 intensity 값 (파일명에서 추출)
                    if 'intensity' in df.columns:
                        intensity = float(df['intensity'].iloc[0])
                        magnitude_text = f"{intensity:.2f}"
                    else:
                        # 2순위: 파일명에서 intensity 파싱 시도 (event_포트_시간.csv 형태일 경우 원본 데이터에서)
                        # influx_new2_realtime.py의 event_info['intensity'] 값 찾기
                        possible_intensity_cols = ['intensity', 'magnitude', 'mag']
                        for col in possible_intensity_cols:
                            if col in df.columns:
                                intensity = float(df[col].iloc[0])
                                # 0~10 범위로 정규화 (InfluxDB intensity 값)
                                if intensity > 100:  # 센서 원시값인 경우
                                    intensity = min(10.0, intensity / 100.0)  # 스케일 조정
                                magnitude_text = f"{intensity:.2f}"
                                break
                        
                        # 3순위: 센서 데이터가 있다면 추정값 계산 (하지만 매우 낮은 값으로)
                        if magnitude_text == "-":
                            max_values = []
                            for sensor_num in range(1, 4):
                                for axis in ['x', 'y', 'z']:
                                    col_name = f'sensor_{sensor_num}_{axis}'
                                    if col_name in df.columns:
                                        max_val = abs(df[col_name]).max()
                                        max_values.append(max_val)
                            
                            if max_values:
                                raw_max = max(max_values)
                                # 센서 원시값을 0~10 진도 스케일로 변환 (추정)
                                estimated_intensity = min(10.0, raw_max / 1000.0)  # 매우 보수적 스케일링
                                magnitude_text = f"{estimated_intensity:.2f}"
                except Exception as e:
                    magnitude_text = "-"
                    print(f"진도 추출 오류: {e}")
                
                # 다운로드 버튼 키 생성
                filename = file_info['filename']
                download_key = f"download_{start_idx + i + 1}_{filename}"
                
                # 테이블 행 데이터 추가 (발생시간순 번호)
                table_data.append({
                    '번호': start_idx + i + 1,
                    '센서위치': parsed_info['location'],
                    '진도': magnitude_text,
                    '분석결과': analysis_result,
                    '신뢰도': f"{confidence_value:.1%}" if confidence_value > 0 else "-" if not is_expert_corrected else "100% ✏️",
                    '발생시간': parsed_info['datetime_str'],
                    '경과시간': time_text,
                    '다운로드': "다운로드",  # 다운로드 버튼
                    '_filepath': file_info['filepath'],  # 다운로드용 숨김 필드
                    '_download_key': download_key,  # 버튼 키
                    '_filename': filename  # 파일명
                })
            
            except Exception as e:
                # 오류 발생시에도 기본 정보는 표시
                parsed_info = parse_filename_info(file_info['filename'])
                filename = file_info['filename']
                download_key = f"download_{start_idx + i + 1}_{filename}"
                table_data.append({
                    '번호': start_idx + i + 1,
                    '센서위치': parsed_info['location'],
                    '진도': "-",
                    '분석결과': f"❌ 오류: {str(e)[:30]}",
                    '신뢰도': "-",
                    '발생시간': parsed_info['datetime_str'],
                    '경과시간': "-",
                    '다운로드': "다운로드",
                    '_filepath': file_info['filepath'],
                    '_download_key': download_key,
                    '_filename': filename
                })
        
        # 테이블 표시
        if table_data:
            # 디버깅: 첫 번째 행의 키 확인
            try:
                if len(table_data) > 0:
                    first_row_keys = list(table_data[0].keys())
                    print(f"테이블 데이터 키들: {first_row_keys}")
            except Exception as e:
                print(f"디버깅 오류: {e}")
            
            df_table = pd.DataFrame(table_data)
            
            # 테이블 데이터와 다운로드 버튼을 함께 표시
            # 각 행을 개별적으로 처리하여 다운로드 버튼 포함
            
            # 테이블 헤더
            col_header = st.columns([0.6, 1.2, 0.8, 1.8, 1.8, 0.8, 1.0, 0.8, 0.8, 1.0])
            headers = ['번호', '센서위치', '진도', '발생시간', '분석결과', '재학습', '파형보기', '수정', '삭제', '다운로드']
            
            for i, header in enumerate(headers):
                with col_header[i]:
                    st.markdown(f"<div style='text-align: center;'><b>{header}</b></div>", unsafe_allow_html=True)
            
            # 헤더와 데이터 사이 간격 최소화
            st.markdown("<hr style='margin: 2px 0; opacity: 0.3;'>", unsafe_allow_html=True)
            
            # 테이블 버튼 스타일 최적화
            st.markdown("""
                <style>
                .stButton > button {
                    width: 100%;
                    height: 32px;
                    font-size: 11px;
                    padding: 3px 6px;
                    margin: 1px 0;
                }
                div[data-testid="column"] {
                    padding: 0px 2px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # 각 데이터 행 표시
            for idx, row in df_table.iterrows():
                col_data = st.columns([0.6, 1.2, 0.8, 1.8, 1.8, 0.8, 1.0, 0.8, 0.8, 1.0])
                
                # 파일 정보 가져오기
                file_info = current_page_files[idx]
                filename = file_info['filename']
                
                # 재학습 상태 확인
                retraining_status = get_retraining_status_for_file(file_info['filepath'])
                
                # 데이터 칼럼들
                data_values = [
                    str(row['번호']),
                    row['센서위치'], 
                    row['진도'],
                    row['발생시간'],
                    row['분석결과'],
                    retraining_status
                ]
                
                # 데이터 표시 (중앙 정렬, 간격 최소화)
                for i, value in enumerate(data_values):
                    with col_data[i]:
                        if i == 5:  # 재학습 상태 칼럼
                            if value == "완료":
                                st.markdown(f"<div style='text-align: center; margin: 0; padding: 2px 0; color: green; font-weight: bold;'>✅ {value}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='text-align: center; margin: 0; padding: 2px 0; color: orange; font-weight: bold;'>⏳ {value}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='text-align: center; margin: 0; padding: 2px 0;'>{value}</div>", unsafe_allow_html=True)
                
                # 파형 보기 버튼 (6번째 컬럼)
                with col_data[6]:
                    waveform_key = f"waveform_{start_idx + idx + 1}_{row['_filename']}"
                    
                    # 파형 데이터가 있는지 확인
                    try:
                        df_check = pd.read_csv(row['_filepath'])
                        sensor_patterns = [
                            ('sensor_1_x', 'sensor_1_y', 'sensor_1_z'),
                            ('x', 'y', 'z'),
                            ('X', 'Y', 'Z'),
                            ('acc_x', 'acc_y', 'acc_z')
                        ]
                        
                        has_sensor_data = False
                        for pattern in sensor_patterns:
                            if all(col in df_check.columns for col in pattern):
                                has_sensor_data = True
                                break
                        
                        if has_sensor_data:
                            # 현재 선택된 파일인지 확인
                            is_selected = (st.session_state.selected_file_for_waveform and 
                                         st.session_state.selected_file_for_waveform['filepath'] == row['_filepath'])
                            
                            button_text = "📊 보는 중" if is_selected else "📊 파형"
                            button_type = "secondary" if is_selected else "primary"
                            
                            if st.button(button_text, key=waveform_key, use_container_width=True, 
                                       type=button_type, help="위 파형 섹션에서 3축 가속도 파형 보기"):
                                # 선택된 파일 정보를 세션 상태에 저장
                                current_file_info = {
                                    'filename': row['_filename'],
                                    'filepath': row['_filepath'],
                                    'modified_datetime': current_page_files[idx]['modified_datetime'],
                                    'type': current_page_files[idx]['type']
                                }
                                st.session_state.selected_file_for_waveform = current_file_info
                                
                                # 파형 섹션으로 자동 스크롤
                                safe_rerun()
                        else:
                            st.button("센서없음", disabled=True, key=f"no_sensor_{waveform_key}", 
                                    use_container_width=True, help="3축 가속도 데이터가 없습니다")
                            
                    except Exception as e:
                        st.button("오류", disabled=True, key=f"error_wave_{waveform_key}", 
                                use_container_width=True, help=f"파일 읽기 오류: {str(e)[:30]}")
                
                # 수정 버튼 (7번째 컬럼)
                with col_data[7]:
                    modify_key = f"modify_{start_idx + idx + 1}_{row['_filename']}"
                    
                    # AI 분석된 결과만 수정 가능 (🔴🟠🟢 이모지 또는 🔄 이모지가 있는 경우)
                    analysis_text = row['분석결과']
                    has_ai_analysis = any(emoji in analysis_text for emoji in ['🔴', '🟠', '🟢', '🔄'])
                    is_processing_wait = '⏳' in analysis_text or '처리 대기' in analysis_text
                    is_error = '❌' in analysis_text or '오류' in analysis_text
                    
                    if has_ai_analysis and not is_processing_wait and not is_error:  
                        if st.button("✏️ 수정", key=modify_key, use_container_width=True, help="분석 결과 수정"):
                            st.session_state[f'show_modify_modal_{idx}'] = True
                            st.session_state[f'modify_file_info_{idx}'] = {
                                'filepath': row['_filepath'],
                                'filename': row['_filename'],
                                'current_result': row['분석결과'],
                                'current_confidence': row['신뢰도']
                            }
                            safe_rerun()
                    else:
                        reason = "처리 대기 중" if is_processing_wait else "오류 발생" if is_error else "분석 미완료"
                        st.button("수정불가", disabled=True, key=f"disabled_modify_{modify_key}", use_container_width=True, help=f"{reason} - 수정할 수 없습니다")
                
                # 삭제 버튼 (8번째 컬럼)
                with col_data[8]:
                    delete_key = f"delete_{start_idx + idx + 1}_{row['_filename']}"
                    
                    if st.button("🗑️ 삭제", key=delete_key, use_container_width=True, type="secondary", help="파일을 완전히 삭제합니다"):
                        st.session_state[f'show_delete_modal_{idx}'] = True
                        st.session_state[f'delete_file_info_{idx}'] = {
                            'filepath': row['_filepath'],
                            'filename': row['_filename'],
                            'location': row['센서위치'],
                            'analysis_result': row['분석결과']
                        }
                        safe_rerun()
                
                # 다운로드 버튼 (9번째 컬럼)
                with col_data[9]:
                    filepath = row['_filepath']
                    filename = row['_filename']
                    
                    try:
                        if os.path.exists(filepath):
                            with open(filepath, 'rb') as file:
                                file_data = file.read()
                            
                            st.download_button(
                                label="다운로드",
                                data=file_data,
                                file_name=filename,
                                mime="text/csv",
                                key=row['_download_key'],
                                use_container_width=True
                            )
                        else:
                            st.button("파일없음", disabled=True, key=f"disabled_{row['_download_key']}", use_container_width=True)
                    except Exception as e:
                        st.button("오류", disabled=True, key=f"error_{row['_download_key']}", use_container_width=True)
                
                # 행 간격 최소화
                st.markdown("<div style='margin: 3px 0;'></div>", unsafe_allow_html=True)
            
            # 전체 ZIP 다운로드
            st.markdown("### 📦 전체 파일 다운로드")
            # st.write("🔍 **디버그**: 이 섹션은 한 번만 표시되어야 합니다")
            if len(df_table) > 1:
                try:
                    import zipfile
                    import io
                    
                    # ZIP 파일 생성
                    zip_buffer = io.BytesIO()
                    zip_file_count = 0
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for _, row in df_table.iterrows():
                            filepath = row['_filepath']
                            if os.path.exists(filepath):
                                zip_file.write(filepath, row['다운로드'])
                                zip_file_count += 1
                    
                    zip_buffer.seek(0)
                    
                    if zip_file_count > 0:
                        st.download_button(
                            label=f"📦 현재 페이지 전체 파일 다운로드 ({zip_file_count}개)",
                            data=zip_buffer.getvalue(),
                            file_name=f"earthquake_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.warning("다운로드할 파일이 없습니다.")
                        
                except Exception as e:
                    st.error(f"❌ ZIP 생성 오류: {e}")
            
            
            # 페이지 네비게이션 (테이블 아래)
            st.markdown("---")
            col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 2, 1, 1])
            
            with col_nav1:
                if st.button("⏪ 첫 페이지", key="nav_first") and st.session_state.page_number > 0:
                    st.session_state.page_number = 0
                    safe_rerun()
            
            with col_nav2:
                if st.button("◀ 이전", key="nav_prev") and st.session_state.page_number > 0:
                    st.session_state.page_number -= 1
                    safe_rerun()
            
            with col_nav3:
                st.markdown(f"<div style='text-align: center; padding: 8px;'><b>{st.session_state.page_number + 1} / {total_pages} 페이지</b></div>", unsafe_allow_html=True)
            
            with col_nav4:
                if st.button("다음 ▶", key="nav_next") and st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number += 1
                    safe_rerun()
            
            with col_nav5:
                if st.button("마지막 페이지 ⏩", key="nav_last") and st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number = total_pages - 1
                    safe_rerun()
            
            # =================== 수정 팝업 처리 ===================
            # 수정 확인은 사이드바에서 처리되므로 여기서는 간단한 안내만 표시
            modify_modal_shown = False
            for idx in range(len(df_table)):
                modal_key = f'show_modify_modal_{idx}'
                if st.session_state.get(modal_key, False):
                    modify_modal_shown = True
                    break
            
            # 수정 모달이 활성화되면 안내 메시지만 표시
            if modify_modal_shown:
                st.info("✏️ **분석 결과 수정창이 왼쪽 사이드바에 표시되었습니다. 사이드바를 확인해주세요!**")
            
            # =================== 삭제 확인 팝업 처리 ===================
            # 삭제 확인은 사이드바에서 처리되므로 여기서는 간단한 안내만 표시
            delete_modal_shown = False
            for idx in range(len(df_table)):
                delete_modal_key = f'show_delete_modal_{idx}'
                if st.session_state.get(delete_modal_key, False):
                    delete_modal_shown = True
                    break
            
            # 삭제 모달이 활성화되면 안내 메시지만 표시
            if delete_modal_shown:
                st.info("🔍 **삭제 확인창이 왼쪽 사이드바에 표시되었습니다. 사이드바를 확인해주세요!**")
            
            # 수정 완료 메시지 표시
            for idx in range(len(df_table)):
                success_key = f'correction_success_{idx}'
                if st.session_state.get(success_key):
                    st.success(st.session_state[success_key])
                    # 메시지 표시 후 제거
                    del st.session_state[success_key]
            
            # 삭제 완료 메시지 표시
            for idx in range(len(df_table)):
                delete_success_key = f'delete_success_{idx}'
                if st.session_state.get(delete_success_key):
                    st.success(st.session_state[delete_success_key])
                    # 메시지 표시 후 제거
                    del st.session_state[delete_success_key]
        
        else:
            st.warning("표시할 이벤트가 없습니다.")
    
    else:
        st.warning("🔍 처리된 이벤트가 없습니다. 새로운 데이터를 기다리는 중...")
    

# =========================== 메인 실행 ===========================

def main():
    """메인 함수"""
    # 캐시 클리어 (KeyError 방지)
    try:
        st.cache_data.clear()
    except:
        pass
    
    # 기본 세션 상태 초기화
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # 지진 오보 분석 시스템 모델 로딩
    if 'model' not in st.session_state:
        model, model_name = load_new2_model()
        st.session_state.model = model
        st.session_state.model_name = model_name
        st.session_state.model_loaded = model is not None
    
    # UI 렌더링
    render_system_sidebar()
    render_system_dashboard()

if __name__ == "__main__":
    main()
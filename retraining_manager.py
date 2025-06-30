#!/usr/bin/env python3
"""
재학습 관리 모듈
재학습 상태 확인, 파일 업데이트, 사이드바 렌더링
"""

import os
import pandas as pd
from datetime import datetime
from typing import Tuple
import streamlit as st
from dashboard_utils import CONFIG

# =========================== 재학습 상태 관리 ===========================

def count_retraining_status() -> Tuple[int, int]:
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

def get_retraining_status_for_file(filepath: str) -> str:
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

def mark_all_files_as_retrained() -> int:
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

# =========================== 재학습 시스템 연동 ===========================

def get_retraining_system():
    """재학습 시스템 인스턴스 가져오기"""
    try:
        from retraining_system import NEW2RetrainingSystem
        
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
        
    except ImportError:
        print("재학습 시스템을 사용할 수 없습니다. retraining_system.py를 확인하세요.")
        return None

# =========================== 사이드바 렌더링 ===========================

def render_retraining_sidebar():
    """재학습 관리 사이드바"""
    # safe_rerun 함수 정의 (순환 import 방지)
    def safe_rerun():
        """Streamlit 버전에 따른 안전한 rerun 함수"""
        try:
            st.rerun()
        except AttributeError:
            try:
                st.experimental_rerun()
            except AttributeError:
                pass
    
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
import streamlit as st
st.set_page_config(layout="wide", page_title="3클래스 지진 모니터링 시스템", page_icon="🌍")

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# ------------------------- 설정 -------------------------
CONFIG = {
    'BASE_OUTPUT_DIR': "C:/earthquake_modeling/earthquake_project_v3/influxLogs",
    'RAW_DATA_DIR': "C:/earthquake_modeling/earthquake_project_v3/influxLogs/base",
    'PROCESSED_DATA_DIR': "C:/earthquake_modeling/earthquake_project_v3/influxLogs/processed",
    'MODEL_PATH': "convlstm_3class_model.h5",  # 3클래스 모델
    'PORTS': [6060, 7001, 7053, 7060, 7070, 8010, 8080],
    'AUTO_REFRESH_INTERVAL': 10  # 10초마다 자동 새로고침
}

# 3클래스 정의 (실시간 코드와 동일)
CLASS_NAMES = {0: '지진', 1: '불규칙생활', 2: '모터진동'}
CLASS_COLORS = {0: '🔴', 1: '🟢', 2: '🟠'}
CLASS_ALERT_STATUS = {0: 'EARTHQUAKE_ALERT', 1: 'NO_ALERT', 2: 'NO_ALERT'}

# 디렉토리 생성
for dir_path in [CONFIG['RAW_DATA_DIR'], CONFIG['PROCESSED_DATA_DIR']]:
    os.makedirs(dir_path, exist_ok=True)

# ------------------------- 모델 로딩 -------------------------
@st.cache_resource
def load_earthquake_model():
    """3클래스 지진 예측 모델 로드"""
    
    possible_paths = [
        CONFIG['MODEL_PATH'],
        f"C:/earthquake_project/{CONFIG['MODEL_PATH']}",
        f"C:/earthquake_clean_project/{CONFIG['MODEL_PATH']}"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        st.error("❌ 3클래스 모델 파일을 찾을 수 없습니다.")
        st.write("**확인할 경로들:**")
        for path in possible_paths:
            st.write(f"- {path}")
        return None
    
    try:
        st.info(f"🔄 3클래스 모델 로딩 중: {os.path.basename(model_path)}")
        
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        model = load_model(model_path, compile=False)
        st.success(f"✅ 3클래스 모델 로딩 완료!")
        return model
        
    except Exception as e:
        st.error(f"❌ 모델 로딩 실패: {e}")
        return None

# ------------------------- 데이터 처리 함수들 -------------------------
@st.cache_data(ttl=5)  # 5초 캐시
def get_event_files():
    """AI 처리된 이벤트 파일 목록 가져오기 (processed 폴더 우선)"""
    all_files = []
    
    # 1. AI 처리된 파일 (processed 폴더)
    if os.path.exists(CONFIG['PROCESSED_DATA_DIR']):
        for filename in os.listdir(CONFIG['PROCESSED_DATA_DIR']):
            if filename.endswith('.csv') and filename.startswith('ai_'):
                filepath = os.path.join(CONFIG['PROCESSED_DATA_DIR'], filename)
                try:
                    modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    all_files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'modified_datetime': modified_time,
                        'type': 'AI_PROCESSED',
                        'original_filename': filename.replace('ai_', 'event_')
                    })
                except:
                    continue
    
    # 2. 원시 데이터 파일 (base 폴더) - AI 처리 결과가 없는 경우만
    if os.path.exists(CONFIG['RAW_DATA_DIR']):
        processed_files = {f['original_filename'] for f in all_files}
        
        for filename in os.listdir(CONFIG['RAW_DATA_DIR']):
            if filename.endswith('.csv') and filename.startswith('event_'):
                if filename not in processed_files:  # AI 처리 결과가 없는 경우만
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
    
    # 수정 시간순 정렬 (최신 파일 먼저)
    all_files.sort(key=lambda x: x['modified_datetime'], reverse=True)
    return all_files

def parse_filename_info(filename):
    """파일명에서 정보 추출"""
    try:
        # ai_6060_2025-06-02_21-04-13.csv 또는 event_6060_2025-06-02_21-04-13.csv
        parts = filename.replace('.csv', '').split('_')
        
        if filename.startswith('ai_'):
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

def check_for_new_events():
    """새로운 이벤트 확인"""
    if 'last_file_count' not in st.session_state:
        st.session_state.last_file_count = 0
    
    current_files = get_event_files()
    current_count = len(current_files)
    
    if current_count > st.session_state.last_file_count:
        new_count = current_count - st.session_state.last_file_count
        st.session_state.last_file_count = current_count
        st.session_state.last_check_time = time.time()
        return True, new_count
    else:
        st.session_state.last_file_count = current_count
        st.session_state.last_check_time = time.time()
        return False, 0

# ------------------------- 사이드바 -------------------------
def render_sidebar():
    """사이드바 렌더링"""
    st.sidebar.title("🌍 시스템 제어")
    
    # 자동 새로고침 설정
    st.sidebar.subheader("⏱️ 자동 새로고침")
    auto_refresh = st.sidebar.checkbox(
        "자동 새로고침 활성화", 
        value=st.session_state.get('auto_refresh', True),
        key='auto_refresh'
    )
    
    if auto_refresh:
        st.sidebar.info(f"🔄 {CONFIG['AUTO_REFRESH_INTERVAL']}초마다 업데이트")
    else:
        if st.sidebar.button("🔄 수동 새로고침"):
            st.cache_data.clear()
            st.rerun()
    
    # 새 이벤트 확인
    has_new, new_count = check_for_new_events()
    if has_new:
        st.sidebar.success(f"🆕 새로운 이벤트 {new_count}개 발견!")
    
    # 마지막 체크 시간
    if 'last_check_time' in st.session_state:
        last_check = datetime.fromtimestamp(st.session_state.last_check_time)
        st.sidebar.caption(f"마지막 체크: {last_check.strftime('%H:%M:%S')}")
    
    st.sidebar.markdown("---")
    
    # 시스템 정보
    st.sidebar.subheader("📊 시스템 정보")
    st.sidebar.text(f"3클래스 모델: {os.path.basename(CONFIG['MODEL_PATH'])}")
    st.sidebar.text(f"원시 데이터: base/")
    st.sidebar.text(f"AI 처리 결과: processed/")
    
    # 통계 정보
    current_files = get_event_files()
    ai_processed_count = len([f for f in current_files if f['type'] == 'AI_PROCESSED'])
    raw_only_count = len([f for f in current_files if f['type'] == 'RAW_DATA'])
    
    st.sidebar.metric("총 이벤트 수", len(current_files))
    st.sidebar.metric("AI 처리 완료", ai_processed_count)
    st.sidebar.metric("원시 데이터만", raw_only_count)
    
    if current_files:
        latest_file = current_files[0]
        time_diff = datetime.now() - latest_file['modified_datetime']
        st.sidebar.metric("최근 이벤트", f"{int(time_diff.total_seconds())}초 전")

# ------------------------- 이벤트 목록 및 페이지네이션 -------------------------
def render_event_list(files):
    """이벤트 목록과 페이지네이션 렌더링"""
    # =================== 전체 이벤트 목록 섹션 ===================
    st.markdown("---")
    st.subheader("📋 전체 이벤트 목록")

    # 파일이 없는 경우 처리
    if not files:
        st.info("📁 표시할 이벤트가 없습니다.")
        return

    # 페이지네이션 설정
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    # 상단 컨트롤
    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])

    with control_col1:
        items_per_page = st.selectbox(
            "페이지당 표시 수", 
            [5, 10, 20, 50], 
            index=1,  # 기본값 10
            key="items_per_page"
        )

    with control_col2:
        total_pages = max(1, (len(files) + items_per_page - 1) // items_per_page)
        st.info(f"📄 총 {total_pages}페이지 / {len(files)}개 이벤트")

    with control_col3:
        # 페이지 직접 이동
        col_input1, col_input2 = st.columns([2, 1])
        with col_input1:
            page_input = st.number_input(
                "페이지 이동", 
                min_value=1, 
                max_value=total_pages, 
                value=st.session_state.current_page,
                key="page_direct_input"
            )
        with col_input2:
            if st.button("이동", key="page_go"):
                if 1 <= page_input <= total_pages:
                    st.session_state.current_page = page_input
                    st.rerun()

    # 페이지 경계값 체크 및 조정
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    elif st.session_state.current_page < 1:
        st.session_state.current_page = 1

    # 현재 페이지 데이터 계산
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(files))
    current_files = files[start_idx:end_idx]

    # 페이지 정보 표시
    st.write(f"**📊 현재 페이지:** {st.session_state.current_page}/{total_pages} | **표시 범위:** {start_idx + 1}-{end_idx} / {len(files)}")

    # 파일 목록 표시
    for idx, file_info in enumerate(current_files):
        global_index = start_idx + idx
        
        try:
            df = pd.read_csv(file_info['filepath'])
            filename = file_info['filename']
            parsed_info = parse_filename_info(filename)
            
            # 예측 결과 확인
            if file_info['type'] == 'AI_PROCESSED' and 'predicted_class' in df.columns:
                pred_class = int(df['predicted_class'].iloc[0])
                class_name = CLASS_NAMES.get(pred_class, 'Unknown')
                confidence = float(df.get('confidence', [0.0])[0])
                class_icon = CLASS_COLORS[pred_class]
                pred_result = f"{class_icon} {class_name} ({confidence:.2%})"
            else:
                pred_result = "⏳ AI 처리 대기"
            
            # 새 파일인지 확인
            time_since = datetime.now() - file_info['modified_datetime']
            is_new = time_since.total_seconds() < 300  # 5분 이내
            
            # Expander 제목
            file_number = global_index + 1
            title = f"{'🆕' if is_new else '📂'} #{file_number} {parsed_info['location']} | {pred_result}"
            
            with st.expander(title):
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    st.write(f"**파일 번호:** #{file_number}")
                    st.write(f"**센서 포트:** {parsed_info['port']}")
                    st.write(f"**이벤트 시간:** {parsed_info['datetime_str']}")
                    st.write(f"**파일 타입:** {file_info['type']}")
                
                with col2:
                    st.write(f"**파일 크기:** {len(df)} 행")
                    st.write(f"**수정 시간:** {file_info['modified_datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # 진도 정보
                    magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower()]
                    if magnitude_cols:
                        max_mag = df[magnitude_cols[0]].max()
                        st.write(f"**최대 진도:** {max_mag:.3f}")
                
                with col3:
                    # AI 처리 결과 상세 정보
                    if file_info['type'] == 'AI_PROCESSED':
                        if 'predicted_class' in df.columns:
                            st.write("**🧠 AI 분석 결과:**")
                            st.write(f"클래스: {class_name}")
                            st.write(f"신뢰도: {confidence:.2%}")
                            
                            if 'alert_status' in df.columns:
                                alert_status = df['alert_status'].iloc[0]
                                if alert_status == 'EARTHQUAKE_ALERT':
                                    st.write("**상태: 🚨 경보 발령**")
                                else:
                                    st.write("**상태: ✅ 경보 억제**")
                    else:
                        st.write("**⏳ AI 처리 대기 중**")
                
                # 센서 데이터 미리보기 옵션
                if st.checkbox("센서 데이터 차트 보기", key=f"chart_{filename}_{global_index}"):
                    # 센서 데이터 찾기
                    sensor_cols = []
                    for i in range(1, 4):
                        x_col = f'sensor_{i}_x'
                        y_col = f'sensor_{i}_y'
                        z_col = f'sensor_{i}_z'
                        
                        if all(col in df.columns for col in [x_col, y_col, z_col]):
                            sensor_cols.extend([x_col, y_col, z_col])
                            break
                    
                    if sensor_cols:
                        st.line_chart(df[sensor_cols])
                    else:
                        st.warning("표시할 수 있는 센서 데이터가 없습니다.")
                
                # 데이터 테이블 미리보기 (선택적)
                if st.checkbox("데이터 테이블 보기", key=f"table_{filename}_{global_index}"):
                    st.dataframe(df.head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"파일 처리 오류: {e}")

    # =================== 하단 페이지네이션 컨트롤 ===================
    if total_pages > 1:
        st.markdown("---")
        st.markdown("#### 📄 페이지 이동")
        
        def create_pagination():
            """페이지네이션 버튼 생성"""
            current_page = st.session_state.current_page
            
            # 페이지 번호 범위 계산
            if total_pages <= 7:
                # 총 페이지가 7 이하면 모든 페이지 표시
                page_range = list(range(1, total_pages + 1))
            else:
                # 복잡한 페이지네이션 로직
                if current_page <= 4:
                    # 앞쪽에 있을 때: 1 2 3 4 5 ... 마지막
                    page_range = [1, 2, 3, 4, 5, '...', total_pages]
                elif current_page >= total_pages - 3:
                    # 뒤쪽에 있을 때: 1 ... (마지막-4) (마지막-3) (마지막-2) (마지막-1) 마지막
                    page_range = [1, '...'] + list(range(total_pages - 4, total_pages + 1))
                else:
                    # 중간에 있을 때: 1 ... (현재-1) 현재 (현재+1) ... 마지막
                    page_range = [1, '...', current_page - 1, current_page, current_page + 1, '...', total_pages]
            
            return page_range
        
        # 페이지 버튼 그룹
        nav_cols = st.columns([1, 1, 1, 1, 1, 1, 2])
        
        with nav_cols[0]:
            if st.button("⏮️ 첫 페이지", disabled=(st.session_state.current_page == 1), key="first_page"):
                st.session_state.current_page = 1
                st.rerun()
        
        with nav_cols[1]:
            if st.button("◀️ 이전", disabled=(st.session_state.current_page == 1), key="prev_page"):
                st.session_state.current_page = max(1, st.session_state.current_page - 1)
                st.rerun()
        
        with nav_cols[2]:
            st.write(f"**{st.session_state.current_page}** / {total_pages}")
        
        with nav_cols[3]:
            if st.button("▶️ 다음", disabled=(st.session_state.current_page == total_pages), key="next_page"):
                st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
                st.rerun()
        
        with nav_cols[4]:
            if st.button("⏭️ 마지막", disabled=(st.session_state.current_page == total_pages), key="last_page"):
                st.session_state.current_page = total_pages
                st.rerun()
        
        with nav_cols[5]:
            # 빠른 이동 버튼들
            if total_pages > 10:
                if st.button("🔟 +10", disabled=(st.session_state.current_page + 10 > total_pages), key="plus_10"):
                    st.session_state.current_page = min(total_pages, st.session_state.current_page + 10)
                    st.rerun()
        
        with nav_cols[6]:
            # 페이지 요약 정보
            start_item = start_idx + 1
            end_item = end_idx
            st.caption(f"📋 {start_item}-{end_item}번째 항목 표시 중 (총 {len(files)}개)")

        # 스마트 페이지 바로가기 (번호 버튼들)
        if total_pages > 5:
            page_range = create_pagination()
            st.markdown("**🔗 페이지 바로가기:**")
            
            # 중앙 정렬된 페이지네이션
            _, col_center, _ = st.columns([1, 2, 1])
            
            with col_center:
                # 페이지 버튼들을 가로로 배치
                num_elements = len(page_range)
                cols = st.columns(num_elements)
                
                for i, page_item in enumerate(page_range):
                    with cols[i]:
                        if page_item == '...':
                            # 점점점을 중앙 정렬로 표시
                            st.markdown('<div style="text-align: center; padding: 8px 0; color: #666;">...</div>', unsafe_allow_html=True)
                        else:
                            # 페이지 번호 버튼
                            is_current = page_item == st.session_state.current_page
                            
                            if is_current:
                                # 현재 페이지는 primary 버튼으로 표시하되 비활성화
                                st.button(
                                    str(page_item), 
                                    key=f"page_btn_{page_item}",
                                    type="primary",
                                    disabled=True,
                                    use_container_width=True
                                )
                            else:
                                # 클릭 가능한 페이지
                                if st.button(
                                    str(page_item), 
                                    key=f"page_btn_{page_item}", 
                                    type="secondary",
                                    use_container_width=True
                                ):
                                    st.session_state.current_page = page_item
                                    st.rerun()
        
        # 페이지 점프 정보
        if total_pages > 20:
            st.info(f"💡 **팁:** 상단의 '페이지 이동' 입력창을 사용하면 원하는 페이지로 바로 이동할 수 있습니다.")
    else:
        st.info("📄 모든 이벤트가 한 페이지에 표시되고 있습니다.")

# ------------------------- 메인 대시보드 -------------------------
def render_dashboard():
    """메인 대시보드 렌더링"""
    st.title("🌍 3클래스 지진 감지 모니터링 대시보드")
    st.markdown("### 실시간 지진/불규칙생활/모터진동 분류 시스템")
    
    # 이벤트 파일 가져오기
    files = get_event_files()
    
    if not files:
        st.info("📁 아직 저장된 이벤트 데이터가 없습니다.")
        st.info("실시간 감지 시스템이 동작하면 자동으로 데이터가 수집됩니다.")
        
        # 디렉토리 상태 확인
        col1, col2 = st.columns(2)
        with col1:
            st.write("**📂 원시 데이터 폴더:**")
            if os.path.exists(CONFIG['RAW_DATA_DIR']):
                raw_files = [f for f in os.listdir(CONFIG['RAW_DATA_DIR']) if f.endswith('.csv')]
                st.write(f"✅ 존재 ({len(raw_files)}개 파일)")
            else:
                st.write("❌ 폴더 없음")
        
        with col2:
            st.write("**🧠 AI 처리 결과 폴더:**")
            if os.path.exists(CONFIG['PROCESSED_DATA_DIR']):
                processed_files = [f for f in os.listdir(CONFIG['PROCESSED_DATA_DIR']) if f.endswith('.csv')]
                st.write(f"✅ 존재 ({len(processed_files)}개 파일)")
            else:
                st.write("❌ 폴더 없음")
        
        # 자동 새로고침 처리
        if st.session_state.get('auto_refresh', True):
            time.sleep(CONFIG['AUTO_REFRESH_INTERVAL'])
            st.rerun()
        return
    
    # =================== 통계 대시보드 섹션 ===================
    st.subheader("📊 3클래스 분류 통계")
    
    # 예측 결과 통계 계산
    total_events = len(files)
    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    alert_count = 0
    suppressed_count = 0
    ai_processed_count = 0
    
    for file_info in files:
        try:
            df = pd.read_csv(file_info['filepath'])
            
            if file_info['type'] == 'AI_PROCESSED':
                ai_processed_count += 1
                
                # 3클래스 예측 결과 확인
                if 'predicted_class' in df.columns:
                    pred_class = int(df['predicted_class'].iloc[0])
                    class_name = CLASS_NAMES.get(pred_class, 'Unknown')
                    class_counts[class_name] += 1
                    
                    # 경보 상태 확인
                    if 'alert_status' in df.columns:
                        alert_status = df['alert_status'].iloc[0]
                        if alert_status == 'EARTHQUAKE_ALERT':
                            alert_count += 1
                        else:
                            suppressed_count += 1
                    elif pred_class == 0:  # 지진
                        alert_count += 1
                    else:
                        suppressed_count += 1
        except:
            continue
    
    # 통계 시각화 (4개 컬럼)
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric(
            label="📊 총 이벤트 수",
            value=f"{total_events:,}",
            delta=f"AI 처리: {ai_processed_count}"
        )
    
    with col_stat2:
        earthquake_percentage = (class_counts['지진'] / ai_processed_count * 100) if ai_processed_count > 0 else 0
        st.metric(
            label="🔴 지진 탐지",
            value=f"{class_counts['지진']:,}",
            delta=f"{earthquake_percentage:.1f}%"
        )
    
    with col_stat3:
        living_percentage = (class_counts['불규칙생활'] / ai_processed_count * 100) if ai_processed_count > 0 else 0
        st.metric(
            label="🟢 불규칙생활",
            value=f"{class_counts['불규칙생활']:,}",
            delta=f"{living_percentage:.1f}%"
        )
    
    with col_stat4:
        motor_percentage = (class_counts['모터진동'] / ai_processed_count * 100) if ai_processed_count > 0 else 0
        st.metric(
            label="🟠 모터진동",
            value=f"{class_counts['모터진동']:,}",
            delta=f"{motor_percentage:.1f}%"
        )
    
    # 경보 통계
    st.markdown("### 🚨 경보 시스템 성능")
    col_alert1, col_alert2, col_alert3 = st.columns(3)
    
    with col_alert1:
        alert_rate = (alert_count / ai_processed_count * 100) if ai_processed_count > 0 else 0
        st.metric(
            label="🚨 경보 발령",
            value=f"{alert_count:,}",
            delta=f"{alert_rate:.1f}%"
        )
    
    with col_alert2:
        suppression_rate = (suppressed_count / ai_processed_count * 100) if ai_processed_count > 0 else 0
        st.metric(
            label="🛡️ 오경보 억제",
            value=f"{suppressed_count:,}",
            delta=f"{suppression_rate:.1f}%"
        )
    
    with col_alert3:
        if ai_processed_count > 0:
            efficiency = suppression_rate
            st.metric(
                label="📈 시스템 효율성",
                value=f"{efficiency:.1f}%",
                delta="오경보 억제율"
            )
    
    # =================== 3클래스 분포 차트 ===================
    if ai_processed_count > 0:
        st.markdown("### 📊 3클래스 분포 차트")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # 파이 차트
            fig_pie = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="클래스별 분포",
                color_discrete_map={'지진': '#FF6B6B', '불규칙생활': '#4ECDC4', '모터진동': '#FFE66D'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # 막대 차트
            fig_bar = px.bar(
                x=list(class_counts.keys()),
                y=list(class_counts.values()),
                title="클래스별 이벤트 수",
                color=list(class_counts.keys()),
                color_discrete_map={'지진': '#FF6B6B', '불규칙생활': '#4ECDC4', '모터진동': '#FFE66D'}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # =================== 최근 이벤트 분석 섹션 ===================
    st.markdown("---")
    st.subheader("📈 최근 이벤트 상세 분석")
    
    latest_file = files[0]
    
    try:
        df_latest = pd.read_csv(latest_file['filepath'])
        parsed_info = parse_filename_info(latest_file['filename'])
        
        # 최신 파일 정보 표시
        info_col1, info_col2 = st.columns([2, 1])
        
        with info_col1:
            st.info(f"📁 **최근 파일:** `{latest_file['filename']}`")
            st.info(f"📄 **데이터 크기:** {len(df_latest)}행, {len(df_latest.columns)}열")
            st.info(f"🕒 **이벤트 시간:** {parsed_info['datetime_str']}")
            st.info(f"📍 **센서 위치:** {parsed_info['location']}")
        
        with info_col2:
            # 예측 결과 박스
            if latest_file['type'] == 'AI_PROCESSED' and 'predicted_class' in df_latest.columns:
                pred_class = int(df_latest['predicted_class'].iloc[0])
                class_name = CLASS_NAMES.get(pred_class, 'Unknown')
                confidence = float(df_latest.get('confidence', [0.0])[0])
                class_icon = CLASS_COLORS[pred_class]
                
                # 클래스에 따른 색상 적용
                if pred_class == 0:  # 지진
                    st.error(f"🚨 **{class_icon} {class_name}**\n신뢰도: {confidence:.2%}")
                elif pred_class == 1:  # 불규칙생활
                    st.success(f"✅ **{class_icon} {class_name}**\n신뢰도: {confidence:.2%}")
                else:  # 모터진동
                    st.warning(f"⚠️ **{class_icon} {class_name}**\n신뢰도: {confidence:.2%}")
                
                # 경보 상태
                if 'alert_status' in df_latest.columns:
                    alert_status = df_latest['alert_status'].iloc[0]
                    if alert_status == 'EARTHQUAKE_ALERT':
                        st.error("🚨 **경보 발령 상태**")
                    else:
                        st.success("🛡️ **경보 억제 상태**")
            else:
                st.warning("⏳ **AI 처리 대기 중**")
        
        # 센서 데이터 시각화
        st.markdown("#### 📊 센서 데이터 시각화")
        
        # 센서 데이터 찾기
        sensor_found = False
        for i in range(1, 4):
            x_col = f'sensor_{i}_x'
            y_col = f'sensor_{i}_y'
            z_col = f'sensor_{i}_z'
            
            if all(col in df_latest.columns for col in [x_col, y_col, z_col]):
                sensor_found = True
                
                # 3축 센서 데이터 차트
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig_sensor = go.Figure()
                    fig_sensor.add_trace(go.Scatter(y=df_latest[x_col], name='X축', line=dict(color='red')))
                    fig_sensor.add_trace(go.Scatter(y=df_latest[y_col], name='Y축', line=dict(color='green')))
                    fig_sensor.add_trace(go.Scatter(y=df_latest[z_col], name='Z축', line=dict(color='blue')))
                    fig_sensor.update_layout(
                        title=f"센서 {i} - 3축 가속도",
                        xaxis_title="시간 (샘플)",
                        yaxis_title="가속도",
                        height=400
                    )
                    st.plotly_chart(fig_sensor, use_container_width=True)
                
                with chart_col2:
                    # 진도 및 복합 가속도 (있는 경우)
                    magnitude_col = f'sensor_{i}_magnitude'
                    composite_col = f'sensor_{i}_composite_acceleration'
                    
                    if magnitude_col in df_latest.columns:
                        fig_mag = go.Figure()
                        fig_mag.add_trace(go.Scatter(y=df_latest[magnitude_col], name='진도', line=dict(color='orange')))
                        if composite_col in df_latest.columns:
                            fig_mag.add_trace(go.Scatter(y=df_latest[composite_col], name='복합가속도', line=dict(color='purple')))
                        fig_mag.update_layout(
                            title=f"센서 {i} - 진도/복합가속도",
                            xaxis_title="시간 (샘플)",
                            yaxis_title="값",
                            height=400
                        )
                        st.plotly_chart(fig_mag, use_container_width=True)
                    else:
                        st.info("진도 데이터가 없습니다.")
                break
        
        if not sensor_found:
            st.warning("표시할 수 있는 센서 데이터가 없습니다.")
        
        # 통계 요약
        st.markdown("#### 📈 데이터 통계 요약")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        # 각 센서의 통계 계산
        for i in range(1, 4):
            magnitude_col = f'sensor_{i}_magnitude'
            composite_col = f'sensor_{i}_composite_acceleration'
            
            if magnitude_col in df_latest.columns:
                max_magnitude = df_latest[magnitude_col].max()
                avg_magnitude = df_latest[magnitude_col].mean()
                
                with eval(f'stat_col{i}'):
                    st.metric(
                        label=f"센서 {i} 최대 진도",
                        value=f"{max_magnitude:.3f}",
                        delta=f"평균: {avg_magnitude:.3f}"
                    )
            
            if composite_col in df_latest.columns:
                max_composite = df_latest[composite_col].max()
                
                if i <= 3:  # 첫 3개 컬럼만 사용
                    with eval(f'stat_col{i}'):
                        st.metric(
                            label=f"센서 {i} 최대 복합가속도",
                            value=f"{max_composite:.3f}"
                        )
        
        # 데이터 테이블 미리보기
        with st.expander("📋 데이터 테이블 미리보기 (처음 10행)"):
            st.dataframe(df_latest.head(10), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ 최신 파일 처리 오류: {e}")
        
        # 기본 정보라도 표시
        col_error1, col_error2 = st.columns(2)
        with col_error1:
            st.write(f"**📁 파일 경로:** {latest_file['filepath']}")
        with col_error2:
            if os.path.exists(latest_file['filepath']):
                file_size = os.path.getsize(latest_file['filepath'])
                st.write(f"**📄 파일 크기:** {file_size} bytes")
            else:
                st.write("**❌ 파일 존재하지 않음**")
    
    # =================== 이벤트 목록 렌더링 ===================
    render_event_list(files)
    
    # 자동 새로고침 처리 (컨텐츠 표시 후 맨 마지막에)
    if st.session_state.get('auto_refresh', True):
        # placeholder로 새로고침 알림만 표시
        refresh_placeholder = st.empty()
        refresh_placeholder.info(f"🔄 {CONFIG['AUTO_REFRESH_INTERVAL']}초 후 자동 새로고침...")
        
        # JavaScript를 이용한 새로고침 (더 부드러움)
        time.sleep(CONFIG['AUTO_REFRESH_INTERVAL'])
        st.cache_data.clear()
        st.rerun()

# ------------------------- 메인 실행 -------------------------
def main():
    """메인 함수"""
    # 세션 상태 초기화
    if 'last_file_count' not in st.session_state:
        st.session_state.last_file_count = 0
    
    # 모델 로딩 (세션에 저장)
    if 'model' not in st.session_state:
        st.session_state.model = load_earthquake_model()
        st.session_state.model_loaded = st.session_state.model is not None
    
    # UI 렌더링
    render_sidebar()
    render_dashboard()

if __name__ == "__main__":
    main()
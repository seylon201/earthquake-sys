#!/usr/bin/env python3
"""
지진 오보 분석 시스템 대시보드 유틸리티 함수들
파일 처리, 데이터 가져오기, 파싱 등의 공통 기능
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time

# =========================== 설정 ===========================

def get_base_path():
    """실행 환경에 따른 기본 경로 반환"""
    import platform
    import os
    
    # 현재 스크립트 위치 기준
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Windows에서 실행 중인지 확인
    if platform.system() == 'Windows':
        # Windows 경로 사용
        if 'C:\\' in current_dir:
            base_path = os.path.join(current_dir, 'influxLogs')
        else:
            # WSL 마운트 경로에서 Windows 경로로 변환
            base_path = current_dir.replace('/mnt/c/', 'C:/').replace('/', '\\') + '\\influxLogs'
    else:
        # Linux/WSL 경로 사용
        if current_dir.startswith('/mnt/c/'):
            base_path = os.path.join(current_dir, 'influxLogs')
        else:
            base_path = "/mnt/c/earthquake_project/influxLogs"
    
    return base_path

# 동적 경로 기반 CONFIG
BASE_PATH = get_base_path()

CONFIG = {
    'BASE_OUTPUT_DIR': BASE_PATH,
    'RAW_DATA_DIR': os.path.join(BASE_PATH, "base"),
    'ANALYSIS_DIR': os.path.join(BASE_PATH, "new2_analysis"),  # 지진 오보 분석 결과
    'ALERTS_DIR': os.path.join(BASE_PATH, "alerts"),  # 지진 경보 로그
    'PROCESSED_DATA_DIR': os.path.join(BASE_PATH, "processed"),  # 기존 호환
    'MODEL_PATHS': [  # 지진 오보 분석 시스템 모델 우선순위
        'new2_convlstm_3class_best.h5',
        'new2_convlstm_3class_final.h5', 
        'convlstm_3class_model.h5'
    ],
    'PORTS': [6060, 7001, 7053, 7060, 7070, 8010, 8080],
    'AUTO_REFRESH_INTERVAL': 10  # 10초마다 자동 새로고침
}

# =========================== 클래스 및 색상 정의 ===========================

# NEW2 시스템 클래스 정의
SYSTEM_CLASS_NAMES = {0: '지진', 1: '규칙적산업진동', 2: '불규칙생활진동'}
SYSTEM_CLASS_COLORS = {0: '🔴', 1: '🟠', 2: '🟢'}
SYSTEM_CLASS_COLOR_HEX = {0: '#FF4444', 1: '#FF8800', 2: '#00AA44'}

# =========================== 파일명 파싱 ===========================

def parse_filename_info(filename: str) -> Dict:
    """파일명에서 정보 추출
    
    예시: 
    - new2_ai_6060_2025-06-30_19-07-25.csv
    - event_6060_2025-06-30_19-07-25.csv
    """
    try:
        # 확장자 제거
        base_name = filename.replace('.csv', '').replace('.json', '')
        
        # 언더스코어로 분할
        parts = base_name.split('_')
        
        if len(parts) >= 4:
            # 포트 번호 추출
            port = parts[2] if parts[2].isdigit() else parts[1]
            
            # 날짜와 시간 부분 찾기
            date_part = None
            time_part = None
            
            for i, part in enumerate(parts):
                if '-' in part and len(part) == 10:  # YYYY-MM-DD 형식
                    date_part = part
                    if i + 1 < len(parts):
                        time_part = parts[i + 1]
                    break
            
            if date_part and time_part:
                # 시간 형식 변환: 19-07-25 -> 19:07:25
                time_formatted = time_part.replace('-', ':')
                datetime_str = f"{date_part} {time_formatted}"
                
                return {
                    'location': f"센서_{port}",
                    'port': port,
                    'datetime_str': datetime_str,
                    'date': date_part,
                    'time': time_formatted
                }
        
        # 파싱 실패시 기본값
        return {
            'location': 'Unknown',
            'port': 'Unknown',
            'datetime_str': '2025-01-01 00:00:00',
            'date': '2025-01-01',
            'time': '00:00:00'
        }
        
    except Exception as e:
        print(f"파일명 파싱 오류 {filename}: {e}")
        return {
            'location': 'Error',
            'port': 'Error',
            'datetime_str': '2025-01-01 00:00:00',
            'date': '2025-01-01',
            'time': '00:00:00'
        }

# =========================== 파일 가져오기 ===========================

def get_system_event_files() -> List[Dict]:
    """지진 오보 분석 시스템으로 분석된 이벤트 파일 목록 가져오기"""
    all_files = []
    
    # 디버깅: 디렉토리 존재 여부와 파일 목록 확인
    analysis_dir = CONFIG['ANALYSIS_DIR']
    if os.path.exists(analysis_dir):
        all_filenames = os.listdir(analysis_dir)
        csv_files = [f for f in all_filenames if f.endswith('.csv')]
        new2_ai_files = [f for f in csv_files if f.startswith('new2_ai_')]
        
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
    
    # 중복 제거 (같은 원본 이벤트에 대해 여러 분석 결과가 있을 경우 최신것만)
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

def get_earthquake_alerts() -> List[Dict]:
    """지진 경보 로그 가져오기"""
    alerts = []
    
    alerts_dir = CONFIG['ALERTS_DIR']
    if not os.path.exists(alerts_dir):
        return alerts
    
    try:
        for filename in os.listdir(alerts_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(alerts_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        alert_data = json.load(f)
                        alerts.append(alert_data)
                except:
                    continue
        
        # 시간순 정렬 (최신 경보가 위로)
        alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
    except Exception as e:
        print(f"경보 로그 가져오기 오류: {e}")
    
    return alerts

# =========================== 시간 유틸리티 ===========================

def should_auto_refresh() -> bool:
    """자동 새로고침이 필요한지 확인"""
    # 이 함수는 사용하지 않음 (수동 새로고침으로 변경됨)
    return False

def get_time_diff_text(target_datetime: datetime) -> str:
    """시간 차이를 사람이 읽기 쉬운 텍스트로 변환"""
    time_diff = datetime.now() - target_datetime
    
    if time_diff.total_seconds() < 60:
        return f"{int(time_diff.total_seconds())}초 전"
    elif time_diff.total_seconds() < 3600:
        return f"{int(time_diff.total_seconds()//60)}분 전"
    elif time_diff.total_seconds() < 86400:
        return f"{int(time_diff.total_seconds()//3600)}시간 전"
    else:
        return f"{time_diff.days}일 전"
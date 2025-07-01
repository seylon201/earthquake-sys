#!/usr/bin/env python3
"""
NEW2 ConvLSTM 3클래스 모델을 활용한 실시간 지진 조기경보 시스템
98.46% 정확도 모델 기반 오경보 저감 시스템
"""

import time
from datetime import datetime, timedelta
import requests
from influxdb_client import InfluxDBClient
from urllib.parse import quote
import os
import csv
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# 환경 설정
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow 안전한 로딩
def safe_load_tensorflow():
    """TensorFlow를 안전하게 로딩"""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import load_model
        print(f"✅ TensorFlow 버전: {tf.__version__}")
        return tf, load_model, True
    except Exception as e:
        print(f"❌ TensorFlow 로딩 실패: {e}")
        return None, None, False

# TensorFlow 로딩
tf, load_model, tf_available = safe_load_tensorflow()

# =========================== 시스템 설정 ===========================

# NEW2 ConvLSTM 3클래스 모델 설정
MODEL_CONFIG = {
    'model_path': 'new2_convlstm_3class_best.h5',  # NEW2 최고 성능 모델
    'backup_path': 'new2_convlstm_3class_final.h5',  # 백업 모델
    'input_shape': (1, 40, 3, 100, 1),  # NEW2 모델 입력 형태
    'accuracy': 0.9846,  # 98.46% 정확도
    'classes': {0: '지진', 1: '규칙적산업진동', 2: '불규칙생활진동'}
}

# 디렉토리 설정
DIRS = {
    'base': "C:/earthquake_modeling/earthquake_project_v3/influxLogs",
    'raw': "C:/earthquake_modeling/earthquake_project_v3/influxLogs/base",
    'processed': "C:/earthquake_modeling/earthquake_project_v3/influxLogs/processed",
    'alerts': "C:/earthquake_modeling/earthquake_project_v3/influxLogs/alerts"
}

# 디렉토리 생성
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# InfluxDB 설정
INFLUX_CONFIG = {
    'token': "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg==",
    'org': "kds",
    'bucket': "Lasung_3sensor of Max",
    'url': "http://118.129.145.82:8086",
    'ports': [6060, 7001, 7053, 7060, 7070, 8010, 8080]
}

# Node-RED 설정
NODERED_CONFIG = {
    'base_url': "http://118.129.145.82:8081/nodered/1min_event_lasung",
    'timeout': 30
}

# 조기경보 시스템 설정 (98.46% 모델 성능 기반)
ALERT_CONFIG = {
    'earthquake_threshold': 0.92,  # 지진 신뢰도 임계값 (92%)
    'confidence_gap_min': 0.25,   # 다른 클래스와 최소 신뢰도 차이 (25%)
    'enable_advanced_filter': True,  # 고급 필터링 활성화
    'enable_multi_check': True,      # 다중 검증 활성화
    'alert_cooldown': 60            # 경보 간 최소 간격 (초)
}

# 진동 특성 분석 임계값 (NEW2 데이터 기반 최적화)
VIBRATION_THRESHOLDS = {
    'min_duration_ratio': 0.20,     # 최소 진동 지속 비율 (20%)
    'max_frequency': 15.0,          # 최대 허용 주파수 (15Hz)
    'max_energy_imbalance': 0.70,   # 최대 에너지 불균형 (70%)
    'max_change_rate': 12.0,        # 최대 변화율
    'min_amplitude': 1.5,           # 최소 진폭
    'low_freq_dominance_min': 0.35  # 저주파 우세성 최소값 (35%)
}

# 시각적 표시
CLASS_DISPLAY = {
    0: {'name': '지진', 'color': '🔴', 'alert': True},
    1: {'name': '규칙적산업진동', 'color': '🟠', 'alert': False},
    2: {'name': '불규칙생활진동', 'color': '🟢', 'alert': False}
}

# =========================== 모델 로딩 ===========================

convlstm_model = None
if tf_available:
    print("🔄 NEW2 ConvLSTM 3클래스 모델 로딩 중...")
    try:
        # 메인 모델 시도
        if os.path.exists(MODEL_CONFIG['model_path']):
            convlstm_model = load_model(MODEL_CONFIG['model_path'])
            print(f"✅ 메인 모델 로딩 완료: {MODEL_CONFIG['model_path']}")
        # 백업 모델 시도
        elif os.path.exists(MODEL_CONFIG['backup_path']):
            convlstm_model = load_model(MODEL_CONFIG['backup_path'])
            print(f"✅ 백업 모델 로딩 완료: {MODEL_CONFIG['backup_path']}")
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다:")
            print(f"   메인: {MODEL_CONFIG['model_path']}")
            print(f"   백업: {MODEL_CONFIG['backup_path']}")
            
        if convlstm_model:
            print(f"🎯 모델 정확도: {MODEL_CONFIG['accuracy']*100:.2f}%")
            print(f"📊 입력 형태: {convlstm_model.input_shape}")
            print(f"📊 출력 형태: {convlstm_model.output_shape}")
            
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        convlstm_model = None

# InfluxDB 클라이언트 초기화
client = InfluxDBClient(
    url=INFLUX_CONFIG['url'], 
    token=INFLUX_CONFIG['token'], 
    org=INFLUX_CONFIG['org']
)
query_api = client.query_api()

# =========================== 데이터 전처리 함수 ===========================

def find_trigger_point_new2(data, threshold=3.0, min_ratio=0.15):
    """NEW2 데이터에 최적화된 트리거 포인트 탐지"""
    # 3축 합성 진도 계산
    magnitude = np.sqrt(np.sum(data**2, axis=1))
    
    # 동적 슬라이딩 윈도우로 트리거 탐지
    window_sizes = [100, 150, 200]  # 1초, 1.5초, 2초 윈도우
    
    for window_size in window_sizes:
        for i in range(0, len(magnitude) - window_size, 50):
            window = magnitude[i:i+window_size]
            high_intensity_count = np.sum(window >= threshold)
            
            if high_intensity_count >= (window_size * min_ratio):
                trigger_point = i + window_size // 2
                print(f"🎯 트리거 발견: {trigger_point}번째 샘플 (윈도우: {window_size})")
                return trigger_point
    
    return None

def analyze_vibration_characteristics_new2(raw_data):
    """NEW2 데이터에 최적화된 진동 특성 분석"""
    try:
        x_data, y_data, z_data = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]
        magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
        # 1. 진동 지속성 분석 (개선된 임계값 적용)
        high_intensity_mask = magnitude >= 2.5  # 낮은 임계값으로 더 민감하게
        duration_ratio = np.sum(high_intensity_mask) / len(magnitude)
        
        # 2. 주파수 분석 (FFT)
        fft = np.fft.fft(magnitude)
        freqs = np.fft.fftfreq(len(magnitude), d=0.01)
        power_spectrum = np.abs(fft)**2
        
        # 주파수별 에너지 분포
        low_freq_mask = (freqs >= 0.1) & (freqs <= 5.0)    # 지진 대역 (0.1-5Hz)
        mid_freq_mask = (freqs > 5.0) & (freqs <= 15.0)    # 산업 대역 (5-15Hz)
        high_freq_mask = (freqs > 15.0) & (freqs <= 50.0)  # 생활 대역 (15-50Hz)
        
        total_energy = np.sum(power_spectrum[freqs >= 0])
        if total_energy > 0:
            low_freq_energy_ratio = np.sum(power_spectrum[low_freq_mask]) / total_energy
            mid_freq_energy_ratio = np.sum(power_spectrum[mid_freq_mask]) / total_energy
            high_freq_energy_ratio = np.sum(power_spectrum[high_freq_mask]) / total_energy
        else:
            low_freq_energy_ratio = mid_freq_energy_ratio = high_freq_energy_ratio = 0.33
        
        # 주요 주파수 찾기
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = abs(freqs[dominant_freq_idx])
        
        # 3. 축간 에너지 분포 분석
        x_energy = np.sum(x_data**2)
        y_energy = np.sum(y_data**2)
        z_energy = np.sum(z_data**2)
        total_axis_energy = x_energy + y_energy + z_energy
        
        if total_axis_energy > 0:
            energy_balance = max(x_energy, y_energy, z_energy) / total_axis_energy
        else:
            energy_balance = 0.33
        
        # 4. 변화율 분석 (충격성 진동 감지)
        x_diff = np.diff(x_data)
        y_diff = np.diff(y_data)
        z_diff = np.diff(z_data)
        max_change_rate = max(np.max(np.abs(x_diff)), np.max(np.abs(y_diff)), np.max(np.abs(z_diff)))
        
        # 5. 신호 복잡도 분석
        x_autocorr = np.correlate(x_data, x_data, mode='full')
        if len(x_autocorr) > 200:
            autocorr_peak = np.max(x_autocorr[len(x_autocorr)//2+100:]) / np.max(x_autocorr)
        else:
            autocorr_peak = 0.5
        
        # 6. 진동 강도 통계
        amplitude_stats = {
            'max': np.max(magnitude),
            'mean': np.mean(magnitude),
            'std': np.std(magnitude),
            'percentile_95': np.percentile(magnitude, 95),
            'percentile_75': np.percentile(magnitude, 75)
        }
        
        # 7. 지진 특성 점수 계산 (0-1 스케일)
        earthquake_score = 0.0
        
        # 저주파 우세성 (지진의 특징)
        if low_freq_energy_ratio > 0.4:
            earthquake_score += 0.3
        elif low_freq_energy_ratio > 0.3:
            earthquake_score += 0.15
            
        # 적절한 지속성 (너무 짧거나 길지 않음)
        if 0.2 <= duration_ratio <= 0.8:
            earthquake_score += 0.2
        elif 0.15 <= duration_ratio <= 0.9:
            earthquake_score += 0.1
            
        # 균등한 축 분포 (지진의 특징)
        if energy_balance < 0.6:
            earthquake_score += 0.2
        elif energy_balance < 0.7:
            earthquake_score += 0.1
            
        # 적절한 주파수 범위
        if 0.5 <= dominant_freq <= 10.0:
            earthquake_score += 0.2
        elif 0.1 <= dominant_freq <= 15.0:
            earthquake_score += 0.1
            
        # 적절한 변화율 (너무 급격하지 않음)
        if max_change_rate < 8.0:
            earthquake_score += 0.1
        
        characteristics = {
            'duration_ratio': duration_ratio,
            'dominant_frequency': dominant_freq,
            'energy_balance': energy_balance,
            'max_change_rate': max_change_rate,
            'autocorr_peak': autocorr_peak,
            'amplitude_max': amplitude_stats['max'],
            'amplitude_mean': amplitude_stats['mean'],
            'amplitude_std': amplitude_stats['std'],
            'low_freq_dominance': low_freq_energy_ratio,
            'mid_freq_dominance': mid_freq_energy_ratio,
            'high_freq_dominance': high_freq_energy_ratio,
            'earthquake_score': earthquake_score,
            'x_energy_ratio': x_energy / total_axis_energy if total_axis_energy > 0 else 0.33,
            'y_energy_ratio': y_energy / total_axis_energy if total_axis_energy > 0 else 0.33,
            'z_energy_ratio': z_energy / total_axis_energy if total_axis_energy > 0 else 0.33
        }
        
        return characteristics
        
    except Exception as e:
        print(f"❌ 진동 특성 분석 실패: {e}")
        return None

def preprocess_for_new2_convlstm(csv_path):
    """NEW2 ConvLSTM 모델에 최적화된 전처리"""
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 원본 데이터 로드: {len(df)}행")
        
        # 센서 데이터 추출 (sensor_1 우선)
        sensor_cols = None
        for i in range(1, 4):
            x_col, y_col, z_col = f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                sensor_cols = [x_col, y_col, z_col]
                print(f"📡 사용 센서: sensor_{i}")
                break
        
        if not sensor_cols:
            return None, None, "센서 데이터 없음"
        
        # 3축 데이터 추출
        x_data = df[sensor_cols[0]].astype(float).values
        y_data = df[sensor_cols[1]].astype(float).values
        z_data = df[sensor_cols[2]].astype(float).values
        
        raw_data = np.stack([x_data, y_data, z_data], axis=1)
        print(f"📊 3축 데이터 결합: {raw_data.shape}")
        
        # 진동 특성 분석
        characteristics = analyze_vibration_characteristics_new2(raw_data)
        
        # NEW2 최적화된 트리거 포인트 탐지
        trigger_idx = find_trigger_point_new2(raw_data)
        if trigger_idx is None:
            print("⚠️ 트리거 포인트 없음 - 최대 진폭 지점 사용")
            magnitude = np.sqrt(np.sum(raw_data**2, axis=1))
            trigger_idx = np.argmax(magnitude)
        
        # NEW2 모델 형식에 맞춰 40초 슬라이싱
        PRE_SAMPLES = 1500   # 전 15초
        POST_SAMPLES = 2500  # 후 25초
        TOTAL_SAMPLES = 4000 # 총 40초
        
        start_idx = max(0, trigger_idx - PRE_SAMPLES)
        end_idx = min(len(raw_data), trigger_idx + POST_SAMPLES)
        
        # 데이터 추출 및 패딩/트림
        if end_idx - start_idx >= TOTAL_SAMPLES:
            sliced_data = raw_data[start_idx:start_idx + TOTAL_SAMPLES]
        else:
            available_data = raw_data[start_idx:end_idx]
            pad_length = TOTAL_SAMPLES - len(available_data)
            
            if pad_length > 0:
                # 앞쪽에 패딩 (배경 노이즈 시뮬레이션)
                noise_std = np.std(available_data) * 0.1
                front_pad = np.random.normal(0, noise_std, (pad_length//2, 3))
                back_pad = np.random.normal(0, noise_std, (pad_length - pad_length//2, 3))
                sliced_data = np.vstack([front_pad, available_data, back_pad])
            else:
                sliced_data = available_data
        
        # 정확히 4000 샘플로 맞추기
        if len(sliced_data) != TOTAL_SAMPLES:
            if len(sliced_data) > TOTAL_SAMPLES:
                sliced_data = sliced_data[:TOTAL_SAMPLES]
            else:
                pad_length = TOTAL_SAMPLES - len(sliced_data)
                padding = np.zeros((pad_length, 3))
                sliced_data = np.vstack([sliced_data, padding])
        
        print(f"✅ 40초 슬라이싱 완료: {sliced_data.shape}")
        
        # NEW2 ConvLSTM 입력 형태로 변환
        # (4000, 3) -> (40, 100, 3) -> (40, 3, 100) -> (40, 3, 100, 1) -> (1, 40, 3, 100, 1)
        reshaped = sliced_data.reshape(40, 100, 3)
        reshaped = np.transpose(reshaped, (0, 2, 1))  # (40, 3, 100)
        reshaped = np.expand_dims(reshaped, axis=-1)  # (40, 3, 100, 1)
        reshaped = np.expand_dims(reshaped, axis=0)   # (1, 40, 3, 100, 1)
        
        # NEW2 모델에 맞는 정규화 (z-score)
        mean = reshaped.mean()
        std = reshaped.std()
        
        if std > 0:
            normalized = (reshaped - mean) / std
            print(f"📐 정규화 완료: 평균={mean:.6f}, 표준편차={std:.6f}")
        else:
            normalized = reshaped
            print("⚠️ 표준편차가 0이므로 정규화 건너뜀")
        
        preprocess_info = {
            'original_length': len(raw_data),
            'trigger_point': trigger_idx,
            'slicing_range': (start_idx, end_idx),
            'final_shape': normalized.shape,
            'normalization': {'mean': float(mean), 'std': float(std)},
            'used_sensor': sensor_cols[0].split('_')[1],
            'characteristics': characteristics
        }
        
        return normalized, df, preprocess_info
        
    except Exception as e:
        print(f"❌ NEW2 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, str(e)

# =========================== 조기경보 로직 ===========================

def advanced_earthquake_detection_new2(predictions, characteristics):
    """NEW2 모델 기반 고도화된 지진 감지 로직"""
    
    # 기본 예측 결과
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    all_probs = predictions[0]
    
    # 클래스별 신뢰도
    earthquake_prob = all_probs[0]
    industrial_prob = all_probs[1]
    living_prob = all_probs[2]
    
    detection_info = {
        'original_class': predicted_class,
        'original_confidence': confidence,
        'earthquake_prob': earthquake_prob,
        'industrial_prob': industrial_prob,
        'living_prob': living_prob,
        'filters_applied': [],
        'pass_reasons': [],
        'suppression_reasons': []
    }
    
    print(f"\\n🧠 === NEW2 모델 예측 분석 ===")
    print(f"   🔴 지진: {earthquake_prob:.4f} ({earthquake_prob*100:.2f}%)")
    print(f"   🟠 규칙적산업진동: {industrial_prob:.4f} ({industrial_prob*100:.2f}%)")
    print(f"   🟢 불규칙생활진동: {living_prob:.4f} ({living_prob*100:.2f}%)")
    print(f"   🎯 최고 예측: {MODEL_CONFIG['classes'][predicted_class]} (신뢰도: {confidence:.4f})")
    
    # 1단계: 지진 예측이 아닌 경우 바로 통과
    if predicted_class != 0:
        detection_info['final_class'] = predicted_class
        detection_info['final_confidence'] = confidence
        detection_info['is_earthquake'] = False
        detection_info['alert_status'] = 'NO_ALERT'
        detection_info['reason'] = f"비지진 예측: {MODEL_CONFIG['classes'][predicted_class]}"
        return detection_info
    
    # 2단계: 지진으로 예측된 경우 다중 검증 시작
    print(f"\\n🔍 === 지진 예측 다중 검증 시작 ===")
    
    # 2-1: 신뢰도 임계값 검사
    if earthquake_prob < ALERT_CONFIG['earthquake_threshold']:
        detection_info['suppression_reasons'].append(
            f"지진 신뢰도 부족: {earthquake_prob:.3f} < {ALERT_CONFIG['earthquake_threshold']}"
        )
        detection_info['filters_applied'].append('confidence_threshold')
        print(f"🛡️ 신뢰도 필터: {detection_info['suppression_reasons'][-1]}")
        
        # 가장 높은 다른 클래스로 재분류
        other_probs = [industrial_prob, living_prob]
        other_classes = [1, 2]
        max_other_idx = np.argmax(other_probs)
        final_class = other_classes[max_other_idx]
        
        detection_info['final_class'] = final_class
        detection_info['final_confidence'] = other_probs[max_other_idx]
        detection_info['is_earthquake'] = False
        detection_info['alert_status'] = 'FALSE_POSITIVE_SUPPRESSED'
        detection_info['reason'] = f"오경보 억제: {detection_info['suppression_reasons'][-1]}"
        return detection_info
    
    detection_info['pass_reasons'].append(f"높은 지진 신뢰도: {earthquake_prob:.3f}")
    
    # 2-2: 신뢰도 차이 검사
    max_other_prob = max(industrial_prob, living_prob)
    confidence_gap = earthquake_prob - max_other_prob
    
    if confidence_gap < ALERT_CONFIG['confidence_gap_min']:
        detection_info['suppression_reasons'].append(
            f"신뢰도 차이 부족: {confidence_gap:.3f} < {ALERT_CONFIG['confidence_gap_min']}"
        )
        detection_info['filters_applied'].append('confidence_gap')
        print(f"🛡️ 신뢰도 차이 필터: {detection_info['suppression_reasons'][-1]}")
        
        # 두 번째로 높은 클래스로 재분류
        if industrial_prob > living_prob:
            final_class = 1
            final_confidence = industrial_prob
        else:
            final_class = 2
            final_confidence = living_prob
            
        detection_info['final_class'] = final_class
        detection_info['final_confidence'] = final_confidence
        detection_info['is_earthquake'] = False
        detection_info['alert_status'] = 'FALSE_POSITIVE_SUPPRESSED'
        detection_info['reason'] = f"오경보 억제: {detection_info['suppression_reasons'][-1]}"
        return detection_info
    
    detection_info['pass_reasons'].append(f"충분한 신뢰도 차이: {confidence_gap:.3f}")
    
    # 2-3: 진동 특성 기반 고급 검증 (활성화된 경우)
    if ALERT_CONFIG['enable_advanced_filter'] and characteristics:
        print(f"\\n🔬 === 진동 특성 기반 고급 검증 ===")
        
        # 지속성 검사
        duration_ratio = characteristics.get('duration_ratio', 0.5)
        if duration_ratio < VIBRATION_THRESHOLDS['min_duration_ratio']:
            detection_info['suppression_reasons'].append(
                f"진동 지속성 부족: {duration_ratio:.3f}"
            )
            detection_info['filters_applied'].append('duration_filter')
            print(f"🛡️ 지속성 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"충분한 지속성: {duration_ratio:.3f}")
        
        # 주파수 검사
        dominant_freq = characteristics.get('dominant_frequency', 10.0)
        if dominant_freq > VIBRATION_THRESHOLDS['max_frequency']:
            detection_info['suppression_reasons'].append(
                f"주파수가 높음: {dominant_freq:.1f}Hz (생활진동 의심)"
            )
            detection_info['filters_applied'].append('frequency_filter')
            print(f"🛡️ 주파수 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"적절한 주파수: {dominant_freq:.1f}Hz")
        
        # 에너지 균형 검사
        energy_balance = characteristics.get('energy_balance', 0.5)
        if energy_balance > VIBRATION_THRESHOLDS['max_energy_imbalance']:
            detection_info['suppression_reasons'].append(
                f"에너지 불균형: {energy_balance:.3f} (충격성 진동 의심)"
            )
            detection_info['filters_applied'].append('energy_filter')
            print(f"🛡️ 에너지 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"균등한 에너지 분포: {energy_balance:.3f}")
        
        # 변화율 검사
        max_change_rate = characteristics.get('max_change_rate', 5.0)
        if max_change_rate > VIBRATION_THRESHOLDS['max_change_rate']:
            detection_info['suppression_reasons'].append(
                f"급격한 변화: {max_change_rate:.3f} (충격성 진동 의심)"
            )
            detection_info['filters_applied'].append('change_rate_filter')
            print(f"🛡️ 변화율 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"완만한 변화: {max_change_rate:.3f}")
        
        # 저주파 우세성 검사
        low_freq_dominance = characteristics.get('low_freq_dominance', 0.5)
        if low_freq_dominance < VIBRATION_THRESHOLDS['low_freq_dominance_min']:
            detection_info['suppression_reasons'].append(
                f"고주파 우세: {low_freq_dominance:.3f} (생활진동 의심)"
            )
            detection_info['filters_applied'].append('freq_dominance_filter')
            print(f"🛡️ 주파수 우세성 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"저주파 우세: {low_freq_dominance:.3f}")
        
        # 지진 특성 점수 검사
        earthquake_score = characteristics.get('earthquake_score', 0.5)
        if earthquake_score < 0.4:  # 40% 미만이면 의심
            detection_info['suppression_reasons'].append(
                f"지진 특성 점수 낮음: {earthquake_score:.3f}"
            )
            detection_info['filters_applied'].append('earthquake_score_filter')
            print(f"🛡️ 지진 특성 필터: {detection_info['suppression_reasons'][-1]}")
        else:
            detection_info['pass_reasons'].append(f"높은 지진 특성 점수: {earthquake_score:.3f}")
        
        # 억제 조건이 3개 이상이면 오경보로 판단
        if len(detection_info['suppression_reasons']) >= 3:
            print(f"🛡️ 다중 필터 억제: {len(detection_info['suppression_reasons'])}개 조건 위반")
            
            # 특성 기반으로 적절한 클래스 선택
            if dominant_freq > 10 and max_change_rate > 8:
                final_class = 2  # 불규칙생활진동
                final_confidence = living_prob
            else:
                final_class = 1  # 규칙적산업진동
                final_confidence = industrial_prob
            
            detection_info['final_class'] = final_class
            detection_info['final_confidence'] = final_confidence
            detection_info['is_earthquake'] = False
            detection_info['alert_status'] = 'FALSE_POSITIVE_SUPPRESSED'
            detection_info['reason'] = f"다중 필터 억제: {', '.join(detection_info['suppression_reasons'][:2])} 등"
            return detection_info
    
    # 3단계: 모든 검증 통과 - 지진으로 확정
    print(f"\\n✅ === 모든 검증 통과: 지진 확정 ===")
    for reason in detection_info['pass_reasons']:
        print(f"   ✓ {reason}")
    
    detection_info['final_class'] = 0
    detection_info['final_confidence'] = earthquake_prob
    detection_info['is_earthquake'] = True
    detection_info['alert_status'] = 'EARTHQUAKE_ALERT'
    detection_info['reason'] = f"지진 확정: {len(detection_info['pass_reasons'])}개 조건 만족"
    
    return detection_info

def predict_and_analyze_new2(raw_csv_path, processed_csv_path):
    """NEW2 모델을 사용한 예측 및 분석"""
    if convlstm_model is None:
        print("⚠️ NEW2 모델이 로드되지 않아 예측을 건너뜁니다")
        return None
    
    try:
        print(f"\\n🔄 === NEW2 AI 분석 시작: {os.path.basename(raw_csv_path)} ===")
        
        # NEW2 전처리
        start_time = time.time()
        X, df, preprocess_info = preprocess_for_new2_convlstm(raw_csv_path)
        preprocess_time = time.time() - start_time
        
        if X is None:
            print(f"❌ NEW2 전처리 실패: {preprocess_info}")
            return None
        
        characteristics = preprocess_info.get('characteristics')
        
        # 진동 특성 출력
        if characteristics:
            print(f"\\n🔍 === 진동 특성 분석 결과 ===")
            print(f"   📏 지속성: {characteristics['duration_ratio']:.3f}")
            print(f"   🎵 주요 주파수: {characteristics['dominant_frequency']:.1f}Hz")
            print(f"   ⚖️ 에너지 균형: {characteristics['energy_balance']:.3f}")
            print(f"   📈 최대 변화율: {characteristics['max_change_rate']:.2f}")
            print(f"   🎯 지진 특성 점수: {characteristics['earthquake_score']:.3f}")
            print(f"   🔊 저주파 우세성: {characteristics['low_freq_dominance']:.3f}")
        
        # NEW2 모델 예측
        prediction_start = time.time()
        predictions = convlstm_model.predict(X, verbose=0)
        prediction_time = time.time() - prediction_start
        
        # 고도화된 지진 감지 로직 적용
        detection_result = advanced_earthquake_detection_new2(predictions, characteristics)
        
        total_time = time.time() - start_time
        
        # 결과 출력
        final_class = detection_result['final_class']
        final_confidence = detection_result['final_confidence']
        is_earthquake = detection_result['is_earthquake']
        alert_status = detection_result['alert_status']
        
        print(f"\\n🎯 === NEW2 최종 분석 결과 ===")
        print(f"원본 예측: {MODEL_CONFIG['classes'][detection_result['original_class']]} ({detection_result['original_confidence']:.4f})")
        print(f"최종 분류: {MODEL_CONFIG['classes'][final_class]} ({final_confidence:.4f})")
        print(f"분석 근거: {detection_result['reason']}")
        
        # 경보 상태 출력
        if is_earthquake:
            print(f"\\n🚨 === 지진 경보 발령! ===")
            print(f"🔴 지진 감지 확정")
            print(f"📊 신뢰도: {final_confidence:.4f} ({final_confidence*100:.2f}%)")
            print(f"⚡ 분석 시간: {total_time:.2f}초")
            
            # 경보 정보 저장
            alert_info = {
                'timestamp': datetime.now().isoformat(),
                'detection_confidence': final_confidence,
                'alert_type': 'EARTHQUAKE_ALERT',
                'model_accuracy': MODEL_CONFIG['accuracy'],
                'processing_time': total_time,
                'characteristics': characteristics
            }
            
            alert_file = os.path.join(DIRS['alerts'], f"earthquake_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alert_info, f, ensure_ascii=False, indent=2)
            print(f"🚨 경보 정보 저장: {os.path.basename(alert_file)}")
            
        else:
            if alert_status == 'FALSE_POSITIVE_SUPPRESSED':
                print(f"\\n🛡️ === 오경보 억제됨 ===")
                print(f"🟡 잠재적 오경보 차단")
            else:
                print(f"\\n✅ === 정상 상태 ===")
                print(f"🟢 비지진 진동 감지")
            
            print(f"📊 분류: {MODEL_CONFIG['classes'][final_class]}")
            print(f"📊 신뢰도: {final_confidence:.4f} ({final_confidence*100:.2f}%)")
            print(f"⚡ 분석 시간: {total_time:.2f}초")
        
        # 결과 데이터프레임에 추가
        if df is not None:
            # 기본 예측 정보
            df['original_predicted_class'] = detection_result['original_class']
            df['original_predicted_class_name'] = MODEL_CONFIG['classes'][detection_result['original_class']]
            df['original_confidence'] = detection_result['original_confidence']
            
            # 최종 분류 정보
            df['final_predicted_class'] = final_class
            df['final_predicted_class_name'] = MODEL_CONFIG['classes'][final_class]
            df['final_confidence'] = final_confidence
            df['is_earthquake'] = is_earthquake
            df['alert_status'] = alert_status
            
            # 모든 클래스 확률
            df['prob_earthquake'] = detection_result['earthquake_prob']
            df['prob_industrial'] = detection_result['industrial_prob']
            df['prob_living'] = detection_result['living_prob']
            
            # 분석 정보
            df['model_name'] = 'NEW2_ConvLSTM_3Class'
            df['model_accuracy'] = MODEL_CONFIG['accuracy']
            df['preprocess_time'] = preprocess_time
            df['prediction_time'] = prediction_time
            df['total_analysis_time'] = total_time
            
            # 전처리 정보
            df['trigger_point'] = preprocess_info['trigger_point']
            df['original_data_length'] = preprocess_info['original_length']
            df['used_sensor'] = preprocess_info['used_sensor']
            df['normalization_mean'] = preprocess_info['normalization']['mean']
            df['normalization_std'] = preprocess_info['normalization']['std']
            
            # 진동 특성 정보
            if characteristics:
                for key, value in characteristics.items():
                    df[f'vibration_{key}'] = value
            
            # 검증 정보
            df['filters_applied'] = '|'.join(detection_result['filters_applied'])
            df['suppression_reasons'] = '|'.join(detection_result['suppression_reasons'])
            df['pass_reasons'] = '|'.join(detection_result['pass_reasons'])
            
            # 시스템 설정 정보
            df['earthquake_threshold'] = ALERT_CONFIG['earthquake_threshold']
            df['confidence_gap_min'] = ALERT_CONFIG['confidence_gap_min']
            df['advanced_filter_enabled'] = ALERT_CONFIG['enable_advanced_filter']
            
            # 처리된 결과 저장
            df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 NEW2 분석 결과 저장: {os.path.basename(processed_csv_path)}")
        
        return {
            'is_earthquake': is_earthquake,
            'final_class': final_class,
            'final_class_name': MODEL_CONFIG['classes'][final_class],
            'final_confidence': final_confidence,
            'alert_status': alert_status,
            'detection_result': detection_result,
            'preprocess_info': preprocess_info,
            'processing_time': total_time
        }
        
    except Exception as e:
        print(f"❌ NEW2 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# =========================== 유틸리티 함수 ===========================

def is_file_already_processed(raw_filename):
    """파일 중복 처리 방지"""
    ai_filename = raw_filename.replace("event_", "new2_ai_")
    processed_path = os.path.join(DIRS['processed'], ai_filename)
    return os.path.exists(processed_path)

def system_health_check_new2():
    """NEW2 시스템 건강성 체크"""
    health = {
        'tensorflow_available': tf_available,
        'new2_model_loaded': convlstm_model is not None,
        'directories_ready': all(os.path.exists(path) for path in DIRS.values()),
        'influxdb_connected': False,
        'model_info': None
    }
    
    # InfluxDB 연결 테스트
    try:
        test_query = f'from(bucket: "{INFLUX_CONFIG["bucket"]}") |> range(start: -1m) |> limit(n:1)'
        result = query_api.query(org=INFLUX_CONFIG['org'], query=test_query)
        health['influxdb_connected'] = True
    except:
        health['influxdb_connected'] = False
    
    # 모델 정보
    if convlstm_model:
        health['model_info'] = {
            'input_shape': convlstm_model.input_shape,
            'output_shape': convlstm_model.output_shape,
            'accuracy': MODEL_CONFIG['accuracy'],
            'classes': MODEL_CONFIG['classes']
        }
    
    return health

# =========================== 통계 및 모니터링 ===========================

# NEW2 성능 통계
new2_stats = {
    'total_events': 0,
    'earthquake_alerts': 0,
    'false_positive_suppressed': 0,
    'normal_detections': 0,
    'class_distribution': {name: 0 for name in MODEL_CONFIG['classes'].values()},
    'processing_times': [],
    'alert_history': [],
    'suppression_effectiveness': {
        'confidence_threshold': 0,
        'confidence_gap': 0,
        'duration_filter': 0,
        'frequency_filter': 0,
        'energy_filter': 0,
        'change_rate_filter': 0,
        'freq_dominance_filter': 0,
        'earthquake_score_filter': 0,
        'multi_filter': 0
    },
    'start_time': datetime.now()
}

def update_new2_stats(analysis_result):
    """NEW2 통계 업데이트"""
    if not analysis_result:
        return
    
    new2_stats['total_events'] += 1
    new2_stats['processing_times'].append(analysis_result['processing_time'])
    
    # 클래스별 분포
    class_name = analysis_result['final_class_name']
    new2_stats['class_distribution'][class_name] += 1
    
    # 경보 상태별 통계
    if analysis_result['is_earthquake']:
        new2_stats['earthquake_alerts'] += 1
        new2_stats['alert_history'].append({
            'timestamp': datetime.now().isoformat(),
            'confidence': analysis_result['final_confidence'],
            'class': class_name
        })
    elif analysis_result['alert_status'] == 'FALSE_POSITIVE_SUPPRESSED':
        new2_stats['false_positive_suppressed'] += 1
    else:
        new2_stats['normal_detections'] += 1
    
    # 필터 효과 통계
    detection_result = analysis_result.get('detection_result', {})
    filters_applied = detection_result.get('filters_applied', [])
    
    for filter_name in filters_applied:
        if filter_name in new2_stats['suppression_effectiveness']:
            new2_stats['suppression_effectiveness'][filter_name] += 1
    
    if len(filters_applied) >= 3:
        new2_stats['suppression_effectiveness']['multi_filter'] += 1

def print_new2_stats():
    """NEW2 통계 출력"""
    if new2_stats['total_events'] == 0:
        return
    
    runtime = datetime.now() - new2_stats['start_time']
    avg_processing_time = np.mean(new2_stats['processing_times'])
    
    print(f"\\n📈 === NEW2 시스템 성능 리포트 ===")
    print(f"🕒 실행 시간: {runtime}")
    print(f"📊 총 이벤트: {new2_stats['total_events']}건")
    print(f"🚨 지진 경보: {new2_stats['earthquake_alerts']}건")
    print(f"🛡️ 오경보 억제: {new2_stats['false_positive_suppressed']}건")
    print(f"✅ 정상 감지: {new2_stats['normal_detections']}건")
    print(f"⚡ 평균 처리 시간: {avg_processing_time:.2f}초")
    
    # 비율 계산
    alert_rate = new2_stats['earthquake_alerts'] / new2_stats['total_events'] * 100
    suppression_rate = new2_stats['false_positive_suppressed'] / new2_stats['total_events'] * 100
    
    print(f"📈 지진 경보율: {alert_rate:.1f}%")
    print(f"🛡️ 오경보 억제율: {suppression_rate:.1f}%")
    
    # 오경보 저감 효과
    if new2_stats['false_positive_suppressed'] > 0:
        potential_false_alarms = new2_stats['earthquake_alerts'] + new2_stats['false_positive_suppressed']
        reduction_effectiveness = (new2_stats['false_positive_suppressed'] / potential_false_alarms) * 100
        print(f"🎯 오경보 저감 효과: {reduction_effectiveness:.1f}%")
    
    # 클래스별 분포
    print(f"\\n📊 클래스별 분포:")
    for class_name, count in new2_stats['class_distribution'].items():
        if count > 0:
            percentage = count / new2_stats['total_events'] * 100
            icon = '🔴' if class_name == '지진' else '🟠' if class_name == '규칙적산업진동' else '🟢'
            print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")
    
    # 필터 효과
    print(f"\\n🛡️ 필터별 억제 횟수:")
    for filter_name, count in new2_stats['suppression_effectiveness'].items():
        if count > 0:
            print(f"   📌 {filter_name}: {count}건")

# =========================== 메인 시스템 시작 ===========================

def main():
    """NEW2 실시간 지진 조기경보 시스템 메인"""
    
    print("\\n🚀 === NEW2 ConvLSTM 실시간 지진 조기경보 시스템 시작! ===")
    print(f"🧠 모델: NEW2 ConvLSTM 3클래스 (정확도: {MODEL_CONFIG['accuracy']*100:.2f}%)")
    print(f"📊 분류: {list(MODEL_CONFIG['classes'].values())}")
    print(f"🔍 감시 포트: {INFLUX_CONFIG['ports']}")
    print(f"⏱️ 체크 주기: 1초")
    print(f"🌐 Node-RED: {NODERED_CONFIG['base_url']}")
    print(f"💾 저장 위치:")
    for name, path in DIRS.items():
        print(f"   {name}: {path}")
    
    print(f"\\n🛡️ 조기경보 설정:")
    print(f"   📊 지진 신뢰도 임계값: {ALERT_CONFIG['earthquake_threshold']*100:.0f}%")
    print(f"   📏 신뢰도 차이 최소값: {ALERT_CONFIG['confidence_gap_min']*100:.0f}%")
    print(f"   🔬 고급 필터링: {'활성화' if ALERT_CONFIG['enable_advanced_filter'] else '비활성화'}")
    print(f"   🔄 다중 검증: {'활성화' if ALERT_CONFIG['enable_multi_check'] else '비활성화'}")
    
    # 시스템 건강성 체크
    health = system_health_check_new2()
    print(f"\\n🏥 시스템 상태:")
    print(f"   ✅ TensorFlow: {'정상' if health['tensorflow_available'] else '❌ 오류'}")
    print(f"   ✅ NEW2 모델: {'로딩됨' if health['new2_model_loaded'] else '❌ 실패'}")
    print(f"   ✅ 디렉토리: {'준비됨' if health['directories_ready'] else '❌ 오류'}")
    print(f"   ✅ InfluxDB: {'연결됨' if health['influxdb_connected'] else '❌ 연결 실패'}")
    
    if health['model_info']:
        print(f"\\n🧠 모델 정보:")
        print(f"   입력 형태: {health['model_info']['input_shape']}")
        print(f"   출력 형태: {health['model_info']['output_shape']}")
    
    print("="*80)
    
    # 실시간 감시 루프
    consecutive_no_data = 0
    last_alert_time = 0
    
    try:
        while True:
            now = datetime.utcnow()
            start = now - timedelta(seconds=1)
            data_found = False
            
            for port in INFLUX_CONFIG['ports']:
                query = f'''
                from(bucket: "{INFLUX_CONFIG["bucket"]}")
                  |> range(start: {start.isoformat()}Z, stop: {now.isoformat()}Z)
                  |> filter(fn: (r) => r._field == "intensity" and r._measurement == "{port}")
                  |> sort(columns: ["_time"], desc: true)
                  |> limit(n:1)
                '''
                
                result = query_api.query(org=INFLUX_CONFIG['org'], query=query)
                
                for table in result:
                    for record in table.records:
                        data_found = True
                        intensity = record.get_value()
                        
                        if not isinstance(intensity, (int, float)) or intensity < 3.0:
                            continue
                        
                        event_time = record.get_time().astimezone()
                        kst_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 경보 쿨다운 체크
                        current_time = time.time()
                        if current_time - last_alert_time < ALERT_CONFIG['alert_cooldown']:
                            print(f"⏰ 경보 쿨다운 중... ({current_time - last_alert_time:.0f}초)")
                            continue
                        
                        print(f"\\n🔥 === 진도 {intensity:.2f} 감지 ===")
                        print(f"📅 시간: {kst_time}")
                        print(f"🌐 포트: {port}")
                        
                        # 파일명 생성
                        safe_time = kst_time.replace(':', '-').replace(' ', '_')
                        raw_filename = f"event_{port}_{safe_time}.csv"
                        
                        # 중복 처리 방지
                        if is_file_already_processed(raw_filename):
                            print(f"⚠️ 이미 처리된 이벤트: {raw_filename}")
                            continue
                        
                        print("⏳ 40초 데이터 수집 대기...")
                        
                        # 대기 중 진행 상황 표시
                        for i in range(8):
                            remaining = 40 - (i * 5)
                            print(f"   🕐 {remaining}초 남음...")
                            time.sleep(5)
                        
                        # Node-RED 데이터 요청
                        encoded_time = quote(kst_time)
                        url = f"{NODERED_CONFIG['base_url']}/{encoded_time}/{port}"
                        
                        try:
                            print(f"🔗 Node-RED 데이터 요청: {url}")
                            response = requests.get(url, timeout=NODERED_CONFIG['timeout'])
                            
                            if response.status_code != 200:
                                print(f"❌ Node-RED 응답 오류: {response.status_code}")
                                continue
                            
                            # 파일 경로
                            raw_csv_path = os.path.join(DIRS['raw'], raw_filename)
                            ai_filename = raw_filename.replace("event_", "new2_ai_")
                            processed_csv_path = os.path.join(DIRS['processed'], ai_filename)
                            
                            # 파일 생성 대기
                            max_wait = 20
                            for wait_count in range(max_wait):
                                if os.path.exists(raw_csv_path):
                                    file_size = os.path.getsize(raw_csv_path)
                                    print(f"✅ 데이터 파일 확인: {file_size} bytes")
                                    break
                                print(f"⏳ 파일 생성 대기... ({wait_count + 1}/{max_wait})")
                                time.sleep(1)
                            else:
                                print(f"❌ 데이터 파일을 찾을 수 없음: {raw_csv_path}")
                                continue
                            
                            # NEW2 AI 분석 수행
                            analysis_result = predict_and_analyze_new2(raw_csv_path, processed_csv_path)
                            
                            if analysis_result:
                                # 통계 업데이트
                                update_new2_stats(analysis_result)
                                
                                # 지진 경보 처리
                                if analysis_result['is_earthquake']:
                                    last_alert_time = current_time
                                    print(f"\\n🚨🚨🚨 지진 경보 발령! 🚨🚨🚨")
                                    print(f"🔴 신뢰도: {analysis_result['final_confidence']*100:.2f}%")
                                    print(f"⚡ 처리 시간: {analysis_result['processing_time']:.2f}초")
                                
                                # 주기적 통계 출력
                                if new2_stats['total_events'] % 5 == 0:
                                    print_new2_stats()
                            
                        except requests.exceptions.Timeout:
                            print(f"❌ Node-RED 요청 타임아웃")
                        except Exception as e:
                            print(f"❌ Node-RED 요청 실패: {e}")
            
            # 무데이터 모니터링
            if not data_found:
                consecutive_no_data += 1
                if consecutive_no_data % 300 == 0:  # 5분마다
                    print(f"⏰ 데이터 없음 ({consecutive_no_data}초 지속)")
            else:
                consecutive_no_data = 0
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\\n\\n🛑 === 사용자 중단 ===")
        print_new2_stats()
        print(f"\\n🎉 NEW2 시스템이 안전하게 종료되었습니다!")
    
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
        print_new2_stats()

if __name__ == "__main__":
    main()
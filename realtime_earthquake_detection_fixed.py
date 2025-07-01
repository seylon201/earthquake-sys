import time
from datetime import datetime, timedelta
import requests
from influxdb_client import InfluxDBClient
from urllib.parse import quote
import os
import csv
import pandas as pd
import numpy as np

# 환경 변수 설정 (protobuf 문제 해결)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow 안전한 로딩
def safe_load_tensorflow():
    """TensorFlow를 안전하게 로딩하는 함수"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        # TensorFlow 로딩 시도
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        # 버전 체크
        tf_version = tf.__version__
        print(f"✅ TensorFlow 버전: {tf_version}")
        
        from tensorflow.keras.models import load_model
        return tf, load_model, True
        
    except Exception as e:
        print(f"❌ TensorFlow 로딩 실패: {e}")
        print("💡 해결 방법:")
        print("   1. pip install tensorflow==2.13.0")
        print("   2. pip install protobuf==3.20.3")
        print("   3. 시스템 재시작 후 재시도")
        return None, None, False

# TensorFlow 로딩 시도
tf, load_model, tf_available = safe_load_tensorflow()

# ✅ 3클래스 모델 경로
MODEL_PATH = 'convlstm_3class_model.h5'

# ✅ 저장 디렉토리 설정
BASE_SAVE_DIR = "C:/earthquake_modeling/earthquake_project_v3/influxLogs"
RAW_DATA_DIR = os.path.join(BASE_SAVE_DIR, "base")  # 원시 데이터
PROCESSED_DATA_DIR = os.path.join(BASE_SAVE_DIR, "processed")  # AI 처리 결과
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ✅ 3클래스 정의
CLASS_NAMES = {0: '지진', 1: '불규칙생활', 2: '모터진동'}
CLASS_COLORS = {0: '🔴', 1: '🟢', 2: '🟠'}

# 🛡️ 오경보 저감 설정 (핵심 수정 부분)
EARTHQUAKE_CONFIDENCE_THRESHOLD = 0.88  # 지진 분류를 위한 최소 신뢰도 (88%)
ENABLE_MULTI_CRITERIA_FILTER = True      # 다중 조건 필터 사용 여부
FALSE_POSITIVE_LOG = True                # 오경보 억제 로그 출력
ENABLE_ADVANCED_FILTERING = True         # 고급 필터링 활성화

# 🔧 고급 필터 임계값들
ADVANCED_FILTER_THRESHOLDS = {
    'duration_ratio': 0.25,          # 지속성 최소 비율 (25%)
    'max_frequency': 18,             # 최대 허용 주파수 (18Hz)
    'energy_balance_max': 0.65,      # 에너지 불균형 최대값 (65%)
    'max_change_rate': 10,           # 최대 변화율
    'confidence_gap_min': 0.20,      # 최소 신뢰도 차이 (20%)
    'consecutive_checks': 1          # 연속 검증 횟수
}

# 모델 로딩 (안전하게)
convlstm_model = None
if tf_available and load_model:
    print("🔄 3클래스 ConvLSTM 모델 로딩 중...")
    try:
        convlstm_model = load_model(MODEL_PATH)
        print("✅ 3클래스 모델 로딩 완료!")
        print(f"🔍 모델 입력 형태: {convlstm_model.input_shape}")
        print(f"🔍 모델 출력 형태: {convlstm_model.output_shape}")
        print(f"🛡️ 지진 신뢰도 임계값: {EARTHQUAKE_CONFIDENCE_THRESHOLD*100:.0f}%")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        print("⚠️ AI 예측 없이 데이터 수집만 수행합니다.")
        convlstm_model = None
else:
    print("⚠️ TensorFlow를 사용할 수 없어 AI 예측을 건너뜁니다.")

# ✅ InfluxDB 설정
INFLUX_TOKEN = "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg=="
INFLUX_ORG = "kds"
INFLUX_BUCKET = "Lasung_3sensor of Max"
INFLUX_URL = "http://118.129.145.82:8086"
PORTS = [6060, 7001, 7053, 7060, 7070, 8010, 8080]
CHECK_INTERVAL = 1

# ✅ Node-RED 설정
NODERED_BASE_URL = "http://118.129.145.82:8081/nodered/1min_event_lasung"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# ✅ ===== 데이터 처리 함수들 =====

def flatten_row(row):
    """기존 코드의 데이터 평면화 함수 (3센서 지원)"""
    flat = {
        "timestamp": "'" + row.get("timestamp"),
        "counter": row.get("counter")
    }
    for i in range(1, 4):  # sensor_1, sensor_2, sensor_3
        sensor_key = f"sensor_{i}"
        sensor_data = row.get(sensor_key, {})
        for key, value in sensor_data.items():
            flat[f"{sensor_key}_{key}"] = value
    return flat

def find_trigger_point(data, threshold=3.0, min_ratio=0.25):
    """학습 데이터와 동일한 트리거 포인트 탐지"""
    # 3축 합성 진도 계산
    magnitude = np.sqrt(np.sum(data**2, axis=1))
    
    # 1초 윈도우(100샘플)로 스캔
    for i in range(0, len(magnitude) - 100, 100):
        window = magnitude[i:i+100]
        high_intensity_count = np.sum(window >= threshold)
        
        if high_intensity_count >= (100 * min_ratio):  # 25개 이상
            return i + 50  # 윈도우 중앙점을 트리거로 사용
    
    return None

# 🛡️ 새로운 진동 특성 분석 함수 (오경보 저감용)
def analyze_vibration_characteristics(raw_data):
    """진동 특성 분석으로 오경보 저감 지원"""
    try:
        # 3축 데이터 분리
        x_data = raw_data[:, 0]
        y_data = raw_data[:, 1] 
        z_data = raw_data[:, 2]
        
        # 합성 진도 계산
        magnitude = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
        # 1. 진동 지속성 분석 (전체 구간 중 고진동 비율)
        high_intensity_mask = magnitude >= 3.0
        duration_ratio = np.sum(high_intensity_mask) / len(magnitude)
        
        # 2. 주파수 분석 (FFT)
        fft = np.fft.fft(magnitude)
        freqs = np.fft.fftfreq(len(magnitude), d=0.01)  # 100Hz 샘플링
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_freq = abs(freqs[dominant_freq_idx])
        
        # 3. 축간 에너지 분포 분석
        x_energy = np.sum(x_data**2)
        y_energy = np.sum(y_data**2)
        z_energy = np.sum(z_data**2)
        total_energy = x_energy + y_energy + z_energy
        
        if total_energy > 0:
            x_ratio = x_energy / total_energy
            y_ratio = y_energy / total_energy
            z_ratio = z_energy / total_energy
            energy_balance = max(x_ratio, y_ratio, z_ratio)  # 한 축에 에너지가 집중되는 정도
        else:
            energy_balance = 0.33
        
        # 4. 급격한 변화율 분석 (충격성 진동 감지)
        x_diff = np.diff(x_data)
        y_diff = np.diff(y_data)
        z_diff = np.diff(z_data)
        max_change_rate = max(np.max(np.abs(x_diff)), np.max(np.abs(y_diff)), np.max(np.abs(z_diff)))
        
        # 5. 진동 패턴의 규칙성 분석
        if len(x_data) > 200:  # 충분한 데이터가 있는 경우만
            x_autocorr = np.correlate(x_data, x_data, mode='full')
            autocorr_peak = np.max(x_autocorr[len(x_autocorr)//2+100:]) / np.max(x_autocorr)
        else:
            autocorr_peak = 0.5
        
        # 6. 주파수 대역별 에너지 분석
        low_freq_energy = np.sum(np.abs(fft[freqs <= 10]))    # 10Hz 이하
        high_freq_energy = np.sum(np.abs(fft[freqs > 10]))    # 10Hz 초과
        low_freq_dominance = low_freq_energy / (low_freq_energy + high_freq_energy) if (low_freq_energy + high_freq_energy) > 0 else 0.5
        
        characteristics = {
            'duration_ratio': duration_ratio,           # 고진동 지속 비율
            'dominant_frequency': abs(dominant_freq),   # 주요 주파수
            'energy_balance': energy_balance,           # 에너지 불균형 정도
            'max_change_rate': max_change_rate,         # 최대 변화율
            'autocorr_peak': autocorr_peak,            # 자기상관 피크 (규칙성)
            'max_amplitude': np.max(magnitude),         # 최대 진폭
            'mean_amplitude': np.mean(magnitude),       # 평균 진폭
            'std_amplitude': np.std(magnitude),         # 진폭 표준편차
            'low_freq_dominance': low_freq_dominance,   # 저주파수 우세성
            'x_energy_ratio': x_energy / total_energy if total_energy > 0 else 0.33,
            'y_energy_ratio': y_energy / total_energy if total_energy > 0 else 0.33,
            'z_energy_ratio': z_energy / total_energy if total_energy > 0 else 0.33
        }
        
        return characteristics
        
    except Exception as e:
        print(f"❌ 진동 특성 분석 실패: {e}")
        return None

# 🛡️ 향상된 지진 감지 로직 (핵심 오경보 저감 함수)
def enhanced_earthquake_detection_logic(predictions, characteristics):
    """향상된 지진 감지 로직 - 다중 기준 적용하여 오경보 저감"""
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # 기본 예측이 지진이 아닌 경우 - 바로 허용
    if predicted_class != 0:
        return predicted_class, confidence, False, f"모델 예측: {CLASS_NAMES[predicted_class]} (신뢰도: {confidence:.3f})"
    
    # 지진으로 예측된 경우 추가 검증 시작
    suppression_reasons = []
    pass_reasons = []
    
    # 1. 신뢰도 검사 (가장 중요한 필터)
    if confidence < EARTHQUAKE_CONFIDENCE_THRESHOLD:
        suppression_reasons.append(f"신뢰도 부족: {confidence:.3f} < {EARTHQUAKE_CONFIDENCE_THRESHOLD}")
        if FALSE_POSITIVE_LOG:
            print(f"🛡️ 1차 필터: {suppression_reasons[0]}")
        return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[0]}"
    pass_reasons.append(f"높은 신뢰도: {confidence:.3f}")
    
    # 2. 다중 조건 필터 (활성화된 경우만)
    if ENABLE_MULTI_CRITERIA_FILTER and characteristics:
        
        # 지속성 검사 (지진은 보통 더 오래 지속)
        if characteristics['duration_ratio'] < ADVANCED_FILTER_THRESHOLDS['duration_ratio']:
            suppression_reasons.append(f"진동 지속성 부족: {characteristics['duration_ratio']:.3f}")
            if FALSE_POSITIVE_LOG:
                print(f"🛡️ 지속성 필터: {suppression_reasons[-1]}")
            return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
        pass_reasons.append(f"충분한 지속성: {characteristics['duration_ratio']:.3f}")
        
        # 주파수 검사 (생활진동은 보통 고주파수)
        if characteristics['dominant_frequency'] > ADVANCED_FILTER_THRESHOLDS['max_frequency']:
            suppression_reasons.append(f"주파수가 높음: {characteristics['dominant_frequency']:.1f}Hz (생활진동 의심)")
            if FALSE_POSITIVE_LOG:
                print(f"🛡️ 주파수 필터: {suppression_reasons[-1]}")
            return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
        pass_reasons.append(f"적절한 주파수: {characteristics['dominant_frequency']:.1f}Hz")
        
        # 에너지 균형 검사 (한 축에만 집중된 에너지는 충격성 진동)
        if characteristics['energy_balance'] > ADVANCED_FILTER_THRESHOLDS['energy_balance_max']:
            suppression_reasons.append(f"에너지 불균형: {characteristics['energy_balance']:.3f} (충격성 진동 의심)")
            if FALSE_POSITIVE_LOG:
                print(f"🛡️ 에너지 필터: {suppression_reasons[-1]}")
            return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
        pass_reasons.append(f"균등한 에너지 분포: {characteristics['energy_balance']:.3f}")
        
        # 급격한 변화율 검사 (너무 급격한 변화는 충격성)
        if characteristics['max_change_rate'] > ADVANCED_FILTER_THRESHOLDS['max_change_rate']:
            suppression_reasons.append(f"급격한 변화: {characteristics['max_change_rate']:.3f} (충격성 진동 의심)")
            if FALSE_POSITIVE_LOG:
                print(f"🛡️ 변화율 필터: {suppression_reasons[-1]}")
            return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
        pass_reasons.append(f"완만한 변화: {characteristics['max_change_rate']:.3f}")
        
        # 🆕 저주파수 우세성 검사 (지진은 저주파가 우세)
        if ENABLE_ADVANCED_FILTERING and characteristics['low_freq_dominance'] < 0.4:  # 40% 미만이면 의심
            suppression_reasons.append(f"고주파 우세: {characteristics['low_freq_dominance']:.3f} (생활진동 의심)")
            if FALSE_POSITIVE_LOG:
                print(f"🛡️ 주파수 우세성 필터: {suppression_reasons[-1]}")
            return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
        pass_reasons.append(f"저주파 우세: {characteristics['low_freq_dominance']:.3f}")
    
    # 3. 다른 클래스와의 신뢰도 차이 검사
    other_confidences = [predictions[0][i] for i in range(3) if i != 0]
    max_other_confidence = max(other_confidences)
    confidence_gap = confidence - max_other_confidence
    
    if confidence_gap < ADVANCED_FILTER_THRESHOLDS['confidence_gap_min']:
        suppression_reasons.append(f"다른 클래스와 신뢰도 차이 부족: {confidence_gap:.3f}")
        if FALSE_POSITIVE_LOG:
            print(f"🛡️ 신뢰도 차이 필터: {suppression_reasons[-1]}")
        return 1, confidence, True, f"🛡️ 오경보 억제: {suppression_reasons[-1]}"
    pass_reasons.append(f"충분한 신뢰도 차이: {confidence_gap:.3f}")
    
    # 모든 검증 통과 - 지진으로 확정
    if FALSE_POSITIVE_LOG:
        print(f"✅ 모든 필터 통과: 지진으로 확정")
    return 0, confidence, False, f"✅ 지진 확정: {' | '.join(pass_reasons)}"

def preprocess_for_convlstm(csv_path):
    """학습 데이터와 완전히 동일한 전처리 적용 (3센서 지원)"""
    try:
        df = pd.read_csv(csv_path)
        
        # 센서 선택 (sensor_1 우선, 없으면 다른 센서 사용)
        sensor_cols = []
        for i in range(1, 4):
            x_col = f'sensor_{i}_x'
            y_col = f'sensor_{i}_y' 
            z_col = f'sensor_{i}_z'
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                sensor_cols = [x_col, y_col, z_col]
                print(f"📡 사용 센서: sensor_{i}")
                break
        
        if not sensor_cols:
            print("❌ 사용 가능한 센서 데이터가 없습니다")
            return None, None, "센서 데이터 없음"
        
        # 3축 데이터 추출
        sensor_x = df[sensor_cols[0]].astype(float).values
        sensor_y = df[sensor_cols[1]].astype(float).values
        sensor_z = df[sensor_cols[2]].astype(float).values
        
        # 데이터 결합
        raw_data = np.stack([sensor_x, sensor_y, sensor_z], axis=1)
        print(f"📊 원시 데이터 크기: {raw_data.shape}")
        
        # 🛡️ 진동 특성 분석 (오경보 저감용)
        characteristics = analyze_vibration_characteristics(raw_data)
        
        # 트리거 포인트 탐지
        trigger_idx = find_trigger_point(raw_data)
        
        if trigger_idx is None:
            print("⚠️ 트리거 포인트를 찾을 수 없음 - 데이터 중앙 사용")
            trigger_idx = len(raw_data) // 2
        else:
            print(f"🎯 트리거 포인트 발견: {trigger_idx}번째 샘플")
        
        # 40초 슬라이싱 (학습과 동일)
        PRE_SAMPLES = 1500   # 전 15초
        POST_SAMPLES = 2500  # 후 25초
        TOTAL_SAMPLES = 4000 # 총 40초
        
        start_idx = max(0, trigger_idx - PRE_SAMPLES)
        end_idx = trigger_idx + POST_SAMPLES
        
        print(f"📏 슬라이싱 구간: {start_idx} ~ {end_idx}")
        
        # 데이터 추출 및 패딩
        if end_idx <= len(raw_data):
            sliced_data = raw_data[start_idx:end_idx, :3]
        else:
            available_data = raw_data[start_idx:, :3]
            pad_length = TOTAL_SAMPLES - len(available_data)
            padding = np.zeros((pad_length, 3))
            sliced_data = np.vstack([available_data, padding])
            print(f"⚠️ 데이터 부족으로 {pad_length}개 샘플 패딩 적용")
        
        if len(sliced_data) > TOTAL_SAMPLES:
            sliced_data = sliced_data[:TOTAL_SAMPLES]
        
        if len(sliced_data) != TOTAL_SAMPLES:
            pad_length = TOTAL_SAMPLES - len(sliced_data)
            padding = np.zeros((pad_length, 3))
            sliced_data = np.vstack([sliced_data, padding])
        
        print(f"✅ 슬라이싱 완료: {sliced_data.shape}")
        
        # ConvLSTM 입력 형태로 변환
        reshaped = sliced_data.reshape(40, 100, 3)      # 40프레임, 각 100샘플, 3축
        reshaped = np.transpose(reshaped, (0, 2, 1))    # (40, 3, 100)
        reshaped = np.expand_dims(reshaped, axis=-1)    # (40, 3, 100, 1)
        reshaped = np.expand_dims(reshaped, axis=0)     # (1, 40, 3, 100, 1)
        
        print(f"🔄 형태 변환 완료: {reshaped.shape}")
        
        # z-score 정규화
        mean = reshaped.mean()
        std = reshaped.std()
        
        if std > 0:
            normalized = (reshaped - mean) / std
            print(f"📐 정규화 완료: 평균={mean:.6f}, 표준편차={std:.6f}")
        else:
            normalized = reshaped
            print("⚠️ 표준편차가 0이므로 정규화 생략")
        
        preprocess_info = {
            'original_length': len(raw_data),
            'trigger_point': trigger_idx,
            'slicing_range': (start_idx, end_idx),
            'final_shape': normalized.shape,
            'normalization': {'mean': float(mean), 'std': float(std)},
            'used_sensor': sensor_cols[0].split('_')[1],  # sensor_1 -> 1
            'characteristics': characteristics  # 🛡️ 진동 특성 추가
        }
        
        return normalized, df, preprocess_info
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, str(e)

def predict_and_save_result(raw_csv_path, processed_csv_path):
    """AI 예측 수행 및 결과 저장 (오경보 저감 로직 적용)"""
    if convlstm_model is None:
        print("⚠️ 모델이 로드되지 않아 예측을 건너뜁니다")
        return None
    
    try:
        print(f"\n🔄 === AI 예측 시작: {os.path.basename(raw_csv_path)} ===")
        
        # 전처리 (진동 특성 분석 포함)
        X, df, preprocess_info = preprocess_for_convlstm(raw_csv_path)
        
        if X is None:
            print(f"❌ 전처리 실패: {preprocess_info}")
            return None
        
        characteristics = preprocess_info.get('characteristics')
        
        # 🛡️ 진동 특성 출력
        if characteristics and FALSE_POSITIVE_LOG:
            print(f"🔍 진동 특성 분석:")
            print(f"   - 지속성: {characteristics['duration_ratio']:.3f}")
            print(f"   - 주요 주파수: {characteristics['dominant_frequency']:.1f}Hz")
            print(f"   - 에너지 균형: {characteristics['energy_balance']:.3f}")
            print(f"   - 최대 변화율: {characteristics['max_change_rate']:.3f}")
            print(f"   - 최대 진폭: {characteristics['max_amplitude']:.3f}")
            print(f"   - 저주파 우세성: {characteristics['low_freq_dominance']:.3f}")
        
        # 모델 예측
        start_time = time.time()
        predictions = convlstm_model.predict(X, verbose=0)
        inference_time = time.time() - start_time
        
        # 🛡️ 향상된 지진 감지 로직 적용
        final_class, final_confidence, is_suppressed, detection_reason = enhanced_earthquake_detection_logic(
            predictions, characteristics
        )
        
        # 예측 결과 출력
        print(f"\n🧠 === 모델 예측 결과 ===")
        for i, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
            prob = predictions[0][i]
            icon = CLASS_COLORS[class_id]
            print(f"   {icon} {class_name}: {prob:.6f} ({prob*100:.2f}%)")
        
        original_class = int(np.argmax(predictions[0]))
        original_confidence = float(predictions[0][original_class])
        
        print(f"\n🎯 === 최종 분류 결과 ===")
        print(f"원본 예측: {CLASS_NAMES[original_class]} (신뢰도: {original_confidence:.4f})")
        print(f"최종 분류: {CLASS_NAMES[final_class]} (신뢰도: {final_confidence:.4f})")
        print(f"검증 결과: {detection_reason}")
        
        # 경보 판별
        if final_class == 0:  # 지진
            print(f"\n🚨 === 지진 감지! 경보 발령! ===")
            alert_status = "EARTHQUAKE_ALERT"
        else:
            if is_suppressed:
                print(f"\n🛡️ === 오경보 억제됨 - 경보 차단 ===")
                alert_status = "FALSE_POSITIVE_SUPPRESSED"
            else:
                print(f"\n✅ === 비지진 진동 감지 - 정상 상태 ===")
                alert_status = "NO_ALERT"
        
        # 예측 결과가 포함된 CSV 생성
        if df is not None:
            # 예측 결과 추가
            df['predicted_class'] = final_class
            df['predicted_class_name'] = CLASS_NAMES[final_class]
            df['confidence'] = final_confidence
            df['original_predicted_class'] = original_class  # 🛡️ 원본 예측 보존
            df['original_confidence'] = original_confidence  # 🛡️ 원본 신뢰도 보존
            df['is_false_positive_suppressed'] = is_suppressed  # 🛡️ 억제 여부
            df['inference_time'] = inference_time
            df['alert_status'] = alert_status
            
            # 전처리 정보 추가
            df['trigger_point'] = preprocess_info['trigger_point']
            df['original_length'] = preprocess_info['original_length']
            df['used_sensor'] = preprocess_info['used_sensor']
            df['normalization_mean'] = preprocess_info['normalization']['mean']
            df['normalization_std'] = preprocess_info['normalization']['std']
            
            # 🛡️ 진동 특성 정보 추가
            if characteristics:
                for key, value in characteristics.items():
                    df[f'vibration_{key}'] = value
            
            # 모든 클래스 확률 추가
            for i, name in CLASS_NAMES.items():
                df[f'prob_{name}'] = predictions[0][i]
            
            # 🛡️ 필터링 통과 여부 정보 추가
            df['earthquake_confidence_threshold'] = EARTHQUAKE_CONFIDENCE_THRESHOLD
            df['multi_criteria_filter_enabled'] = ENABLE_MULTI_CRITERIA_FILTER
            df['advanced_filtering_enabled'] = ENABLE_ADVANCED_FILTERING
            
            # 처리된 결과 저장
            df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
            print(f"📁 AI 처리 결과 저장: {os.path.basename(processed_csv_path)}")
        
        return {
            'predicted_class': final_class,
            'class_name': CLASS_NAMES[final_class],
            'confidence': final_confidence,
            'original_class': original_class,  # 🛡️ 원본 정보 보존
            'original_confidence': original_confidence,
            'is_false_positive_suppressed': is_suppressed,
            'alert_status': alert_status,
            'preprocess_info': preprocess_info,
            'all_probabilities': predictions[0].tolist(),
            'filter_applied': 'enhanced_multi_criteria'  # 🛡️ 적용된 필터 정보
        }
        
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# 🛡️ 파일 중복 처리 방지 함수
def is_already_processed(raw_filename):
    """이미 처리된 파일인지 확인"""
    ai_filename = raw_filename.replace("event_", "ai_")
    processed_path = f"{PROCESSED_DATA_DIR}/{ai_filename}"
    return os.path.exists(processed_path)

# 🛡️ 시스템 건강성 체크 함수
def system_health_check():
    """시스템 건강성 체크"""
    health_status = {
        'tensorflow_available': tf_available,
        'model_loaded': convlstm_model is not None,
        'directories_ready': all([
            os.path.exists(RAW_DATA_DIR),
            os.path.exists(PROCESSED_DATA_DIR)
        ]),
        'influxdb_connected': False
    }
    
    # InfluxDB 연결 테스트
    try:
        test_query = f'from(bucket: "{INFLUX_BUCKET}") |> range(start: -1m) |> limit(n:1)'
        result = query_api.query(org=INFLUX_ORG, query=test_query)
        health_status['influxdb_connected'] = True
    except:
        health_status['influxdb_connected'] = False
    
    return health_status

# ✅ 통계 추적 (오경보 저감 통계 추가)
detection_stats = {
    'total_events': 0,
    'class_counts': {name: 0 for name in CLASS_NAMES.values()},
    'alert_count': 0,
    'suppressed_count': 0,
    'false_positive_suppressed': 0,  # 🛡️ 오경보 억제 횟수
    'processing_times': [],  # 🆕 처리 시간 추적
    'filter_effectiveness': {  # 🆕 필터별 효과 추적
        'confidence_filter': 0,
        'duration_filter': 0,
        'frequency_filter': 0,
        'energy_filter': 0,
        'change_rate_filter': 0,
        'confidence_gap_filter': 0,
        'low_freq_filter': 0
    },
    'start_time': datetime.now()
}

# 시스템 시작 정보 출력
print("\n🚀 === 향상된 3센서 지원 실시간 지진 감지 시스템 시작! ===")
print(f"📊 AI 분류 클래스: {list(CLASS_NAMES.values())}")
print(f"🔍 감시 대상 포트: {PORTS}")
print(f"⏱️ 체크 주기: {CHECK_INTERVAL}초")
print(f"📡 지원 센서: sensor_1, sensor_2, sensor_3")
print(f"🌐 Node-RED 서버: {NODERED_BASE_URL}")
print(f"💾 원시 데이터 저장: {RAW_DATA_DIR}")
print(f"🧠 AI 처리 결과: {PROCESSED_DATA_DIR}")
print(f"🛡️ 오경보 저감 설정:")
print(f"   - 신뢰도 임계값: {EARTHQUAKE_CONFIDENCE_THRESHOLD*100:.0f}%")
print(f"   - 다중 조건 필터: {'활성화' if ENABLE_MULTI_CRITERIA_FILTER else '비활성화'}")
print(f"   - 고급 필터링: {'활성화' if ENABLE_ADVANCED_FILTERING else '비활성화'}")

# 🛡️ 시스템 건강성 체크
health = system_health_check()
print(f"\n🏥 시스템 건강성 체크:")
print(f"   - TensorFlow: {'✅' if health['tensorflow_available'] else '❌'}")
print(f"   - 모델 로딩: {'✅' if health['model_loaded'] else '❌'}")
print(f"   - 디렉토리: {'✅' if health['directories_ready'] else '❌'}")
print(f"   - InfluxDB: {'✅' if health['influxdb_connected'] else '❌'}")

print("="*70)

# ✅ 실시간 감시 루프
try:
    consecutive_no_data = 0  # 연속으로 데이터가 없는 횟수
    
    while True:
        now = datetime.utcnow()
        start = now - timedelta(seconds=CHECK_INTERVAL)
        data_found = False

        for port in PORTS:
            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: {start.isoformat()}Z, stop: {now.isoformat()}Z)
              |> filter(fn: (r) => r._field == "intensity" and r._measurement == "{port}")
              |> sort(columns: ["_time"], desc: true)
              |> limit(n:1)
            '''

            result = query_api.query(org=INFLUX_ORG, query=query)

            for table in result:
                for record in table.records:
                    data_found = True
                    intensity = record.get_value()
                    if not isinstance(intensity, (int, float)):
                        continue

                    if intensity >= 3.0:
                        event_time = record.get_time().astimezone()
                        kst_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        print(f"\n🔥 === 진도 3 이상 감지 ===")
                        print(f"📅 시간: {kst_time}")
                        print(f"🌐 포트: {port}")
                        print(f"📊 진도: {intensity:.2f}")
                        
                        # 🛡️ 중복 처리 방지 체크
                        estimated_filename = f"event_{port}_{kst_time.replace(':', '-').replace(' ', '_')}.csv"
                        if is_already_processed(estimated_filename):
                            print(f"⚠️ 이미 처리된 이벤트 건너뜀: {estimated_filename}")
                            continue
                        
                        print("⏳ 40초 데이터 수집 대기 중...")
                        
                        # 진행 상황 표시
                        for i in range(40, 0, -5):
                            print(f"   🕐 {i}초 남음...")
                            time.sleep(5)

                        encoded_time = quote(kst_time)
                        url = f"{NODERED_BASE_URL}/{encoded_time}/{port}"
                        print(f"🔗 Node-RED 호출: {url}")

                        try:
                            res = requests.get(url, timeout=30)
                            if res.status_code == 200:
                                data = res.json()
                                
                                # 🔧 Node-RED 응답에서 파일명 추출
                                filename = None
                                
                                # 방법 1: 응답에서 파일명 추출 시도
                                if isinstance(data, dict):
                                    if "message" in data:
                                        message = str(data["message"])
                                        print(f"🔍 Node-RED 메시지: {message}")
                                        
                                        if "generated" in message and ".csv" in message:
                                            import re
                                            csv_match = re.search(r'event_\d+_[\d\-_]+\.csv', message)
                                            if csv_match:
                                                filename = csv_match.group()
                                                print(f"📁 추출된 파일명: {filename}")
                                    
                                    elif "filename" in data:
                                        filename = data["filename"]
                                        print(f"📁 직접 파일명: {filename}")
                                
                                # 방법 2: 예상 파일명으로 대체
                                if not filename:
                                    formatted_time = kst_time.replace(':', '-').replace(' ', '_')
                                    filename = f"event_{port}_{formatted_time}.csv"
                                    print(f"🎯 예상 파일명 사용: {filename}")
                                
                                csv_path = f"{RAW_DATA_DIR}/{filename}"
                                
                                # 파일 존재 확인 (최대 15초 대기)
                                max_wait = 15
                                wait_count = 0
                                while wait_count < max_wait and not os.path.exists(csv_path):
                                    print(f"⏳ 파일 생성 대기 중... ({wait_count + 1}/{max_wait}초)")
                                    time.sleep(1)
                                    wait_count += 1
                                
                                if os.path.exists(csv_path):
                                    file_size = os.path.getsize(csv_path)
                                    print(f"✅ 파일 발견! 크기: {file_size} bytes")
                                    
                                    # 중복 처리 방지 체크
                                    ai_filename = filename.replace("event_", "ai_")
                                    processed_csv_path = f"{PROCESSED_DATA_DIR}/{ai_filename}"
                                    
                                    if os.path.exists(processed_csv_path):
                                        print(f"⚠️ 이미 처리된 파일 건너뜀: {ai_filename}")
                                        continue
                                    
                                    # 🛡️ AI 예측 수행 (오경보 저감 로직 적용)
                                    processing_start = time.time()
                                    ai_result = predict_and_save_result(csv_path, processed_csv_path)
                                    processing_time = time.time() - processing_start
                                    
                                    if ai_result:
                                        # 🛡️ 오경보 저감 통계 업데이트
                                        detection_stats['total_events'] += 1
                                        detection_stats['class_counts'][ai_result['class_name']] += 1
                                        detection_stats['processing_times'].append(processing_time)
                                        
                                        if ai_result['alert_status'] == "EARTHQUAKE_ALERT":
                                            detection_stats['alert_count'] += 1
                                        elif ai_result['alert_status'] == "FALSE_POSITIVE_SUPPRESSED":
                                            detection_stats['suppressed_count'] += 1
                                            detection_stats['false_positive_suppressed'] += 1
                                        else:
                                            detection_stats['suppressed_count'] += 1
                                        
                                        # 🛡️ 향상된 통계 출력
                                        print(f"\n📈 === 오경보 저감 성과 리포트 ===")
                                        print(f"📊 총 이벤트: {detection_stats['total_events']}건")
                                        print(f"🚨 지진 경보: {detection_stats['alert_count']}건")
                                        print(f"🛡️ 오경보 억제: {detection_stats['false_positive_suppressed']}건")
                                        print(f"✅ 정상 억제: {detection_stats['suppressed_count'] - detection_stats['false_positive_suppressed']}건")
                                        print(f"⚡ 처리 시간: {processing_time:.2f}초")
                                        
                                        if detection_stats['total_events'] > 0:
                                            alert_rate = detection_stats['alert_count'] / detection_stats['total_events'] * 100
                                            false_positive_suppression_rate = detection_stats['false_positive_suppressed'] / detection_stats['total_events'] * 100
                                            total_suppression_rate = detection_stats['suppressed_count'] / detection_stats['total_events'] * 100
                                            avg_processing_time = np.mean(detection_stats['processing_times'])
                                            
                                            print(f"📈 지진 경보율: {alert_rate:.1f}%")
                                            print(f"🛡️ 오경보 억제율: {false_positive_suppression_rate:.1f}%")
                                            print(f"📊 총 억제율: {total_suppression_rate:.1f}%")
                                            print(f"⚡ 평균 처리 시간: {avg_processing_time:.2f}초")
                                            
                                            # 🆕 오경보 저감 효과 계산
                                            if detection_stats['false_positive_suppressed'] > 0:
                                                potential_false_alarms = detection_stats['alert_count'] + detection_stats['false_positive_suppressed']
                                                reduction_effectiveness = (detection_stats['false_positive_suppressed'] / potential_false_alarms) * 100
                                                print(f"🎯 오경보 저감 효과: {reduction_effectiveness:.1f}%")
                                        
                                        print(f"\n📊 클래스별 분포:")
                                        for class_name, count in detection_stats['class_counts'].items():
                                            if count > 0:
                                                icon = CLASS_COLORS[list(CLASS_NAMES.values()).index(class_name)]
                                                percentage = count / detection_stats['total_events'] * 100
                                                print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")
                                                
                                        runtime = datetime.now() - detection_stats['start_time']
                                        print(f"🕒 총 실행 시간: {runtime}")
                                        print("="*50)
                                    else:
                                        print(f"❌ AI 예측 실패")
                                else:
                                    print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
                                    
                                    # 최신 파일 찾기 시도
                                    try:
                                        csv_files = [f for f in os.listdir(RAW_DATA_DIR) 
                                                   if f.startswith(f"event_{port}_") and f.endswith('.csv')]
                                        if csv_files:
                                            latest_file = max(csv_files, 
                                                            key=lambda x: os.path.getmtime(os.path.join(RAW_DATA_DIR, x)))
                                            print(f"🔍 최신 파일 시도: {latest_file}")
                                            csv_path = f"{RAW_DATA_DIR}/{latest_file}"
                                            
                                            if os.path.exists(csv_path):
                                                print(f"✅ 최신 파일 사용: {latest_file}")
                                                # AI 처리 재시도...
                                    except Exception as e:
                                        print(f"❌ 최신 파일 찾기 실패: {e}")
                            else:
                                print(f"❌ Node-RED 응답 오류: {res.status_code}")
                                if hasattr(res, 'text'):
                                    print(f"   응답: {res.text[:200]}")
                                    
                        except requests.exceptions.Timeout:
                            print(f"❌ Node-RED 요청 타임아웃 (30초 초과)")
                        except Exception as e:
                            print(f"❌ Node-RED 요청 실패: {e}")
                            import traceback
                            traceback.print_exc()
        
        # 🛡️ 연속 무데이터 모니터링
        if not data_found:
            consecutive_no_data += 1
            if consecutive_no_data % 300 == 0:  # 5분마다 알림
                print(f"⏰ 데이터 없음 지속 중... ({consecutive_no_data}초)")
        else:
            consecutive_no_data = 0

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n\n🛑 === 사용자에 의해 중단됨 ===")
    print("📊 최종 오경보 저감 성과 리포트:")
    print(f"   총 이벤트: {detection_stats['total_events']}건")
    print(f"   지진 경보: {detection_stats['alert_count']}건") 
    print(f"   오경보 억제: {detection_stats['false_positive_suppressed']}건")
    print(f"   정상 억제: {detection_stats['suppressed_count'] - detection_stats['false_positive_suppressed']}건")
    
    if detection_stats['total_events'] > 0:
        alert_rate = detection_stats['alert_count'] / detection_stats['total_events'] * 100
        false_positive_suppression_rate = detection_stats['false_positive_suppressed'] / detection_stats['total_events'] * 100
        total_suppression_rate = detection_stats['suppressed_count'] / detection_stats['total_events'] * 100
        avg_processing_time = np.mean(detection_stats['processing_times']) if detection_stats['processing_times'] else 0
        
        print(f"   지진 경보율: {alert_rate:.1f}%")
        print(f"   오경보 억제율: {false_positive_suppression_rate:.1f}%")
        print(f"   총 억제율: {total_suppression_rate:.1f}%")
        print(f"   평균 처리 시간: {avg_processing_time:.2f}초")
        
        # 🛡️ 최종 오경보 저감 효과 분석
        if detection_stats['false_positive_suppressed'] > 0:
            potential_false_alarms = detection_stats['alert_count'] + detection_stats['false_positive_suppressed']
            reduction_effectiveness = (detection_stats['false_positive_suppressed'] / potential_false_alarms) * 100
            original_false_alarm_rate = (potential_false_alarms / detection_stats['total_events']) * 100
            final_false_alarm_rate = alert_rate
            improvement = original_false_alarm_rate - final_false_alarm_rate
            
            print(f"\n🎯 === 오경보 저감 효과 분석 ===")
            print(f"   원래 예상 오경보율: {original_false_alarm_rate:.1f}%")
            print(f"   최종 실제 경보율: {final_false_alarm_rate:.1f}%")
            print(f"   개선 효과: {improvement:.1f}%p 감소")
            print(f"   저감 효과: {reduction_effectiveness:.1f}%")
    
    print(f"\n📈 클래스별 최종 분포:")
    for class_name, count in detection_stats['class_counts'].items():
        if count > 0:
            icon = CLASS_COLORS[list(CLASS_NAMES.values()).index(class_name)]
            percentage = count / detection_stats['total_events'] * 100 if detection_stats['total_events'] > 0 else 0
            print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")
    
    total_runtime = datetime.now() - detection_stats['start_time']
    print(f"\n🕒 총 실행 시간: {total_runtime}")
    print(f"💾 저장 위치:")
    print(f"   원시 데이터: {RAW_DATA_DIR}")
    print(f"   AI 처리 결과: {PROCESSED_DATA_DIR}")
    print("\n🎉 향상된 오경보 저감 지진 감지 시스템이 성공적으로 종료되었습니다!")
    print("👋 시스템 종료")

except Exception as e:
    print(f"❌ 예상치 못한 오류: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n💾 오류 발생 시점 통계:")
    print(f"   처리된 이벤트: {detection_stats['total_events']}건")
    print(f"   실행 시간: {datetime.now() - detection_stats['start_time']}")
    print("🔧 시스템 재시작을 권장합니다.")
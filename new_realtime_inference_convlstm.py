import time
from datetime import datetime, timedelta
import requests
from influxdb_client import InfluxDBClient
from urllib.parse import quote
import os
import csv
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# ✅ 4클래스 ConvLSTM 모델 경로
MODEL_PATH = 'C:/earthquake_project/convlstm_4class_model.h5'

# ✅ 실시간 CSV 저장 위치
OUTPUT_DIR = "C:/earthquake_modeling/earthquake_project_v3/influxLogs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 4클래스 ConvLSTM 모델 불러오기
print("🔄 4클래스 ConvLSTM 모델 로딩 중...")
convlstm_4class_model = load_model(MODEL_PATH)
print("✅ 모델 로딩 완료!")

# 🔍 모델 정보 확인
print(f"🔍 모델 입력 shape: {convlstm_4class_model.input_shape}")
print(f"🔍 모델 출력 shape: {convlstm_4class_model.output_shape}")

# ✅ 클래스 정의
CLASS_NAMES = {
    0: '지진',
    1: '산업진동', 
    2: '불규칙생활',
    3: '모터진동'
}

CLASS_COLORS = {
    0: '🔴',  # 지진 - 빨강
    1: '🔵',  # 산업진동 - 파랑  
    2: '🟢',  # 불규칙생활 - 초록
    3: '🟠'   # 모터진동 - 주황
}

# ✅ 개선된 전처리 함수
def improved_preprocessing(csv_path):
    """
    개선된 전처리 함수 - 반복 패딩 및 축별 정규화 적용
    """
    try:
        df = pd.read_csv(csv_path)
        
        # sensor1_x, y, z 컬럼 확인
        required_cols = ['sensor1_x', 'sensor1_y', 'sensor1_z']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ 필수 컬럼 없음: {required_cols}")
            return None, None
            
        # 3축 데이터 추출
        sensor_data = df[required_cols].values.astype(float)
        original_length = len(sensor_data)
        
        # 4000 샘플로 맞추기 (40초 × 100Hz)
        target_samples = 4000
        
        if len(sensor_data) < target_samples:
            # 🔧 개선: 단순 0 패딩 대신 반복 패딩
            if len(sensor_data) >= 100:  # 최소 1초 이상 데이터가 있을 때만
                repeat_count = target_samples // len(sensor_data) + 1
                sensor_data = np.tile(sensor_data, (repeat_count, 1))[:target_samples]
                print(f"🔄 반복 패딩 적용: {original_length} → {target_samples}")
            else:
                # 너무 적은 데이터는 0 패딩
                pad_width = ((0, target_samples - len(sensor_data)), (0, 0))
                sensor_data = np.pad(sensor_data, pad_width, mode='constant')
                print(f"⚠️ 제로 패딩 적용: {original_length} → {target_samples} (데이터 부족)")
        elif len(sensor_data) > target_samples:
            # 넘치면 자르기
            sensor_data = sensor_data[:target_samples]
            
        # (4000, 3) → (40, 3, 100, 1) 변환
        reshaped = sensor_data.reshape(40, 100, 3)  # 40프레임, 각 100샘플, 3축
        reshaped = np.transpose(reshaped, (0, 2, 1))  # (40, 3, 100)
        reshaped = np.expand_dims(reshaped, axis=-1)  # (40, 3, 100, 1)
        reshaped = np.expand_dims(reshaped, axis=0)   # (1, 40, 3, 100, 1)
        
        # 🔧 개선: 축별 정규화 (더 안정적)
        for axis in range(3):
            axis_data = reshaped[0, :, axis, :, 0].flatten()
            if axis_data.std() > 1e-6:  # 표준편차가 0이 아닐 때만
                mean_val = axis_data.mean()
                std_val = axis_data.std()
                reshaped[0, :, axis, :, 0] = (reshaped[0, :, axis, :, 0] - mean_val) / std_val
            
        return reshaped, df, original_length
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return None, None, 0

# ✅ 데이터 특성 분석 함수
def analyze_data_features(df):
    """실시간 데이터 특성 분석"""
    x_data = df['sensor1_x'].values
    y_data = df['sensor1_y'].values
    z_data = df['sensor1_z'].values
    
    # 1. 샘플 수
    sample_count = len(df)
    
    # 2. 에너지 분포 계산
    x_energy = np.sum(x_data**2)
    y_energy = np.sum(y_data**2)
    z_energy = np.sum(z_data**2)
    total_energy = x_energy + y_energy + z_energy
    
    z_ratio = z_energy / total_energy if total_energy > 0 else 0
    
    # 3. 주파수 분석
    combined = np.concatenate([x_data, y_data, z_data])
    if len(combined) > 10:
        fft_result = np.fft.fft(combined)
        freqs = np.fft.fftfreq(len(combined), 1/100)
        
        # 0-50Hz 범위에서 주요 주파수 찾기
        valid_range = (freqs >= 0) & (freqs <= 50)
        if np.any(valid_range):
            dominant_freq_idx = np.argmax(np.abs(fft_result[valid_range]))
            dominant_freq = freqs[valid_range][dominant_freq_idx]
        else:
            dominant_freq = 0
    else:
        dominant_freq = 0
    
    # 4. 진도 분석
    if 'sensor1_magnitude' in df.columns:
        magnitude = df['sensor1_magnitude'].values
        max_magnitude = np.max(magnitude)
        avg_magnitude = np.mean(magnitude)
    else:
        max_magnitude = 0
        avg_magnitude = 0
    
    # 5. 규칙성 분석 (간단한 버전)
    amplitude_ranges = [np.ptp(x_data), np.ptp(y_data), np.ptp(z_data)]
    regularity_score = 1.0 / (1.0 + np.std(amplitude_ranges))  # 범위가 비슷할수록 규칙적
    
    return {
        'sample_count': sample_count,
        'z_energy_ratio': z_ratio,
        'dominant_freq': abs(dominant_freq),
        'max_magnitude': max_magnitude,
        'avg_magnitude': avg_magnitude,
        'regularity_score': regularity_score,
        'data_sufficient': sample_count >= 1000  # 10초 이상
    }

# ✅ 규칙 기반 보정 시스템
def apply_correction_rules(predictions, features, original_length):
    """
    규칙 기반 예측 보정
    """
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    corrections_applied = []
    
    # 수정된 예측값 복사
    modified_predictions = predictions[0].copy()
    
    # 규칙 1: 데이터 부족 시 지진 예측 억제
    if original_length < 500:  # 5초 미만
        if predicted_class == 0:  # 지진으로 예측했다면
            modified_predictions[0] = 0.01  # 지진 확률 거의 0으로
            corrections_applied.append(f"데이터 부족({original_length}개) - 지진 억제")
    
    # 규칙 2: 매우 부족한 데이터에서는 지진 완전 배제
    elif original_length < 200:  # 2초 미만
        modified_predictions[0] = 0.0
        corrections_applied.append(f"심각한 데이터 부족({original_length}개) - 지진 완전 배제")
    
    # 규칙 3: 모터 주파수 대역 감지 (10-30Hz)
    if 10 <= features['dominant_freq'] <= 30:
        if predicted_class == 0:  # 지진으로 예측했지만 모터 주파수
            modified_predictions[3] = max(modified_predictions[3], 0.8)  # 모터진동 확률 높임
            modified_predictions[0] = min(modified_predictions[0], 0.3)  # 지진 확률 낮춤
            corrections_applied.append(f"모터 주파수({features['dominant_freq']:.1f}Hz) - 모터진동 우선")
    
    # 규칙 4: Z축 에너지 집중 + 데이터 부족 → 산업진동 가능성
    if features['z_energy_ratio'] > 0.8 and original_length < 1000:
        if predicted_class == 0 and confidence < 0.9:
            modified_predictions[1] = max(modified_predictions[1], 0.7)  # 산업진동 확률 높임
            modified_predictions[0] = min(modified_predictions[0], 0.4)  # 지진 확률 낮춤
            corrections_applied.append(f"Z축 집중({features['z_energy_ratio']:.2f}) + 데이터 부족 - 산업진동 가능성")
    
    # 규칙 5: 지진 신뢰도 임계값 (매우 높은 기준)
    if predicted_class == 0 and confidence < 0.95:  # 지진은 95% 이상만 인정
        modified_predictions[0] = 0.0
        corrections_applied.append(f"지진 신뢰도 부족({confidence:.3f} < 0.95) - 지진 배제")
    
    # 최종 예측 클래스 결정
    final_predicted_class = int(np.argmax(modified_predictions))
    final_confidence = float(modified_predictions[final_predicted_class])
    
    return final_predicted_class, final_confidence, corrections_applied

# ✅ 개선된 예측 및 저장 함수
def predict_and_save_result_improved(filepath):
    """
    개선된 4클래스 예측 수행 및 결과 저장
    """
    try:
        print(f"\n🔍 파일 분석 시작: {os.path.basename(filepath)}")
        
        # 1. 개선된 전처리
        X, df, original_length = improved_preprocessing(filepath)
        if X is None:
            return None
            
        # 2. 데이터 특성 분석
        features = analyze_data_features(df)
        
        # 3. 원본 모델 예측
        start_time = time.time()
        predictions = convlstm_4class_model.predict(X, verbose=0)
        inference_time = time.time() - start_time
        
        raw_predicted_class = int(np.argmax(predictions[0]))
        raw_confidence = float(predictions[0][raw_predicted_class])
        
        # 4. 규칙 기반 보정 적용
        corrected_class, corrected_confidence, corrections = apply_correction_rules(
            predictions, features, original_length
        )
        
        # 5. 결과 출력
        print(f"\n🧠 === 개선된 예측 결과 ===")
        print(f"📊 원본 예측: {CLASS_NAMES[raw_predicted_class]} ({raw_confidence:.4f})")
        
        if corrections:
            print(f"🔧 적용된 보정 규칙:")
            for correction in corrections:
                print(f"   - {correction}")
        
        final_class_name = CLASS_NAMES[corrected_class]
        final_icon = CLASS_COLORS[corrected_class]
        
        print(f"{final_icon} 최종 예측: {final_class_name} ({corrected_confidence:.4f})")
        print(f"⚡ 추론 시간: {inference_time:.3f}초")
        
        # 6. 상세 특성 정보
        print(f"\n🔍 데이터 특성 분석:")
        print(f"   📏 원본 샘플 수: {original_length}개 → 4000개로 변환")
        print(f"   ⚖️ Z축 에너지 비율: {features['z_energy_ratio']:.3f}")
        print(f"   🎵 주요 주파수: {features['dominant_freq']:.1f} Hz")
        print(f"   📊 최대 진도: {features['max_magnitude']:.1f}")
        print(f"   📈 데이터 충분성: {'✅' if features['data_sufficient'] else '❌'}")
        
        # 7. 모든 클래스 확률 출력
        print(f"\n📊 전체 클래스별 확률:")
        for i, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
            prob = predictions[0][i]
            icon = CLASS_COLORS[class_id]
            print(f"   {icon} {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # 8. CSV에 예측 결과 추가
        if df is not None:
            df['predicted_class'] = corrected_class
            df['predicted_class_name'] = final_class_name
            df['confidence'] = corrected_confidence
            df['raw_prediction'] = raw_predicted_class
            df['raw_confidence'] = raw_confidence
            df['inference_time'] = inference_time
            df['original_sample_count'] = original_length
            df['corrections_applied'] = '; '.join(corrections) if corrections else 'None'
            
            # 모든 클래스 확률도 추가
            for i, name in CLASS_NAMES.items():
                df[f'prob_{name}'] = predictions[0][i]
                
            # 특성 정보도 추가
            for key, value in features.items():
                df[f'feature_{key}'] = value
                
            # 저장
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"📁 상세 결과 포함 저장 완료: {os.path.basename(filepath)}")
            
        return {
            'predicted_class': corrected_class,
            'class_name': final_class_name,
            'confidence': corrected_confidence,
            'raw_prediction': raw_predicted_class,
            'raw_confidence': raw_confidence,
            'corrections_applied': corrections,
            'features': features,
            'inference_time': inference_time
        }
        
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

# ✅ InfluxDB 설정
INFLUX_TOKEN = "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg=="
INFLUX_ORG = "kds"
INFLUX_BUCKET = "Lasung_3sensor of Max"
PORTS = [6060, 7001, 7053, 7060, 7070, 8010, 8080]  # 감시 대상 포트
CHECK_INTERVAL = 1  # 초 단위 주기
COLLECTION_TIME = 60  # 🔧 개선: 40초 → 60초로 연장

client = InfluxDBClient(
    url="http://118.129.145.82:8086",
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)

query_api = client.query_api()

# ✅ 통계 추적용 변수
detection_stats = {
    'total_events': 0,
    'class_counts': {name: 0 for name in CLASS_NAMES.values()},
    'corrections_stats': {
        'total_corrections': 0,
        'correction_types': {}
    },
    'start_time': datetime.now()
}

print("\n🚀 개선된 4클래스 ConvLSTM 실시간 진동 분류 시작!")
print(f"📊 분류 가능한 클래스: {list(CLASS_NAMES.values())}")
print(f"🔍 감시 대상 포트: {PORTS}")
print(f"⏱️  체크 주기: {CHECK_INTERVAL}초")
print(f"📏 데이터 수집 시간: {COLLECTION_TIME}초 (개선됨)")
print(f"🛡️  보정 규칙: 데이터 부족, 주파수 기반, Z축 집중, 신뢰도 임계값")
print("="*70)

# ✅ 개선된 실시간 감시 루프
try:
    while True:
        now = datetime.utcnow()
        start = now - timedelta(seconds=CHECK_INTERVAL)

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
                    intensity = record.get_value()
                    event_time = record.get_time().astimezone()

                    if intensity >= 3.0:
                        kst_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n🔥 진도 3 이상 감지: {kst_time} (포트: {port}, 진도: {intensity:.2f})")
                        print(f"⏳ {COLLECTION_TIME}초 데이터 수집 대기 중... (개선된 수집 시간)")
                        time.sleep(COLLECTION_TIME)  # 🔧 개선: 60초로 연장

                        encoded_time = quote(kst_time)
                        url = f"http://192.168.68.57:1880/1min_event/{encoded_time}/{port}"

                        try:
                            res = requests.get(url)
                            print(f"🌐 API 호출 결과: {res.status_code}")

                            if res.status_code == 200:
                                # CSV 저장
                                file_time = kst_time.replace(":", "-").replace(" ", "_")
                                filename = f"event_{port}_{file_time}_improved.csv"
                                filepath = os.path.join(OUTPUT_DIR, filename)

                                try:
                                    data = res.json()
                                    for row in data:
                                        row["timestamp"] = "'" + row["timestamp"]

                                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                                        writer.writeheader()
                                        writer.writerows(data)

                                    print(f"📁 원본 파일 저장: {filename}")

                                    # 🔧 개선된 4클래스 예측 수행
                                    result = predict_and_save_result_improved(filepath)
                                    
                                    if result:
                                        # 통계 업데이트
                                        detection_stats['total_events'] += 1
                                        detection_stats['class_counts'][result['class_name']] += 1
                                        
                                        # 보정 통계 업데이트
                                        if result['corrections_applied']:
                                            detection_stats['corrections_stats']['total_corrections'] += 1
                                            for correction in result['corrections_applied']:
                                                correction_type = correction.split(' - ')[0]
                                                if correction_type not in detection_stats['corrections_stats']['correction_types']:
                                                    detection_stats['corrections_stats']['correction_types'][correction_type] = 0
                                                detection_stats['corrections_stats']['correction_types'][correction_type] += 1
                                        
                                        # 현재까지 통계 출력
                                        print(f"\n📈 현재까지 통계 (총 {detection_stats['total_events']}건):")
                                        for class_name, count in detection_stats['class_counts'].items():
                                            if count > 0:
                                                icon = CLASS_COLORS[list(CLASS_NAMES.values()).index(class_name)]
                                                percentage = count / detection_stats['total_events'] * 100
                                                print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")
                                        
                                        # 보정 통계
                                        total_corrections = detection_stats['corrections_stats']['total_corrections']
                                        if total_corrections > 0:
                                            print(f"\n🔧 보정 통계:")
                                            print(f"   전체 보정 적용: {total_corrections}/{detection_stats['total_events']}건 ({total_corrections/detection_stats['total_events']*100:.1f}%)")
                                            for correction_type, count in detection_stats['corrections_stats']['correction_types'].items():
                                                print(f"   - {correction_type}: {count}건")
                                        
                                        runtime = datetime.now() - detection_stats['start_time']
                                        print(f"🕒 실행 시간: {runtime}")

                                except Exception as e:
                                    print(f"❌ 파일 처리 실패: {e}")

                        except Exception as e:
                            print(f"❌ API 호출 실패: {e}")

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n\n🛑 사용자에 의해 중단됨")
    print("📊 최종 통계:")
    print(f"   총 이벤트: {detection_stats['total_events']}건")
    for class_name, count in detection_stats['class_counts'].items():
        if count > 0:
            icon = CLASS_COLORS[list(CLASS_NAMES.values()).index(class_name)]
            percentage = count / detection_stats['total_events'] * 100 if detection_stats['total_events'] > 0 else 0
            print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")
    
    total_corrections = detection_stats['corrections_stats']['total_corrections']
    if total_corrections > 0:
        print(f"\n🔧 최종 보정 통계:")
        print(f"   전체 보정 적용: {total_corrections}/{detection_stats['total_events']}건")
        for correction_type, count in detection_stats['corrections_stats']['correction_types'].items():
            print(f"   - {correction_type}: {count}건")
    
    print("👋 개선된 시스템 종료")

except Exception as e:
    print(f"❌ 예상치 못한 오류: {e}")
    import traceback
    traceback.print_exc()
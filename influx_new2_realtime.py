#!/usr/bin/env python3
"""
influx_base.py를 기반으로 한 NEW2 ConvLSTM 실시간 지진 분석 시스템
기존 influx_base.py의 간결한 구조 + NEW2 모델의 98.46% 정확도 활용
"""

import time
from datetime import datetime, timedelta
import requests
from influxdb_client import InfluxDBClient
from urllib.parse import quote
import os
import csv
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# 환경 설정
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TensorFlow 안전 로딩
def load_tensorflow():
    """TensorFlow 안전 로딩"""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import load_model
        print(f"✅ TensorFlow 로딩 완료: {tf.__version__}")
        return tf, load_model, True
    except Exception as e:
        print(f"❌ TensorFlow 로딩 실패: {e}")
        return None, None, False

# TensorFlow 로딩
tf, load_model, tf_available = load_tensorflow()

# =========================== 기본 설정 (influx_base.py 기반) ===========================

# InfluxDB 설정 (기존과 동일)
INFLUX_TOKEN = "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg=="
INFLUX_ORG = "kds"
INFLUX_BUCKET = "Lasung_3sensor of Max"
INFLUX_URL = "http://118.129.145.82:8086"
PORTS = [6060, 7001, 7053, 7060, 7070, 8010, 8080]
CHECK_INTERVAL = 1

# Node-RED 설정 (기존과 동일)
NODERED_BASE_URL = "http://118.129.145.82:8081/nodered/1min_event_lasung"

# 저장 디렉토리 설정
BASE_SAVE_DIR = "C:/earthquake_project/influxLogs"
RAW_SAVE_DIR = os.path.join(BASE_SAVE_DIR, "base")          # 원시 데이터 (기존과 동일)
AI_SAVE_DIR = os.path.join(BASE_SAVE_DIR, "new2_analysis")  # NEW2 AI 분석 결과
ALERT_SAVE_DIR = os.path.join(BASE_SAVE_DIR, "alerts")      # 지진 경보 로그

# 디렉토리 생성
for save_dir in [RAW_SAVE_DIR, AI_SAVE_DIR, ALERT_SAVE_DIR]:
    os.makedirs(save_dir, exist_ok=True)

# InfluxDB 클라이언트 초기화
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

# =========================== NEW2 모델 설정 ===========================

# NEW2 모델 설정
NEW2_CONFIG = {
    'model_paths': [
        'new2_convlstm_3class_best.h5',
        'new2_convlstm_3class_final.h5',
        'convlstm_3class_model.h5'  # 백업
    ],
    'classes': {0: '지진', 1: '규칙적산업진동', 2: '불규칙생활진동'},
    'accuracy': 0.9846,
    'earthquake_threshold': 0.90,  # 지진 신뢰도 임계값 90%
    'confidence_gap_min': 0.20     # 신뢰도 차이 최소 20%
}

# NEW2 모델 로딩
new2_model = None
if tf_available:
    print("🔄 NEW2 ConvLSTM 모델 로딩 중...")
    for model_path in NEW2_CONFIG['model_paths']:
        if os.path.exists(model_path):
            try:
                new2_model = load_model(model_path)
                print(f"✅ NEW2 모델 로딩 성공: {model_path}")
                print(f"🎯 모델 정확도: {NEW2_CONFIG['accuracy']*100:.2f}%")
                break
            except Exception as e:
                print(f"⚠️ {model_path} 로딩 실패: {e}")
                continue
    
    if new2_model is None:
        print("❌ 모든 NEW2 모델 로딩 실패 - 데이터 수집만 수행")
else:
    print("⚠️ TensorFlow 없음 - 데이터 수집만 수행")

# =========================== 데이터 처리 함수 (기존 + NEW2 확장) ===========================

def flatten_row(row):
    """기존 influx_base.py의 데이터 평면화 함수 (동일)"""
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

def preprocess_for_new2(csv_path):
    """NEW2 모델용 전처리 (간단 버전)"""
    try:
        df = pd.read_csv(csv_path)
        
        # 센서 선택 (sensor_1 우선)
        sensor_cols = None
        for i in range(1, 4):
            x_col, y_col, z_col = f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z'
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                sensor_cols = [x_col, y_col, z_col]
                break
        
        if not sensor_cols:
            return None, "센서 데이터 없음"
        
        # 3축 데이터 추출
        x_data = df[sensor_cols[0]].astype(float).values
        y_data = df[sensor_cols[1]].astype(float).values
        z_data = df[sensor_cols[2]].astype(float).values
        raw_data = np.stack([x_data, y_data, z_data], axis=1)
        
        # 트리거 포인트 찾기 (간단 버전)
        magnitude = np.sqrt(np.sum(raw_data**2, axis=1))
        trigger_idx = np.argmax(magnitude)  # 최대 진폭 지점
        
        # 40초 슬라이싱 (4000 샘플)
        PRE_SAMPLES = 1500   # 전 15초
        POST_SAMPLES = 2500  # 후 25초
        TOTAL_SAMPLES = 4000
        
        start_idx = max(0, trigger_idx - PRE_SAMPLES)
        end_idx = min(len(raw_data), trigger_idx + POST_SAMPLES)
        
        # 데이터 추출 및 패딩
        if end_idx - start_idx >= TOTAL_SAMPLES:
            sliced_data = raw_data[start_idx:start_idx + TOTAL_SAMPLES]
        else:
            available_data = raw_data[start_idx:end_idx]
            pad_length = TOTAL_SAMPLES - len(available_data)
            padding = np.zeros((pad_length, 3))
            sliced_data = np.vstack([available_data, padding])
        
        # NEW2 입력 형태로 변환: (1, 40, 3, 100, 1)
        reshaped = sliced_data.reshape(40, 100, 3)
        reshaped = np.transpose(reshaped, (0, 2, 1))  # (40, 3, 100)
        reshaped = np.expand_dims(reshaped, axis=-1)  # (40, 3, 100, 1)
        reshaped = np.expand_dims(reshaped, axis=0)   # (1, 40, 3, 100, 1)
        
        # z-score 정규화
        mean = reshaped.mean()
        std = reshaped.std()
        if std > 0:
            normalized = (reshaped - mean) / std
        else:
            normalized = reshaped
        
        return normalized, f"전처리 완료: {normalized.shape}"
        
    except Exception as e:
        return None, f"전처리 실패: {e}"

def analyze_with_new2(X):
    """NEW2 모델로 지진 분석"""
    if new2_model is None:
        return None, "모델 없음"
    
    try:
        # 예측 수행
        predictions = new2_model.predict(X, verbose=0)
        
        # 결과 분석
        earthquake_prob = predictions[0][0]    # 지진 확률
        industrial_prob = predictions[0][1]    # 규칙적산업진동 확률
        living_prob = predictions[0][2]        # 불규칙생활진동 확률
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # 지진 감지 로직 (간단 버전)
        is_earthquake = False
        alert_status = "NO_ALERT"
        
        if predicted_class == 0:  # 지진으로 예측됨
            # 신뢰도 검사
            if earthquake_prob >= NEW2_CONFIG['earthquake_threshold']:
                # 다른 클래스와 차이 검사
                max_other_prob = max(industrial_prob, living_prob)
                if earthquake_prob - max_other_prob >= NEW2_CONFIG['confidence_gap_min']:
                    is_earthquake = True
                    alert_status = "EARTHQUAKE_ALERT"
                else:
                    alert_status = "LOW_CONFIDENCE_GAP"
            else:
                alert_status = "LOW_CONFIDENCE"
        
        result = {
            'predicted_class': predicted_class,
            'class_name': NEW2_CONFIG['classes'][predicted_class],
            'confidence': confidence,
            'earthquake_prob': earthquake_prob,
            'industrial_prob': industrial_prob,
            'living_prob': living_prob,
            'is_earthquake': is_earthquake,
            'alert_status': alert_status
        }
        
        return result, "분석 완료"
        
    except Exception as e:
        return None, f"분석 실패: {e}"

def save_ai_result(raw_filepath, ai_result, event_info=None):
    """AI 분석 결과 저장"""
    try:
        # 원본 CSV 읽기
        df = pd.read_csv(raw_filepath)
        
        # AI 결과 추가
        for key, value in ai_result.items():
            df[f'ai_{key}'] = value
        
        df['ai_model'] = 'NEW2_ConvLSTM_3Class'
        df['ai_accuracy'] = NEW2_CONFIG['accuracy']
        df['analysis_time'] = datetime.now().isoformat()
        
        # event_info가 있으면 intensity 값 추가 (대시보드에서 사용)
        if event_info and 'intensity' in event_info:
            df['intensity'] = event_info['intensity']
            df['event_port'] = event_info['port']
            df['event_time'] = event_info['time']
        
        # AI 결과 파일 저장
        filename = os.path.basename(raw_filepath).replace('event_', 'new2_ai_')
        ai_filepath = os.path.join(AI_SAVE_DIR, filename)
        df.to_csv(ai_filepath, index=False, encoding='utf-8-sig')
        
        print(f"💾 AI 분석 결과 저장: {filename}")
        return ai_filepath
        
    except Exception as e:
        print(f"❌ AI 결과 저장 실패: {e}")
        return None

def save_earthquake_alert(ai_result, event_info):
    """지진 경보 정보 저장"""
    try:
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'event_time': event_info['time'],
            'port': event_info['port'],
            'intensity': event_info['intensity'],
            'ai_analysis': ai_result,
            'model_info': {
                'name': 'NEW2_ConvLSTM_3Class',
                'accuracy': NEW2_CONFIG['accuracy']
            }
        }
        
        alert_filename = f"earthquake_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        alert_filepath = os.path.join(ALERT_SAVE_DIR, alert_filename)
        
        with open(alert_filepath, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, ensure_ascii=False, indent=2)
        
        print(f"🚨 지진 경보 로그 저장: {alert_filename}")
        return alert_filepath
        
    except Exception as e:
        print(f"❌ 경보 로그 저장 실패: {e}")
        return None

# =========================== 통계 추적 ===========================

stats = {
    'total_events': 0,
    'earthquake_alerts': 0,
    'class_counts': {name: 0 for name in NEW2_CONFIG['classes'].values()},
    'start_time': datetime.now()
}

def update_stats(ai_result):
    """통계 업데이트"""
    if ai_result:
        stats['total_events'] += 1
        stats['class_counts'][ai_result['class_name']] += 1
        
        if ai_result['is_earthquake']:
            stats['earthquake_alerts'] += 1

def print_stats():
    """통계 출력"""
    if stats['total_events'] == 0:
        return
    
    runtime = datetime.now() - stats['start_time']
    alert_rate = stats['earthquake_alerts'] / stats['total_events'] * 100
    
    print(f"\\n📊 === NEW2 시스템 통계 ===")
    print(f"🕒 실행 시간: {runtime}")
    print(f"📈 총 이벤트: {stats['total_events']}건")
    print(f"🚨 지진 경보: {stats['earthquake_alerts']}건 ({alert_rate:.1f}%)")
    print(f"📊 클래스별 분포:")
    for class_name, count in stats['class_counts'].items():
        if count > 0:
            percentage = count / stats['total_events'] * 100
            icon = '🔴' if class_name == '지진' else '🟠' if class_name == '규칙적산업진동' else '🟢'
            print(f"   {icon} {class_name}: {count}건 ({percentage:.1f}%)")

# =========================== 메인 루프 (influx_base.py 스타일) ===========================

print("\\n🚀 === influx_base.py + NEW2 실시간 지진 분석 시스템 시작! ===")
print(f"🧠 NEW2 모델: {'로딩됨' if new2_model else '없음'}")
print(f"📊 분류 클래스: {list(NEW2_CONFIG['classes'].values())}")
print(f"🔍 감시 포트: {PORTS}")
print(f"⏱️ 체크 주기: {CHECK_INTERVAL}초")
print(f"💾 저장 위치:")
print(f"   원시 데이터: {RAW_SAVE_DIR}")
print(f"   AI 분석: {AI_SAVE_DIR}")
print(f"   지진 경보: {ALERT_SAVE_DIR}")
print("="*60)

try:
    # influx_base.py와 동일한 메인 루프 구조
    while True:
        now = datetime.utcnow()
        start = now - timedelta(seconds=CHECK_INTERVAL)

        for port in PORTS:
            # InfluxDB 쿼리 (기존과 동일)
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
                    if not isinstance(intensity, (int, float)):
                        continue

                    if intensity >= 3.0:
                        event_time = record.get_time().astimezone()
                        kst_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        print(f"\\n🔥 === 진도 {intensity:.2f} 감지 ===")
                        print(f"📅 시간: {kst_time}")
                        print(f"🌐 포트: {port}")

                        # 이벤트 정보 저장 (NEW2 분석용)
                        event_info = {
                            'time': kst_time,
                            'port': port,
                            'intensity': intensity
                        }

                        print("⏳ 40초 데이터 수집 대기...")
                        time.sleep(40)  # 기존과 동일한 대기 시간

                        # Node-RED 호출 (기존과 동일)
                        encoded_time = quote(kst_time)
                        url = f"{NODERED_BASE_URL}/{encoded_time}/{port}"
                        print(f"🔗 Node-RED 호출: {url}")

                        try:
                            res = requests.get(url, timeout=30)
                            if res.status_code == 200:
                                data = res.json()
                                if not data:
                                    print("⚠️ 응답 데이터 없음 (저장 생략)")
                                    continue

                                # 기존 방식대로 CSV 저장
                                flattened_data = [flatten_row(row) for row in data]
                                
                                file_time = kst_time.replace(":", "-").replace(" ", "_")
                                filename = f"event_{port}_{file_time}.csv"
                                raw_filepath = os.path.join(RAW_SAVE_DIR, filename)

                                with open(raw_filepath, "w", newline="", encoding="utf-8") as f:
                                    writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                                    writer.writeheader()
                                    writer.writerows(flattened_data)

                                print(f"📁 원시 데이터 저장: {filename}")

                                # === NEW2 AI 분석 추가 ===
                                if new2_model is not None:
                                    print(f"\\n🧠 === NEW2 AI 분석 시작 ===")
                                    
                                    # 전처리
                                    X, preprocess_msg = preprocess_for_new2(raw_filepath)
                                    print(f"🔄 전처리: {preprocess_msg}")
                                    
                                    if X is not None:
                                        # AI 분석
                                        ai_result, analysis_msg = analyze_with_new2(X)
                                        print(f"🎯 분석: {analysis_msg}")
                                        
                                        if ai_result:
                                            # 결과 출력
                                            print(f"\\n📊 === NEW2 분석 결과 ===")
                                            print(f"🔴 지진: {ai_result['earthquake_prob']:.4f} ({ai_result['earthquake_prob']*100:.1f}%)")
                                            print(f"🟠 규칙적산업진동: {ai_result['industrial_prob']:.4f} ({ai_result['industrial_prob']*100:.1f}%)")
                                            print(f"🟢 불규칙생활진동: {ai_result['living_prob']:.4f} ({ai_result['living_prob']*100:.1f}%)")
                                            print(f"🎯 최종 분류: {ai_result['class_name']} (신뢰도: {ai_result['confidence']:.4f})")
                                            
                                            # 지진 경보 처리
                                            if ai_result['is_earthquake']:
                                                print(f"\\n🚨🚨🚨 지진 경보 발령! 🚨🚨🚨")
                                                print(f"🔴 지진 확률: {ai_result['earthquake_prob']*100:.1f}%")
                                                print(f"📊 신뢰도: {ai_result['confidence']*100:.1f}%")
                                                
                                                # 지진 경보 로그 저장
                                                save_earthquake_alert(ai_result, event_info)
                                                
                                            else:
                                                status_msg = {
                                                    'NO_ALERT': '정상 - 비지진 진동',
                                                    'LOW_CONFIDENCE': '신뢰도 부족으로 경보 억제',
                                                    'LOW_CONFIDENCE_GAP': '신뢰도 차이 부족으로 경보 억제'
                                                }.get(ai_result['alert_status'], '알 수 없는 상태')
                                                
                                                print(f"\\n✅ {status_msg}")
                                                print(f"📊 분류: {ai_result['class_name']}")
                                                print(f"📊 신뢰도: {ai_result['confidence']*100:.1f}%")
                                            
                                            # AI 결과 저장 (event_info 포함)
                                            save_ai_result(raw_filepath, ai_result, event_info)
                                            
                                            # 통계 업데이트
                                            update_stats(ai_result)
                                            
                                            # 주기적 통계 출력 (5회마다)
                                            if stats['total_events'] % 5 == 0:
                                                print_stats()
                                        
                                    else:
                                        print("❌ NEW2 전처리 실패")
                                else:
                                    print("⚠️ NEW2 모델 없음 - 데이터 수집만 수행")

                            else:
                                print("❌ Node-RED 응답 오류:", res.status_code)
                                
                        except requests.exceptions.Timeout:
                            print("❌ Node-RED 요청 타임아웃")
                        except Exception as e:
                            print("❌ 요청 실패:", e)

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print(f"\\n\\n🛑 === 사용자 중단 ===")
    print_stats()
    
    print(f"\\n💾 데이터 저장 위치:")
    print(f"   원시 데이터: {RAW_SAVE_DIR}")
    print(f"   AI 분석 결과: {AI_SAVE_DIR}")
    print(f"   지진 경보 로그: {ALERT_SAVE_DIR}")
    
    print(f"\\n🎉 influx_base.py + NEW2 시스템이 안전하게 종료되었습니다!")

except Exception as e:
    print(f"\\n❌ 시스템 오류: {e}")
    import traceback
    traceback.print_exc()
    print_stats()
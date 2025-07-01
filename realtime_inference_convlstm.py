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

# ✅ 3클래스 모델 경로로 수정
MODEL_PATH = 'convlstm_3class_model.h5'

# ✅ 실시간 CSV 저장 위치
OUTPUT_DIR = "C:/earthquake_modeling/earthquake_project_v3/influxLogs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 3클래스 ConvLSTM 모델 불러오기
print("🔄 3클래스 ConvLSTM 모델 로딩 중...")
convlstm_model = load_model(MODEL_PATH)
print("✅ 3클래스 모델 로딩 완료!")

# ✅ 3클래스 정의
CLASS_NAMES = {0: '지진', 1: '불규칙생활진동', 2: '규칙적산업진동'}
CLASS_COLORS = {0: '🔴', 1: '🟢', 2: '🟠'}

# ✅ 올바른 전처리 함수
def preprocess_csv_for_3class_convlstm(csv_path):
    """학습과 동일한 전처리 방식 적용"""
    try:
        df = pd.read_csv(csv_path)
        
        # 필수 컬럼 확인
        required_cols = ['sensor1_x', 'sensor1_y', 'sensor1_z']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ 필수 컬럼 누락: {required_cols}")
            return None, None
        
        # 3축 데이터 추출
        sensor_x = df['sensor1_x'].astype(float).values
        sensor_y = df['sensor1_y'].astype(float).values
        sensor_z = df['sensor1_z'].astype(float).values
        
        # 데이터 결합
        data = np.stack([sensor_x, sensor_y, sensor_z], axis=1)
        
        # 4000 샘플로 맞추기 (학습과 동일)
        target_samples = 4000
        if data.shape[0] < target_samples:
            # 부족하면 0으로 패딩
            pad_width = ((0, target_samples - data.shape[0]), (0, 0))
            data = np.pad(data, pad_width, mode='constant')
        elif data.shape[0] > target_samples:
            # 넘치면 자르기
            data = data[:target_samples]
        
        # ConvLSTM 입력 형태로 변환 (학습과 동일)
        d = data.reshape(40, 100, 3)        # (40, 100, 3)
        d = np.transpose(d, (0, 2, 1))      # (40, 3, 100)
        d = np.expand_dims(d, axis=-1)      # (40, 3, 100, 1)
        d = np.expand_dims(d, axis=0)       # (1, 40, 3, 100, 1)
        
        # z-score 정규화 (학습과 동일)
        mean = d.mean()
        std = d.std()
        if std > 0:
            d = (d - mean) / std
        
        return d, df
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return None, None

# ✅ 예측 결과 해석 및 저장
def predict_and_save_result(filepath):
    """3클래스 예측 수행 및 결과 저장"""
    try:
        # 전처리
        X, df = preprocess_csv_for_3class_convlstm(filepath)
        if X is None:
            return
        
        # 예측 수행
        start_time = time.time()
        predictions = convlstm_model.predict(X, verbose=0)
        inference_time = time.time() - start_time
        
        # 결과 해석
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        class_name = CLASS_NAMES[predicted_class]
        class_icon = CLASS_COLORS[predicted_class]
        
        # 결과 출력
        print(f"\n🧠 === 3클래스 예측 결과 ===")
        print(f"{class_icon} 예측 클래스: {class_name}")
        print(f"🎯 신뢰도: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"⚡ 추론 시간: {inference_time:.3f}초")
        
        # 모든 클래스 확률 출력
        print(f"📊 클래스별 확률:")
        for i, (class_id, name) in enumerate(CLASS_NAMES.items()):
            prob = predictions[0][i]
            icon = CLASS_COLORS[class_id]
            print(f"  {icon} {name}: {prob:.4f} ({prob*100:.2f}%)")
        
        # 경보 판별
        if predicted_class == 0:  # 지진
            print(f"🚨 지진 감지! 즉시 경보 발령!")
        else:
            print(f"✅ 비지진 진동 ({class_name}) - 경보 억제")
        
        # CSV에 예측 결과 추가
        if df is not None:
            df['predicted_class'] = predicted_class
            df['predicted_class_name'] = class_name
            df['confidence'] = confidence
            df['inference_time'] = inference_time
            
            # 저장
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"📁 예측 결과 포함 저장 완료: {filepath}")
        
    except Exception as e:
        print(f"❌ 예측 실패: {e}")

# InfluxDB A (진도 감지용)
INFLUX_TOKEN = "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg=="
INFLUX_ORG = "kds"
INFLUX_BUCKET = "Lasung_3sensor of Max"
INFLUX_URL = "http://118.129.145.82:8086"
PORTS = [6060, 7001, 7053, 7060, 7070, 8010, 8080]
CHECK_INTERVAL = 1

NODERED_BASE_URL = "http://118.129.145.82:8081/nodered/1min_event_lasung"
SAVE_DIR = "C:/earthquake_modeling/earthquake_project_v3/influxLogs/base"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()


print("\n🚀 3클래스 ConvLSTM 실시간 지진 감지 시작!")
print(f"📊 분류 클래스: {list(CLASS_NAMES.values())}")
print(f"🔍 감시 대상 포트: {PORTS}")
print("="*60)

# ✅ 실시간 감시 루프
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
                        print("⏳ 40초 데이터 수집 대기 중...")
                        time.sleep(40)

                        encoded_time = quote(kst_time)
                        url = f"http://118.129.145.82:8081/nodered/1min_event_lasung/{encoded_time}/{port}"

                        try:
                            res = requests.get(url)
                            print(f"🌐 API 호출 결과: {res.status_code}")

                            if res.status_code == 200:
                                # CSV 저장
                                file_time = kst_time.replace(":", "-").replace(" ", "_")
                                filename = f"event_{port}_{file_time}.csv"
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

                                    # 3클래스 예측 수행
                                    predict_and_save_result(filepath)

                                except Exception as e:
                                    print(f"❌ 파일 처리 실패: {e}")

                        except Exception as e:
                            print(f"❌ API 호출 실패: {e}")

        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\n\n🛑 사용자에 의해 중단됨")
    print("👋 3클래스 실시간 지진 감지 시스템 종료")
#!/usr/bin/env python3
"""
🚀 enhanced_earthquake_detector.py (경로 및 URL 수정 버전)
"""

import os
import time
import numpy as np
import pandas as pd
import csv
import requests
import json
import pickle
from datetime import datetime, timedelta
from urllib.parse import quote
from influxdb_client import InfluxDBClient
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
except Exception as e:
    print("❌ TensorFlow 또는 protobuf 충돌: ", e)
    load_model = None

class EnhancedRealtimeDetector:
    def __init__(self):
        # InfluxDB (influx_base.py 기반)
        self.INFLUX_URL = "http://118.129.145.82:8086"
        self.INFLUX_TOKEN = "ZyegXlVhIdA26zFakbWjgVX863_pAtfXJPfsLGlA0wtfTxl7BHZJlMNLT5HHudXk58VzVScGnugA36w_buC4Zg=="
        self.INFLUX_ORG = "kds"
        self.INFLUX_BUCKET = "Lasung_3sensor of Max"
        self.PORTS = [6060, 7001, 7053, 7060, 7070, 8010, 8080]
        self.CHECK_INTERVAL = 1
        
        self.NODERED_BASE_URL = "http://192.168.68.53:1880/1min_event_lasung"
        
        # 디렉토리 경로 (influx_base.py SAVE_DIR 기반)
        self.BASE_SAVE_DIR = "C:/earthquake_modeling/earthquake_project_v3/influxLogs"
        self.RAW_DATA_DIR = os.path.join(self.BASE_SAVE_DIR, "base")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_SAVE_DIR, "processed")
        self.MODEL_DIR = "models"
        
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
        # 모델 경로 (influx_base.py 경로 기반 적용)
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, "convlstm_3class_model.h5")
        self.ENSEMBLE_MODEL_PATH = os.path.join(self.MODEL_DIR, "ensemble_model.pkl")
        
        self.models = {}
        self.scaler = StandardScaler()
        
        self.initialize_system()

    def initialize_system(self):
        try:
            self.client = InfluxDBClient(url=self.INFLUX_URL, token=self.INFLUX_TOKEN, org=self.INFLUX_ORG)
            self.query_api = self.client.query_api()
            print("✅ InfluxDB 연결 성공")
        except Exception as e:
            print(f"❌ InfluxDB 연결 실패: {e}")
        self.load_models()
        print("✅ 시스템 초기화 완료")

    def load_models(self):
        if load_model and os.path.exists(self.MODEL_PATH):
            self.models['convlstm'] = load_model(self.MODEL_PATH)
            print(f"✅ 모델 로드 성공: {self.MODEL_PATH}")
        else:
            print(f"⚠️ 모델 파일 없음: {self.MODEL_PATH}")
        
        if os.path.exists(self.ENSEMBLE_MODEL_PATH):
            with open(self.ENSEMBLE_MODEL_PATH, 'rb') as f:
                self.models['ensemble'] = pickle.load(f)
            print(f"✅ 앙상블 모델 로드 성공: {self.ENSEMBLE_MODEL_PATH}")
        else:
            print(f"⚠️ 앙상블 모델 파일 없음: {self.ENSEMBLE_MODEL_PATH}")

    def run_monitoring_loop(self):
        print("🚀 실시간 감시 시작")
        try:
            while True:
                now = datetime.utcnow()
                start = now - timedelta(seconds=self.CHECK_INTERVAL)
                
                for port in self.PORTS:
                    query = f'''
                    from(bucket: "{self.INFLUX_BUCKET}")
                      |> range(start: {start.isoformat()}Z, stop: {now.isoformat()}Z)
                      |> filter(fn: (r) => r._field == "intensity" and r._measurement == "{port}")
                      |> sort(columns: ["_time"], desc: true)
                      |> limit(n:1)
                    '''
                    try:
                        result = self.query_api.query(org=self.INFLUX_ORG, query=query)
                        for table in result:
                            for record in table.records:
                                print(f"🌐 {port} | 📅 {record.get_time()} | 값: {record.get_value()}")
                    except Exception as e:
                        print(f"❌ 쿼리 실패: {e}")
                time.sleep(self.CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("🛑 감시 중단")

def main():
    detector = EnhancedRealtimeDetector()
    detector.run_monitoring_loop()

if __name__ == "__main__":
    main()
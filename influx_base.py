import time
from datetime import datetime, timedelta
import requests
from influxdb_client import InfluxDBClient
from urllib.parse import quote
import os
import csv

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

def flatten_row(row):
    flat = {
        "timestamp": "'" + row.get("timestamp"),
        "counter": row.get("counter")
    }
    for i in range(1, 4):
        sensor_key = f"sensor_{i}"
        sensor_data = row.get(sensor_key, {})
        for key, value in sensor_data.items():
            flat[f"{sensor_key}_{key}"] = value
    return flat

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
                if not isinstance(intensity, (int, float)):
                    continue

                if intensity >= 3.0:
                    event_time = record.get_time().astimezone()
                    kst_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                    print("🔥 진도 3 이상 발생:", kst_time, f"(포트: {port})")

                    time.sleep(40)  # 데이터 쌓일 시간 확보

                    encoded_time = quote(kst_time)
                    url = f"{NODERED_BASE_URL}/{encoded_time}/{port}"
                    print("🔗 Node-RED 호출:", url)

                    try:
                        res = requests.get(url)
                        if res.status_code == 200:
                            data = res.json()
                            if not data:
                                print("⚠️ 응답 데이터 없음 (저장 생략)")
                                continue

                            flattened_data = [flatten_row(row) for row in data]

                            os.makedirs(SAVE_DIR, exist_ok=True)
                            file_time = kst_time.replace(":", "-").replace(" ", "_")
                            filename = f"event_{port}_{file_time}.csv"
                            filepath = os.path.join(SAVE_DIR, filename)

                            with open(filepath, "w", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                                writer.writeheader()
                                writer.writerows(flattened_data)

                            print(f"📁 저장 완료: {filepath}")
                        else:
                            print("❌ Node-RED 응답 오류:", res.status_code)
                    except Exception as e:
                        print("❌ 요청 실패:", e)

    time.sleep(CHECK_INTERVAL)

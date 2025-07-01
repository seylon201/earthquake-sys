import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# -----------------------------
# 유틸 함수 - 센서 데이터 파싱
# -----------------------------
def parse_trigger_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            try:
                _, value_part = line.strip().split(':', 1)
                values = value_part.strip().split(',')
                if len(values) < 7:
                    continue
                sensor1_x = float(values[2].strip())
                sensor1_y = float(values[3].strip())
                sensor1_z = float(values[4].strip())
                magnitude = float(values[5].strip())
                data.append([sensor1_x, sensor1_y, sensor1_z, magnitude])
            except:
                continue
    return np.array(data)

# -----------------------------
# 설정
# -----------------------------
input_dir = './data/trigger_sample_3rd/'  # 트리거 데이터 폴더 경로
output_file = './data/trigger_prediction_results_lstm.csv'  # LSTM 결과 저장 경로

# -----------------------------
# LSTM 모델 로딩
# -----------------------------
model = load_model('lstm_model.h5')
print("✅ LSTM 모델 로딩 완료")

# -----------------------------
# 파일 탐색
# -----------------------------
files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
print(f"총 {len(files)}개 파일 발견")

# -----------------------------
# 스킵 카운터 초기화
# -----------------------------
skipped_no_magnitude = 0
skipped_insufficient_length = 0
results = []

# -----------------------------
# 파일별 예측 처리
# -----------------------------
for file in files:
    file_path = os.path.join(input_dir, file)
    data = parse_trigger_txt(file_path)

    if data.shape[0] == 0:
        print(f"⚠ 파싱 실패 또는 빈 파일: {file}")
        continue

    magnitude_column = data[:, 3]
    valid_indices = np.where(magnitude_column >= 3.0)[0]

    if len(valid_indices) == 0:
        print(f"⚠ 진도 3 이상 구간 없음: {file}")
        skipped_no_magnitude += 1
        continue

    idx = valid_indices[0]

    if idx < 1500 or idx + 2500 >= len(data):
        print(f"⚠ 데이터 부족: {file} (이벤트 인덱스: {idx}, 총 샘플 수: {len(data)})")
        continue

    # 슬라이싱 및 LSTM 입력 형태 변환 (1, 40, 300)
    sliced = data[idx - 1500: idx + 2500, :3]  # (4000, 3)
    d = sliced.reshape(40, 300)               # (40, 300)
    d = np.expand_dims(d, axis=0)             # (1, 40, 300)

    # 예측 수행
    y_prob = model.predict(d)[0][0]
    y_pred = 1 if y_prob >= 0.47 else 0  # 필요시 threshold 조정 가능

    results.append({
        'file': file,
        'probability': y_prob,
        'prediction': y_pred
    })

# -----------------------------
# 결과 저장
# -----------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ LSTM 결과 저장 완료: {output_file}")

# -----------------------------
# 처리 요약
# -----------------------------
print(f"⚠ 진도 3 이상 구간 없음 스킵: {skipped_no_magnitude}건")
print(f"⚠ 데이터 부족 스킵: {skipped_insufficient_length}건")
print(f"✅ 최종 처리 완료: {len(results)}건")
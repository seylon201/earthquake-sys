# 공통전처리함수
import numpy as np
import pandas as pd

def parse_trigger_txt(file_path):
    """
    트리거 텍스트 파일에서 sensor1_x/y/z와 magnitude 추출
    """
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
    return np.array(data)  # shape: (6000, 4)
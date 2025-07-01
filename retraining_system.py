#!/usr/bin/env python3
"""
지진 오보 분석 시스템 재학습 시스템
전문가 수정된 데이터를 활용한 증분 학습 (Incremental Learning) 구현
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from typing import List, Dict, Tuple, Optional
import shutil
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retraining_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NEW2RetrainingSystem:
    """NEW2 지진 오보 분석 시스템 재학습 관리자"""
    
    def __init__(self, config: Dict):
        """
        재학습 시스템 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.base_model_path = config.get('base_model_path', 'new2_convlstm_3class_best.h5')
        self.retraining_data_dir = config.get('retraining_data_dir', 'retraining_data')
        self.retrained_models_dir = config.get('retrained_models_dir', 'retrained_models')
        self.analysis_dirs = config.get('analysis_dirs', ['influxLogs/new2_analysis', 'influxLogs/base'])
        
        # 디렉토리 생성
        os.makedirs(self.retraining_data_dir, exist_ok=True)
        os.makedirs(self.retrained_models_dir, exist_ok=True)
        os.makedirs(f"{self.retraining_data_dir}/corrected", exist_ok=True)
        os.makedirs(f"{self.retraining_data_dir}/processed", exist_ok=True)
        
        # 재학습 상태 파일
        self.status_file = os.path.join(self.retraining_data_dir, 'retraining_status.json')
        self.load_status()
        
        # NEW2 클래스 정의
        self.class_names = {0: '지진', 1: '규칙적산업진동', 2: '불규칙생활진동'}
        
        logger.info("NEW2 재학습 시스템 초기화 완료")
    
    def load_status(self):
        """재학습 상태 로드"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r', encoding='utf-8') as f:
                self.status = json.load(f)
        else:
            self.status = {
                'total_corrections': 0,
                'last_retraining': None,
                'retraining_count': 0,
                'corrected_files': [],
                'model_versions': [],
                'performance_history': []
            }
    
    def save_status(self):
        """재학습 상태 저장"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, ensure_ascii=False, indent=2)
    
    def collect_corrected_data(self) -> List[Dict]:
        """
        전문가 수정된 데이터 수집
        
        Returns:
            수정된 데이터 파일 목록
        """
        corrected_files = []
        new_corrections = 0
        
        logger.info("전문가 수정 데이터 수집 시작...")
        
        for analysis_dir in self.analysis_dirs:
            if not os.path.exists(analysis_dir):
                continue
                
            for filename in os.listdir(analysis_dir):
                if filename.endswith('.csv'):
                    filepath = os.path.join(analysis_dir, filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        
                        # 전문가 수정 여부 확인
                        if 'expert_corrected' in df.columns and df['expert_corrected'].iloc[0]:
                            file_info = {
                                'filepath': filepath,
                                'filename': filename,
                                'original_class': None,
                                'corrected_class': int(df['expert_corrected_class'].iloc[0]),
                                'corrected_class_name': df['expert_corrected_class_name'].iloc[0],
                                'correction_timestamp': df['expert_corrected_timestamp'].iloc[0],
                                'file_size': len(df),
                                'data_shape': df.shape
                            }
                            
                            # AI 원본 분석 결과 확인
                            if 'ai_predicted_class' in df.columns:
                                file_info['original_class'] = int(df['ai_predicted_class'].iloc[0])
                            elif 'predicted_class' in df.columns:
                                file_info['original_class'] = int(df['predicted_class'].iloc[0])
                            
                            # 새로운 수정 파일인지 확인
                            if filename not in [f['filename'] for f in self.status['corrected_files']]:
                                new_corrections += 1
                                logger.info(f"새로운 수정 파일 발견: {filename}")
                                logger.info(f"  - 원본: {self.class_names.get(file_info['original_class'], 'Unknown')}")
                                logger.info(f"  - 수정: {file_info['corrected_class_name']}")
                            
                            corrected_files.append(file_info)
                            
                    except Exception as e:
                        logger.warning(f"파일 처리 오류 {filename}: {e}")
        
        # 상태 업데이트
        self.status['corrected_files'] = corrected_files
        self.status['total_corrections'] = len(corrected_files)
        self.save_status()
        
        logger.info(f"수집 완료: 총 {len(corrected_files)}개 수정 파일 ({new_corrections}개 신규)")
        
        return corrected_files
    
    def prepare_retraining_data(self, corrected_files: List[Dict]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        재학습용 데이터 준비
        
        Args:
            corrected_files: 수정된 파일 목록
            
        Returns:
            (X, y) 재학습 데이터 또는 None
        """
        if not corrected_files:
            logger.warning("재학습할 수정 데이터가 없습니다.")
            return None
        
        logger.info(f"재학습 데이터 준비 중... ({len(corrected_files)}개 파일)")
        
        X_data = []
        y_data = []
        processed_count = 0
        
        for file_info in corrected_files:
            try:
                df = pd.read_csv(file_info['filepath'])
                
                # 센서 데이터 추출
                sensor_data = self.extract_sensor_data(df)
                if sensor_data is not None:
                    X_data.append(sensor_data)
                    y_data.append(file_info['corrected_class'])
                    processed_count += 1
                else:
                    logger.warning(f"센서 데이터 추출 실패: {file_info['filename']}")
                    
            except Exception as e:
                logger.error(f"데이터 처리 오류 {file_info['filename']}: {e}")
        
        if processed_count == 0:
            logger.warning("처리 가능한 센서 데이터가 없습니다.")
            return None
        
        # NumPy 배열로 변환
        X = np.array(X_data)
        y = np.array(y_data)
        
        # NEW2 모델 형식으로 변환 (N, 40, 3, 100, 1)
        X_reshaped = self.reshape_for_new2_model(X)
        
        logger.info(f"재학습 데이터 준비 완료: X={X_reshaped.shape}, y={y.shape}")
        
        # 데이터 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        np.save(f"{self.retraining_data_dir}/processed/retraining_X_{timestamp}.npy", X_reshaped)
        np.save(f"{self.retraining_data_dir}/processed/retraining_y_{timestamp}.npy", y)
        
        return X_reshaped, y
    
    def extract_sensor_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        DataFrame에서 센서 데이터 추출
        
        Args:
            df: 센서 데이터가 포함된 DataFrame
            
        Returns:
            센서 데이터 배열 또는 None
        """
        # 센서 데이터 패턴 찾기
        sensor_patterns = [
            ('sensor_1_x', 'sensor_1_y', 'sensor_1_z'),
            ('x', 'y', 'z'),
            ('X', 'Y', 'Z'),
            ('acc_x', 'acc_y', 'acc_z')
        ]
        
        for x_col, y_col, z_col in sensor_patterns:
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                # 데이터 추출 및 정규화
                x_data = pd.to_numeric(df[x_col], errors='coerce').fillna(0).values
                y_data = pd.to_numeric(df[y_col], errors='coerce').fillna(0).values
                z_data = pd.to_numeric(df[z_col], errors='coerce').fillna(0).values
                
                # 3축 데이터 결합
                sensor_data = np.column_stack([x_data, y_data, z_data])
                
                return sensor_data
        
        return None
    
    def reshape_for_new2_model(self, X: np.ndarray) -> np.ndarray:
        """
        NEW2 ConvLSTM 모델 형식에 맞게 데이터 변형
        
        Args:
            X: 원본 센서 데이터 (N, length, 3)
            
        Returns:
            NEW2 형식 데이터 (N, 40, 3, 100, 1)
        """
        N = X.shape[0]
        
        # 타겟 형태: (N, 40, 3, 100, 1)
        X_reshaped = np.zeros((N, 40, 3, 100, 1))
        
        for i in range(N):
            data = X[i]  # (length, 3)
            
            # 40초 × 100Hz = 4000 샘플로 맞추기
            target_length = 4000
            
            if len(data) >= target_length:
                # 데이터가 충분하면 앞에서부터 4000개 사용
                data = data[:target_length]
            else:
                # 데이터가 부족하면 패딩
                padding_length = target_length - len(data)
                padding = np.zeros((padding_length, 3))
                data = np.vstack([data, padding])
            
            # (4000, 3) -> (40, 3, 100, 1) 로 변형
            data_reshaped = data.reshape(40, 100, 3).transpose(0, 2, 1)  # (40, 3, 100)
            data_reshaped = np.expand_dims(data_reshaped, axis=-1)  # (40, 3, 100, 1)
            
            X_reshaped[i] = data_reshaped
        
        return X_reshaped
    
    def perform_incremental_learning(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional[str]:
        """
        증분 학습 수행
        
        Args:
            X_train: 재학습 데이터
            y_train: 재학습 라벨
            
        Returns:
            새로운 모델 경로 또는 None
        """
        try:
            logger.info("증분 학습 시작...")
            
            # 기존 모델 로드
            if not os.path.exists(self.base_model_path):
                logger.error(f"기본 모델을 찾을 수 없습니다: {self.base_model_path}")
                return None
            
            model = load_model(self.base_model_path, compile=False)
            logger.info(f"기본 모델 로드 완료: {self.base_model_path}")
            
            # 모델 컴파일 (낮은 학습률로)
            model.compile(
                optimizer=Adam(learning_rate=0.0001),  # 기존보다 낮은 학습률
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # 라벨을 원-핫 인코딩에서 sparse로 변환 (필요시)
            if len(y_train.shape) > 1:
                y_train = np.argmax(y_train, axis=1)
            
            # 증분 학습 실행
            history = model.fit(
                X_train, y_train,
                batch_size=16,  # 작은 배치 크기
                epochs=10,      # 적은 에포크 수
                validation_split=0.2,
                verbose=1,
                callbacks=[
                    EarlyStopping(patience=3, restore_best_weights=True)
                ]
            )
            
            # 새 모델 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_model_path = os.path.join(
                self.retrained_models_dir, 
                f"new2_retrained_{timestamp}.h5"
            )
            
            model.save(new_model_path)
            logger.info(f"재학습 모델 저장 완료: {new_model_path}")
            
            # 성능 기록
            final_accuracy = history.history['accuracy'][-1]
            final_val_accuracy = history.history['val_accuracy'][-1]
            
            model_info = {
                'model_path': new_model_path,
                'timestamp': timestamp,
                'training_samples': len(X_train),
                'final_accuracy': float(final_accuracy),
                'final_val_accuracy': float(final_val_accuracy),
                'base_model': self.base_model_path,
                'retraining_method': 'incremental_learning'
            }
            
            # 상태 업데이트
            self.status['model_versions'].append(model_info)
            self.status['retraining_count'] += 1
            self.status['last_retraining'] = timestamp
            self.status['performance_history'].append({
                'timestamp': timestamp,
                'accuracy': final_accuracy,
                'val_accuracy': final_val_accuracy,
                'samples_count': len(X_train)
            })
            
            self.save_status()
            
            logger.info(f"증분 학습 완료!")
            logger.info(f"  - 훈련 정확도: {final_accuracy:.4f}")
            logger.info(f"  - 검증 정확도: {final_val_accuracy:.4f}")
            logger.info(f"  - 훈련 샘플 수: {len(X_train)}")
            
            return new_model_path
            
        except Exception as e:
            logger.error(f"증분 학습 실패: {e}")
            return None
    
    def get_retraining_status(self) -> Dict:
        """재학습 상태 정보 반환"""
        return {
            'total_corrections': self.status['total_corrections'],
            'last_retraining': self.status['last_retraining'],
            'retraining_count': self.status['retraining_count'],
            'available_for_retraining': len(self.status['corrected_files']),
            'model_versions': len(self.status['model_versions']),
            'latest_model': self.status['model_versions'][-1] if self.status['model_versions'] else None
        }
    
    def should_trigger_retraining(self, min_corrections: int = 10) -> bool:
        """
        재학습 트리거 조건 확인
        
        Args:
            min_corrections: 최소 수정 개수
            
        Returns:
            재학습이 필요한지 여부
        """
        return self.status['total_corrections'] >= min_corrections
    
    def run_full_retraining_pipeline(self, min_corrections: int = 10) -> Dict:
        """
        전체 재학습 파이프라인 실행
        
        Args:
            min_corrections: 재학습 트리거 최소 수정 개수
            
        Returns:
            재학습 결과 정보
        """
        result = {
            'success': False,
            'message': '',
            'new_model_path': None,
            'performance': None
        }
        
        try:
            # 1. 수정 데이터 수집
            corrected_files = self.collect_corrected_data()
            
            if len(corrected_files) < min_corrections:
                result['message'] = f"수정 데이터 부족: {len(corrected_files)}/{min_corrections}"
                return result
            
            # 2. 재학습 데이터 준비
            retraining_data = self.prepare_retraining_data(corrected_files)
            if retraining_data is None:
                result['message'] = "재학습 데이터 준비 실패"
                return result
            
            X_train, y_train = retraining_data
            
            # 3. 증분 학습 수행
            new_model_path = self.perform_incremental_learning(X_train, y_train)
            if new_model_path is None:
                result['message'] = "증분 학습 실패"
                return result
            
            # 4. 재학습에 사용된 파일들을 완료 상태로 마킹
            self.mark_files_as_retrained()
            
            # 5. 성공
            result['success'] = True
            result['message'] = f"재학습 완료: {len(corrected_files)}개 수정 데이터 사용"
            result['new_model_path'] = new_model_path
            result['performance'] = self.status['performance_history'][-1]
            
        except Exception as e:
            result['message'] = f"재학습 파이프라인 오류: {str(e)}"
            logger.error(result['message'])
        
        return result
    
    def mark_files_as_retrained(self):
        """재학습에 사용된 모든 파일을 완료 상태로 마킹"""
        try:
            logger.info("재학습 완료 파일 상태 업데이트 시작...")
            updated_count = 0
            
            for analysis_dir in self.analysis_dirs:
                if not os.path.exists(analysis_dir):
                    continue
                    
                for filename in os.listdir(analysis_dir):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(analysis_dir, filename)
                        
                        try:
                            df = pd.read_csv(filepath)
                            
                            # retraining_completed 컬럼 추가/업데이트
                            df['retraining_completed'] = True
                            df['retraining_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # 파일 저장
                            df.to_csv(filepath, index=False, encoding='utf-8-sig')
                            updated_count += 1
                            
                        except Exception as e:
                            logger.warning(f"파일 상태 업데이트 실패 {filename}: {e}")
                            continue
            
            logger.info(f"재학습 완료 상태 업데이트 완료: {updated_count}개 파일")
            return updated_count
            
        except Exception as e:
            logger.error(f"파일 상태 업데이트 오류: {e}")
            return 0

def main():
    """재학습 시스템 테스트"""
    config = {
        'base_model_path': 'new2_convlstm_3class_best.h5',
        'retraining_data_dir': 'retraining_data',
        'retrained_models_dir': 'retrained_models',
        'analysis_dirs': [
            'C:/earthquake_modeling/earthquake_project_v3/influxLogs/new2_analysis',
            'C:/earthquake_modeling/earthquake_project_v3/influxLogs/base'
        ]
    }
    
    # 재학습 시스템 초기화
    retraining_system = NEW2RetrainingSystem(config)
    
    # 상태 확인
    status = retraining_system.get_retraining_status()
    print("=== 재학습 시스템 상태 ===")
    print(f"총 수정 개수: {status['total_corrections']}")
    print(f"재학습 횟수: {status['retraining_count']}")
    print(f"마지막 재학습: {status['last_retraining']}")
    
    # 재학습 실행 (10개 이상 수정 시)
    if retraining_system.should_trigger_retraining(min_corrections=5):  # 테스트용으로 5개로 낮춤
        print("\n=== 재학습 실행 ===")
        result = retraining_system.run_full_retraining_pipeline(min_corrections=5)
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"새 모델: {result['new_model_path']}")
            print(f"성능: {result['performance']}")
        else:
            print(f"❌ {result['message']}")
    else:
        print("재학습 조건을 만족하지 않습니다.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EnhancedEarthquakePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
    def load_integrated_dataset(self, filename=None):
        """
        통합 데이터셋 로드
        """
        if filename is None:
            # 가장 최신 통합 데이터셋 파일 찾기
            import glob
            files = glob.glob("integrated_earthquake_dataset_*.csv")
            if not files:
                raise FileNotFoundError("통합 데이터셋 파일을 찾을 수 없습니다.")
            filename = max(files, key=lambda x: x.split('_')[-1])
        
        print(f"📂 데이터셋 로드: {filename}")
        
        try:
            self.data = pd.read_csv(filename)
            print(f"✅ 로드 성공: {len(self.data)}개 레코드, {len(self.data.columns)}개 컬럼")
            return self.data
        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            return None
    
    def analyze_feature_distribution(self):
        """
        특성 분포 분석 및 이상값 감지
        """
        print("\n🔍 특성 분포 분석")
        print("="*50)
        
        # 수치형 컬럼 분석
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print(f"📊 컬럼 타입 분석:")
        print(f"   수치형: {len(numeric_cols)}개")
        print(f"   범주형: {len(categorical_cols)}개")
        
        # 주요 수치형 특성 분석
        key_features = ['mag', 'latitude', 'longitude', 'depth', 'year']
        available_features = [col for col in key_features if col in numeric_cols]
        
        print(f"\n📈 주요 수치형 특성 분석:")
        for feature in available_features:
            if feature in self.data.columns:
                stats = self.data[feature].describe()
                print(f"   {feature}:")
                print(f"     범위: {stats['min']:.2f} ~ {stats['max']:.2f}")
                print(f"     평균: {stats['mean']:.2f}, 표준편차: {stats['std']:.2f}")
                
                # 이상값 감지 (IQR 방법)
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.data[(self.data[feature] < lower_bound) | 
                                   (self.data[feature] > upper_bound)]
                if len(outliers) > 0:
                    print(f"     ⚠️ 이상값: {len(outliers)}개 ({len(outliers)/len(self.data)*100:.1f}%)")
                else:
                    print(f"     ✅ 이상값 없음")
        
        # 범주형 특성 분석
        print(f"\n📋 주요 범주형 특성 분석:")
        key_categorical = ['region', 'net', 'magType', 'event_type']
        for feature in key_categorical:
            if feature in self.data.columns:
                value_counts = self.data[feature].value_counts()
                print(f"   {feature}: {len(value_counts)}개 카테고리")
                print(f"     상위 3개: {dict(value_counts.head(3))}")
        
        return numeric_cols, categorical_cols
    
    def create_advanced_features(self):
        """
        고급 특성 생성
        """
        print("\n🔧 고급 특성 생성")
        print("="*50)
        
        # 1. 지리적 특성
        if 'latitude' in self.data.columns and 'longitude' in self.data.columns:
            print("🌍 지리적 특성 생성...")
            
            # 거리 기반 특성 (주요 도시/단층선으로부터의 거리)
            # 샌프란시스코 (37.7749, -122.4194)
            sf_lat, sf_lon = 37.7749, -122.4194
            self.data['distance_to_sf'] = np.sqrt(
                (self.data['latitude'] - sf_lat)**2 + 
                (self.data['longitude'] - sf_lon)**2
            )
            
            # 로스앤젤레스 (34.0522, -118.2437)
            la_lat, la_lon = 34.0522, -118.2437
            self.data['distance_to_la'] = np.sqrt(
                (self.data['latitude'] - la_lat)**2 + 
                (self.data['longitude'] - la_lon)**2
            )
            
            # 위도대 구분
            self.data['latitude_zone'] = pd.cut(self.data['latitude'], 
                                              bins=[-90, 0, 30, 60, 90], 
                                              labels=['Southern', 'Tropical', 'Temperate', 'Northern'])
            
            print("   ✅ 지리적 특성 생성 완료")
        
        # 2. 시간적 특성
        if 'datetime' in self.data.columns:
            print("⏰ 시간적 특성 생성...")
            
            try:
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])
                
                # 계절 정보
                self.data['season'] = self.data['datetime'].dt.month % 12 // 3 + 1
                season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
                self.data['season_name'] = self.data['season'].map(season_map)
                
                # 하루 중 시간대
                self.data['time_of_day'] = pd.cut(self.data['hour'], 
                                                bins=[0, 6, 12, 18, 24], 
                                                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                                include_lowest=True)
                
                # 지진 활동 주기 (lunar cycle 근사)
                days_since_epoch = (self.data['datetime'] - pd.Timestamp('2000-01-01')).dt.days
                self.data['lunar_phase'] = (days_since_epoch % 29.5) / 29.5
                
                print("   ✅ 시간적 특성 생성 완료")
            except Exception as e:
                print(f"   ⚠️ 시간적 특성 생성 실패: {e}")
        
        # 3. 진도 관련 특성
        if 'mag' in self.data.columns:
            print("⚡ 진도 관련 특성 생성...")
            
            # 에너지 추정 (리히터 스케일 기반)
            self.data['energy_log'] = 11.8 + 1.5 * self.data['mag']  # log10(에너지)
            self.data['energy_relative'] = 10 ** (1.5 * (self.data['mag'] - 3.0))  # 진도 3 기준 상대 에너지
            
            # 진도 카테고리 (더 세분화)
            mag_bins = [0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 10.0]
            mag_labels = ['micro', 'minor', 'light', 'moderate', 'strong', 'major', 'great', 'extreme']
            self.data['magnitude_category'] = pd.cut(self.data['mag'], bins=mag_bins, labels=mag_labels)
            
            # 진도 정규화 (여러 방법)
            self.data['mag_normalized'] = (self.data['mag'] - self.data['mag'].min()) / (self.data['mag'].max() - self.data['mag'].min())
            self.data['mag_standardized'] = (self.data['mag'] - self.data['mag'].mean()) / self.data['mag'].std()
            
            print("   ✅ 진도 관련 특성 생성 완료")
        
        # 4. 네트워크/지역 특성
        if 'net' in self.data.columns:
            print("🌐 네트워크 특성 생성...")
            
            # 네트워크 활동 수준
            network_counts = self.data['net'].value_counts()
            self.data['network_activity_level'] = self.data['net'].map(network_counts)
            
            # 주요 네트워크 여부
            major_networks = ['ci', 'CI', 'NC', 'nc', 'us', 'ak']
            self.data['is_major_network'] = self.data['net'].isin(major_networks)
            
            print("   ✅ 네트워크 특성 생성 완료")
        
        print(f"🎉 특성 생성 완료! 총 {len(self.data.columns)}개 컬럼")
        return self.data
    
    def handle_missing_values(self):
        """
        결측값 처리
        """
        print("\n🔧 결측값 처리")
        print("="*50)
        
        missing_before = self.data.isnull().sum().sum()
        print(f"처리 전 결측값: {missing_before}개")
        
        # 수치형 컬럼: 중간값으로 대체
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().sum() > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                print(f"   {col}: 중간값 {median_val:.2f}로 대체")
        
        # 범주형 컬럼: 최빈값으로 대체
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                mode_val = self.data[col].mode().iloc[0] if len(self.data[col].mode()) > 0 else 'Unknown'
                self.data[col].fillna(mode_val, inplace=True)
                print(f"   {col}: 최빈값 '{mode_val}'로 대체")
        
        missing_after = self.data.isnull().sum().sum()
        print(f"✅ 처리 후 결측값: {missing_after}개")
        
        return self.data
    
    def encode_categorical_features(self):
        """
        범주형 특성 인코딩
        """
        print("\n🔢 범주형 특성 인코딩")
        print("="*50)
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # 기존 프로젝트 호환을 위한 특별 처리
        priority_cols = ['region', 'net', 'magType', 'magnitude_category']
        
        for col in categorical_cols:
            if col in priority_cols or self.data[col].nunique() <= 20:  # 카테고리가 20개 이하인 경우만
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"   {col}: {self.data[col].nunique()}개 카테고리 → 수치형 변환")
        
        print(f"✅ {len(self.label_encoders)}개 범주형 컬럼 인코딩 완료")
        return self.data
    
    def create_convlstm_features(self):
        """
        ConvLSTM 모델용 특성 생성
        """
        print("\n🧠 ConvLSTM 모델용 특성 생성")
        print("="*50)
        
        # 핵심 특성 선택
        core_features = []
        
        # 1. 필수 지진 특성
        essential_features = ['mag', 'latitude', 'longitude', 'depth']
        for feature in essential_features:
            if feature in self.data.columns:
                core_features.append(feature)
        
        # 2. 시간 특성
        time_features = ['year', 'month', 'day', 'hour', 'season']
        for feature in time_features:
            if feature in self.data.columns:
                core_features.append(feature)
        
        # 3. 지역 특성
        region_features = ['region_code', 'distance_to_sf', 'distance_to_la']
        for feature in region_features:
            if feature in self.data.columns:
                core_features.append(feature)
        
        # 4. 고급 특성
        advanced_features = ['energy_relative', 'network_activity_level', 'lunar_phase']
        for feature in advanced_features:
            if feature in self.data.columns:
                core_features.append(feature)
        
        # 5. 범주형 인코딩된 특성
        encoded_features = [col for col in self.data.columns if col.endswith('_encoded')]
        core_features.extend(encoded_features[:5])  # 상위 5개만
        
        # 특성 행렬 생성
        self.feature_matrix = self.data[core_features].copy()
        
        print(f"📊 선택된 특성 ({len(core_features)}개):")
        for i, feature in enumerate(core_features):
            print(f"   {i+1:2d}. {feature}")
        
        # 정규화
        feature_scaled = self.scaler.fit_transform(self.feature_matrix)
        self.feature_matrix_scaled = pd.DataFrame(feature_scaled, columns=core_features)
        
        print(f"✅ ConvLSTM 특성 행렬 생성 완료: {self.feature_matrix_scaled.shape}")
        
        return self.feature_matrix_scaled, core_features
    
    def prepare_for_existing_project(self):
        """
        기존 프로젝트 형태로 데이터 준비
        """
        print("\n🔄 기존 프로젝트 호환 형태로 변환")
        print("="*50)
        
        # 기존 프로젝트에서 사용하는 컬럼 형태로 변환
        project_data = self.data.copy()
        
        # 클래스 라벨 (기존 프로젝트: 지진=0, 불규칙생활진동=1, 규칙적산업진동=2)
        project_data['class_label'] = 0  # 모든 데이터가 지진
        
        # 기존 프로젝트의 주요 특성들
        required_columns = {
            'magnitude': 'mag',
            'event_time': 'datetime', 
            'event_latitude': 'latitude',
            'event_longitude': 'longitude',
            'event_depth': 'depth',
            'event_region': 'region_code',
            'data_source': 'net'
        }
        
        for new_col, old_col in required_columns.items():
            if old_col in project_data.columns:
                project_data[new_col] = project_data[old_col]
        
        # 40초 슬라이싱을 위한 메타데이터 (시뮬레이션)
        project_data['trigger_timestamp'] = project_data['datetime']
        project_data['sampling_rate'] = 100  # 100Hz
        project_data['pre_trigger_seconds'] = 15
        project_data['post_trigger_seconds'] = 25
        project_data['total_samples'] = 4000  # 40초 * 100Hz
        
        # 3축 가속도 시뮬레이션 메타데이터
        project_data['has_xyz_data'] = True
        project_data['preprocessing_applied'] = 'z_score_normalized'
        project_data['tensor_shape'] = '(40, 3, 100, 1)'
        
        print(f"✅ 기존 프로젝트 호환 데이터 생성")
        print(f"   레코드 수: {len(project_data)}")
        print(f"   새로운 컬럼: {list(required_columns.keys())}")
        
        return project_data
    
    def create_train_test_split(self, test_size=0.2, validation_size=0.2):
        """
        훈련/검증/테스트 데이터 분할
        """
        print(f"\n📊 데이터 분할 (훈련:{1-test_size-validation_size:.1f}, 검증:{validation_size:.1f}, 테스트:{test_size:.1f})")
        print("="*50)
        
        # 지역별 stratified split
        if 'region_code' in self.data.columns:
            stratify_col = self.data['region_code']
        else:
            stratify_col = None
        
        # 첫 번째 분할: 훈련+검증 vs 테스트
        train_val_data, test_data = train_test_split(
            self.data, 
            test_size=test_size, 
            stratify=stratify_col,
            random_state=42
        )
        
        # 두 번째 분할: 훈련 vs 검증
        val_ratio = validation_size / (1 - test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio,
            stratify=train_val_data['region_code'] if 'region_code' in train_val_data.columns else None,
            random_state=42
        )
        
        print(f"📋 분할 결과:")
        print(f"   훈련 데이터: {len(train_data):,}개 ({len(train_data)/len(self.data)*100:.1f}%)")
        print(f"   검증 데이터: {len(val_data):,}개 ({len(val_data)/len(self.data)*100:.1f}%)")
        print(f"   테스트 데이터: {len(test_data):,}개 ({len(test_data)/len(self.data)*100:.1f}%)")
        
        # 지역별 분포 확인
        if 'region' in self.data.columns:
            print(f"\n🌍 지역별 분할 확인:")
            for dataset_name, dataset in [('훈련', train_data), ('검증', val_data), ('테스트', test_data)]:
                region_dist = dataset['region'].value_counts()
                print(f"   {dataset_name}: {dict(region_dist)}")
        
        return train_data, val_data, test_data
    
    def save_preprocessed_data(self):
        """
        전처리된 데이터 저장
        """
        print("\n💾 전처리 데이터 저장")
        print("="*50)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 완전 전처리된 데이터
        full_filename = f"earthquake_preprocessed_full_{timestamp}.csv"
        self.data.to_csv(full_filename, index=False)
        print(f"📁 전체 전처리 데이터: {full_filename}")
        
        # 2. ConvLSTM용 특성 행렬
        if hasattr(self, 'feature_matrix_scaled'):
            features_filename = f"earthquake_features_convlstm_{timestamp}.csv"
            self.feature_matrix_scaled.to_csv(features_filename, index=False)
            print(f"📁 ConvLSTM 특성 행렬: {features_filename}")
        
        # 3. 기존 프로젝트 호환 데이터
        project_data = self.prepare_for_existing_project()
        project_filename = f"earthquake_project_compatible_{timestamp}.csv"
        project_data.to_csv(project_filename, index=False)
        print(f"📁 프로젝트 호환 데이터: {project_filename}")
        
        # 4. 데이터 분할
        train_data, val_data, test_data = self.create_train_test_split()
        
        train_filename = f"earthquake_train_{timestamp}.csv"
        val_filename = f"earthquake_val_{timestamp}.csv"
        test_filename = f"earthquake_test_{timestamp}.csv"
        
        train_data.to_csv(train_filename, index=False)
        val_data.to_csv(val_filename, index=False)
        test_data.to_csv(test_filename, index=False)
        
        print(f"📁 훈련 데이터: {train_filename}")
        print(f"📁 검증 데이터: {val_filename}")
        print(f"📁 테스트 데이터: {test_filename}")
        
        # 5. 전처리 메타데이터 저장
        import json
        metadata = {
            'preprocessing_info': {
                'timestamp': timestamp,
                'original_records': len(self.data),
                'final_features': len(self.data.columns),
                'label_encoders': {k: v.classes_.tolist() for k, v in self.label_encoders.items()}
            },
            'file_mapping': {
                'full_data': full_filename,
                'convlstm_features': features_filename if hasattr(self, 'feature_matrix_scaled') else None,
                'project_compatible': project_filename,
                'train_data': train_filename,
                'validation_data': val_filename,
                'test_data': test_filename
            },
            'data_statistics': {
                'magnitude_range': f"{self.data['mag'].min():.1f}-{self.data['mag'].max():.1f}",
                'regions': self.data['region'].value_counts().to_dict() if 'region' in self.data.columns else {},
                'time_range': f"{self.data['year'].min()}-{self.data['year'].max()}" if 'year' in self.data.columns else "Unknown"
            }
        }
        
        metadata_filename = f"preprocessing_metadata_{timestamp}.json"
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📋 전처리 메타데이터: {metadata_filename}")
        
        return {
            'full_data': full_filename,
            'project_compatible': project_filename,
            'train': train_filename,
            'validation': val_filename,
            'test': test_filename,
            'metadata': metadata_filename
        }

# 메인 실행 함수
def main():
    print("🚀 향상된 지진 데이터 전처리 시작")
    print("="*60)
    
    preprocessor = EnhancedEarthquakePreprocessor()
    
    # 1. 데이터 로드
    data = preprocessor.load_integrated_dataset()
    if data is None:
        return
    
    # 2. 특성 분포 분석
    numeric_cols, categorical_cols = preprocessor.analyze_feature_distribution()
    
    # 3. 고급 특성 생성
    enhanced_data = preprocessor.create_advanced_features()
    
    # 4. 결측값 처리
    clean_data = preprocessor.handle_missing_values()
    
    # 5. 범주형 특성 인코딩
    encoded_data = preprocessor.encode_categorical_features()
    
    # 6. ConvLSTM용 특성 생성
    feature_matrix, core_features = preprocessor.create_convlstm_features()
    
    # 7. 데이터 저장
    saved_files = preprocessor.save_preprocessed_data()
    
    print(f"\n🎉 전처리 완료!")
    print(f"📁 생성된 파일들:")
    for purpose, filename in saved_files.items():
        if filename:
            print(f"   - {purpose}: {filename}")
    
    print(f"\n📊 최종 결과:")
    print(f"   최종 특성: {len(enhanced_data.columns):,}개")
    print(f"   ConvLSTM 핵심 특성: {len(core_features)}개")
    
    print(f"\n💡 다음 단계:")
    print(f"   1. {saved_files['project_compatible']}로 기존 ConvLSTM 모델 재학습")
    print(f"   2. 지역별 교차 검증으로 일반화 성능 평가")
    print(f"   3. 새로운 특성들로 모델 성능 향상 확인")
    print(f"   4. 오경보율 11% → 5% 이하 목표 달성")

if __name__ == "__main__":
    main()
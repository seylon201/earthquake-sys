import requests
import json
import pandas as pd
import time
from datetime import datetime
import os

class ExpandedCESMDCollector:
    def __init__(self):
        self.base_url = "https://strongmotioncenter.org/wserv"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def collect_events_by_magnitude_range(self, mag_ranges, max_per_range=200):
        """
        진도 범위별로 이벤트 수집
        """
        print("🚀 확장된 CESMD 데이터 수집 시작")
        print("="*60)
        
        all_events = []
        total_collected = 0
        
        for min_mag, max_mag in mag_ranges:
            print(f"\n📊 진도 {min_mag}-{max_mag} 범위 수집 중...")
            
            params = {
                'minmag': min_mag,
                'maxmag': max_mag,
                'format': 'json',
                'nodata': '404'
            }
            
            url = f"{self.base_url}/events/query"
            
            try:
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('type') == 'FeatureCollection':
                    features = data.get('features', [])
                    print(f"   🔍 총 {len(features)}개 이벤트 발견")
                    
                    # 최대 개수만큼 선택
                    selected_features = features[:max_per_range]
                    print(f"   📋 {len(selected_features)}개 선택 (최대 {max_per_range}개)")
                    
                    # features를 events 형태로 변환
                    range_events = []
                    for i, feature in enumerate(selected_features):
                        properties = feature.get('properties', {})
                        geometry = feature.get('geometry', {})
                        
                        # 좌표 정보 추가
                        if geometry.get('coordinates'):
                            coords = geometry['coordinates']
                            properties['longitude'] = coords[0] if len(coords) > 0 else None
                            properties['latitude'] = coords[1] if len(coords) > 1 else None
                            properties['depth'] = coords[2] if len(coords) > 2 else None
                        
                        # 메타데이터 추가
                        properties['cesmd_index'] = total_collected + i
                        properties['mag_range'] = f"{min_mag}-{max_mag}"
                        properties['collection_timestamp'] = datetime.now().isoformat()
                        
                        range_events.append(properties)
                    
                    all_events.extend(range_events)
                    total_collected += len(range_events)
                    
                    print(f"   ✅ {len(range_events)}개 수집 완료")
                    
                    # API 부하 방지
                    time.sleep(1)
                    
                else:
                    print(f"   ❌ 예상과 다른 응답 형태: {type(data)}")
                    
            except Exception as e:
                print(f"   ❌ 진도 {min_mag}-{max_mag} 수집 실패: {e}")
                continue
        
        print(f"\n🎉 전체 수집 완료!")
        print(f"   총 수집: {total_collected}개 이벤트")
        
        return all_events
    
    def analyze_collected_data(self, events):
        """
        수집된 데이터 분석
        """
        if not events:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        print(f"\n📈 수집 데이터 분석")
        print("="*40)
        
        df = pd.DataFrame(events)
        
        # 기본 통계
        print(f"📊 기본 통계:")
        print(f"   총 이벤트: {len(df)}개")
        
        if 'mag' in df.columns:
            print(f"   진도 범위: {df['mag'].min():.1f} ~ {df['mag'].max():.1f}")
            print(f"   평균 진도: {df['mag'].mean():.2f}")
            
            # 진도별 분포
            mag_distribution = df['mag'].value_counts().sort_index()
            print(f"   진도별 분포:")
            for mag, count in mag_distribution.head(10).items():
                print(f"     진도 {mag}: {count}개")
        
        # 지역별 분포
        if 'country' in df.columns:
            country_dist = df['country'].value_counts()
            print(f"\n🌍 국가별 분포:")
            for country, count in country_dist.head(5).items():
                print(f"   {country}: {count}개")
        
        if 'net' in df.columns:
            network_dist = df['net'].value_counts()
            print(f"\n🌐 네트워크별 분포:")
            for network, count in network_dist.head(5).items():
                print(f"   {network}: {count}개")
        
        # 시간 분포
        if 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['time'])
                year_dist = df['datetime'].dt.year.value_counts().sort_index()
                print(f"\n📅 연도별 분포:")
                for year, count in year_dist.head(10).items():
                    print(f"   {year}: {count}개")
            except:
                print(f"\n📅 시간 분석 실패")
        
        return df
    
    def save_expanded_dataset(self, events):
        """
        확장된 데이터셋 저장
        """
        if not events:
            print("❌ 저장할 데이터가 없습니다.")
            return None, None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # DataFrame 생성
        df = pd.DataFrame(events)
        
        # CSV 저장
        csv_filename = f"cesmd_expanded_events_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"📁 확장 이벤트 데이터 저장: {csv_filename}")
        
        # JSON 백업 저장
        json_filename = f"cesmd_expanded_backup_{timestamp}.json"
        backup_data = {
            'events': events,
            'collection_info': {
                'timestamp': timestamp,
                'total_events': len(events),
                'magnitude_range': f"{df['mag'].min():.1f}-{df['mag'].max():.1f}" if 'mag' in df.columns else "Unknown",
                'collection_method': 'expanded_magnitude_ranges'
            }
        }
        
        with open(json_filename, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        print(f"📁 JSON 백업 저장: {json_filename}")
        
        # 요약 정보
        print(f"\n📋 저장된 데이터 요약:")
        print(f"   파일 크기: {os.path.getsize(csv_filename)} bytes")
        print(f"   데이터 형태: {df.shape[0]}행 × {df.shape[1]}열")
        
        return csv_filename, json_filename
    
    def create_earthquake_project_format(self, events):
        """
        지진 감지 프로젝트용 형태로 데이터 변환
        """
        if not events:
            return None
        
        df = pd.DataFrame(events)
        
        # 필요한 컬럼 선택 및 정리
        project_data = df.copy()
        
        # 시간 정보 추가
        if 'time' in project_data.columns:
            try:
                project_data['datetime'] = pd.to_datetime(project_data['time'])
                project_data['year'] = project_data['datetime'].dt.year
                project_data['month'] = project_data['datetime'].dt.month
                project_data['day'] = project_data['datetime'].dt.day
                project_data['hour'] = project_data['datetime'].dt.hour
                project_data['day_of_week'] = project_data['datetime'].dt.dayofweek
            except:
                print("⚠️ 시간 정보 변환 실패")
        
        # 정규화된 특성 추가
        if 'mag' in project_data.columns:
            project_data['mag_normalized'] = (
                (project_data['mag'] - project_data['mag'].min()) / 
                (project_data['mag'].max() - project_data['mag'].min())
            )
            
            # 진도 카테고리 추가
            project_data['mag_category'] = pd.cut(
                project_data['mag'], 
                bins=[0, 3.5, 4.5, 5.5, 10], 
                labels=['small', 'moderate', 'large', 'major']
            )
        
        # 지진 클래스 라벨 (기존 프로젝트 호환)
        project_data['event_type'] = 'earthquake'
        project_data['class_label'] = 0  # 지진 = 0
        
        # 지역 코딩
        if 'country' in project_data.columns:
            country_map = {'US': 0, 'JP': 1, 'KR': 2}  # 기존 프로젝트 기준
            project_data['country_code'] = project_data['country'].map(country_map).fillna(999)
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"earthquake_project_expanded_{timestamp}.csv"
        project_data.to_csv(filename, index=False)
        
        print(f"🎯 지진 프로젝트용 데이터 저장: {filename}")
        print(f"   데이터 크기: {len(project_data)}개 × {len(project_data.columns)}개 컬럼")
        
        return filename

# 실행 스크립트
def main():
    collector = ExpandedCESMDCollector()
    
    # 진도 범위 설정 (3.0-6.0을 세분화)
    magnitude_ranges = [
        (3.0, 3.5),  # 소규모 지진
        (3.5, 4.0),  # 소-중간 규모
        (4.0, 4.5),  # 중간 규모
        (4.5, 5.0),  # 중-대 규모
        (5.0, 5.5),  # 대규모
        (5.5, 6.0),  # 주요 지진
    ]
    
    print(f"🎯 목표: 진도 3.0-6.0 범위 데이터 수집")
    print(f"📋 범위: {len(magnitude_ranges)}개 구간")
    print(f"🔢 예상 최대 수집량: {len(magnitude_ranges) * 200}개 이벤트")
    
    # 데이터 수집
    all_events = collector.collect_events_by_magnitude_range(
        magnitude_ranges, 
        max_per_range=200  # 각 범위당 최대 200개
    )
    
    if all_events:
        # 데이터 분석
        df = collector.analyze_collected_data(all_events)
        
        # 데이터 저장
        csv_file, json_file = collector.save_expanded_dataset(all_events)
        
        # 프로젝트용 형태로 변환
        project_file = collector.create_earthquake_project_format(all_events)
        
        print(f"\n🎉 수집 완료!")
        print(f"📁 생성된 파일:")
        print(f"   - 원본 데이터: {csv_file}")
        print(f"   - JSON 백업: {json_file}")
        print(f"   - 프로젝트용: {project_file}")
        
        print(f"\n💡 다음 단계:")
        print(f"   1. 기존 3,430개 + 새로운 {len(all_events)}개 = 총 {3430 + len(all_events)}개")
        print(f"   2. 확장된 데이터셋으로 ConvLSTM 모델 재학습")
        print(f"   3. 진도 3.0-6.0 범위에서 모델 성능 검증")
        
    else:
        print("❌ 데이터 수집 실패")

if __name__ == "__main__":
    main()
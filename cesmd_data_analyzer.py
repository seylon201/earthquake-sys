import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import glob
import os

class CESMDDataAnalyzer:
    def __init__(self):
        self.events_data = None
        self.records_data = None
        
    def load_latest_data(self):
        """
        최신 수집된 CESMD 데이터 로드
        """
        print("📂 수집된 CESMD 데이터 파일 검색...")
        
        # CSV 파일들 찾기
        event_files = glob.glob("cesmd_events_*.csv")
        json_files = glob.glob("cesmd_backup_*.json")
        
        if not event_files:
            print("❌ CESMD 이벤트 파일을 찾을 수 없습니다.")
            return False
        
        # 가장 최신 파일 선택
        latest_event_file = max(event_files, key=os.path.getctime)
        latest_json_file = max(json_files, key=os.path.getctime) if json_files else None
        
        print(f"📁 최신 이벤트 파일: {latest_event_file}")
        if latest_json_file:
            print(f"📁 최신 JSON 백업: {latest_json_file}")
        
        try:
            # 이벤트 데이터 로드
            self.events_data = pd.read_csv(latest_event_file)
            print(f"✅ 이벤트 데이터 로드 성공: {len(self.events_data)}개 이벤트")
            
            # JSON 백업 로드 (추가 정보용)
            if latest_json_file:
                with open(latest_json_file, 'r') as f:
                    self.json_backup = json.load(f)
                print(f"✅ JSON 백업 로드 성공")
            
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def analyze_events(self):
        """
        수집된 이벤트 데이터 분석
        """
        if self.events_data is None:
            print("❌ 분석할 이벤트 데이터가 없습니다.")
            return
        
        print("\n🔍 CESMD 이벤트 데이터 분석")
        print("="*50)
        
        # 기본 통계
        print("📊 기본 통계:")
        print(f"   총 이벤트 수: {len(self.events_data)}")
        
        if 'mag' in self.events_data.columns:
            print(f"   진도 범위: {self.events_data['mag'].min():.1f} ~ {self.events_data['mag'].max():.1f}")
            print(f"   평균 진도: {self.events_data['mag'].mean():.2f}")
        
        if 'country' in self.events_data.columns:
            country_counts = self.events_data['country'].value_counts()
            print(f"   주요 국가: {dict(country_counts.head(3))}")
        
        if 'net' in self.events_data.columns:
            network_counts = self.events_data['net'].value_counts()
            print(f"   주요 네트워크: {dict(network_counts.head(3))}")
        
        # 시간 분석
        if 'time' in self.events_data.columns:
            print(f"\n📅 시간 분포:")
            try:
                # 시간 컬럼을 datetime으로 변환
                self.events_data['datetime'] = pd.to_datetime(self.events_data['time'])
                
                earliest = self.events_data['datetime'].min()
                latest = self.events_data['datetime'].max()
                print(f"   기간: {earliest.strftime('%Y-%m-%d')} ~ {latest.strftime('%Y-%m-%d')}")
                
                # 연도별 분포
                yearly_counts = self.events_data['datetime'].dt.year.value_counts().sort_index()
                print(f"   연도별 분포: {dict(yearly_counts.head(5))}")
                
            except Exception as e:
                print(f"   ⚠️ 시간 분석 실패: {e}")
        
        # 지리적 분포
        if 'latitude' in self.events_data.columns and 'longitude' in self.events_data.columns:
            print(f"\n🌍 지리적 분포:")
            lat_range = f"{self.events_data['latitude'].min():.2f} ~ {self.events_data['latitude'].max():.2f}"
            lon_range = f"{self.events_data['longitude'].min():.2f} ~ {self.events_data['longitude'].max():.2f}"
            print(f"   위도 범위: {lat_range}")
            print(f"   경도 범위: {lon_range}")
        
        # 데이터 품질 확인
        print(f"\n🔍 데이터 품질:")
        missing_data = self.events_data.isnull().sum()
        if missing_data.sum() > 0:
            print("   누락 데이터:")
            for col, missing in missing_data[missing_data > 0].items():
                print(f"     {col}: {missing}개 ({missing/len(self.events_data)*100:.1f}%)")
        else:
            print("   ✅ 누락 데이터 없음")
        
        return self.events_data
    
    def create_visualizations(self):
        """
        데이터 시각화 생성
        """
        if self.events_data is None:
            print("❌ 시각화할 데이터가 없습니다.")
            return
        
        print("\n📈 데이터 시각화 생성 중...")
        
        # 시각화 설정
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CESMD 지진 데이터 분석', fontsize=16, fontweight='bold')
        
        # 1. 진도 분포 히스토그램
        if 'mag' in self.events_data.columns:
            axes[0, 0].hist(self.events_data['mag'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('진도 분포')
            axes[0, 0].set_xlabel('진도 (Magnitude)')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 국가별 분포
        if 'country' in self.events_data.columns:
            country_counts = self.events_data['country'].value_counts().head(10)
            axes[0, 1].bar(range(len(country_counts)), country_counts.values, color='lightcoral')
            axes[0, 1].set_title('국가별 지진 발생 횟수')
            axes[0, 1].set_xlabel('국가')
            axes[0, 1].set_ylabel('발생 횟수')
            axes[0, 1].set_xticks(range(len(country_counts)))
            axes[0, 1].set_xticklabels(country_counts.index, rotation=45)
        
        # 3. 시간별 분포 (연도별)
        if 'datetime' in self.events_data.columns:
            yearly_counts = self.events_data['datetime'].dt.year.value_counts().sort_index()
            axes[1, 0].plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6)
            axes[1, 0].set_title('연도별 지진 발생 추세')
            axes[1, 0].set_xlabel('연도')
            axes[1, 0].set_ylabel('발생 횟수')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 지리적 분포 (위도-경도 산점도)
        if 'latitude' in self.events_data.columns and 'longitude' in self.events_data.columns:
            scatter = axes[1, 1].scatter(
                self.events_data['longitude'], 
                self.events_data['latitude'],
                c=self.events_data.get('mag', 'blue'),
                cmap='Reds',
                alpha=0.6,
                s=30
            )
            axes[1, 1].set_title('지진 발생 위치')
            axes[1, 1].set_xlabel('경도')
            axes[1, 1].set_ylabel('위도')
            
            # 진도별 컬러바 추가
            if 'mag' in self.events_data.columns:
                plt.colorbar(scatter, ax=axes[1, 1], label='진도')
        
        plt.tight_layout()
        
        # 시각화 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cesmd_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 시각화 저장: {filename}")
        
        plt.show()
        
        return filename
    
    def export_for_earthquake_project(self):
        """
        지진 감지 프로젝트용 데이터 내보내기
        """
        if self.events_data is None:
            print("❌ 내보낼 데이터가 없습니다.")
            return
        
        print("\n📤 지진 감지 프로젝트용 데이터 내보내기...")
        
        # 필요한 컬럼만 선택
        export_columns = []
        available_columns = self.events_data.columns.tolist()
        
        # 우선순위 컬럼들
        priority_columns = [
            'mag', 'magType', 'time', 'place', 'latitude', 'longitude', 'depth',
            'net', 'country', 'state', 'detail'
        ]
        
        for col in priority_columns:
            if col in available_columns:
                export_columns.append(col)
        
        # 추가 컬럼들
        for col in available_columns:
            if col not in export_columns:
                export_columns.append(col)
        
        export_data = self.events_data[export_columns].copy()
        
        # 데이터 정제
        print("🔧 데이터 정제 중...")
        
        # 시간 컬럼 표준화
        if 'time' in export_data.columns:
            try:
                export_data['time'] = pd.to_datetime(export_data['time'])
                export_data['year'] = export_data['time'].dt.year
                export_data['month'] = export_data['time'].dt.month
                export_data['day'] = export_data['time'].dt.day
                export_data['hour'] = export_data['time'].dt.hour
            except:
                print("⚠️ 시간 컬럼 변환 실패")
        
        # 진도 정규화
        if 'mag' in export_data.columns:
            export_data['mag_normalized'] = (export_data['mag'] - export_data['mag'].min()) / (export_data['mag'].max() - export_data['mag'].min())
        
        # 클래스 라벨 추가 (기존 프로젝트와 호환)
        export_data['event_type'] = 'earthquake'  # 모든 데이터가 지진
        export_data['class_label'] = 0  # 지진 = 0 (기존 프로젝트 기준)
        
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"earthquake_project_data_{timestamp}.csv"
        export_data.to_csv(filename, index=False)
        
        print(f"✅ 지진 감지 프로젝트용 데이터 저장: {filename}")
        print(f"   데이터 크기: {len(export_data)}개 이벤트, {len(export_data.columns)}개 컬럼")
        print(f"   진도 범위: {export_data['mag'].min():.1f} ~ {export_data['mag'].max():.1f}")
        
        # 데이터 요약 리포트
        summary = {
            'total_events': len(export_data),
            'date_range': f"{export_data['time'].min()} ~ {export_data['time'].max()}" if 'time' in export_data.columns else "Unknown",
            'magnitude_range': f"{export_data['mag'].min():.1f} ~ {export_data['mag'].max():.1f}" if 'mag' in export_data.columns else "Unknown",
            'countries': export_data['country'].value_counts().to_dict() if 'country' in export_data.columns else {},
            'networks': export_data['net'].value_counts().to_dict() if 'net' in export_data.columns else {}
        }
        
        summary_filename = f"earthquake_project_summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📋 요약 리포트 저장: {summary_filename}")
        
        return filename, summary_filename

# 사용 예시
if __name__ == "__main__":
    analyzer = CESMDDataAnalyzer()
    
    # 데이터 로드
    if analyzer.load_latest_data():
        # 데이터 분석
        events_df = analyzer.analyze_events()
        
        # 시각화 생성
        plot_file = analyzer.create_visualizations()
        
        # 지진 감지 프로젝트용 데이터 내보내기
        export_file, summary_file = analyzer.export_for_earthquake_project()
        
        print(f"\n🎉 분석 완료!")
        print(f"📊 생성된 파일들:")
        print(f"   - 시각화: {plot_file}")
        print(f"   - 프로젝트 데이터: {export_file}")
        print(f"   - 요약 리포트: {summary_file}")
        
        print(f"\n💡 다음 단계 제안:")
        print(f"   1. {export_file}을 기존 지진 데이터와 합치기")
        print(f"   2. 새로운 지역/시간대 데이터로 모델 일반화 성능 향상")
        print(f"   3. 다양한 진도 범위로 모델 견고성 테스트")
    else:
        print("❌ 먼저 seismic_data_harvester.py를 실행하여 데이터를 수집하세요.")
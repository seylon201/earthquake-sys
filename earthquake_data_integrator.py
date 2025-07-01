import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import glob
import os
from pathlib import Path

class EarthquakeDataIntegrator:
    def __init__(self):
        self.existing_data = None
        self.cesmd_data = None
        self.integrated_data = None
        
    def load_existing_data(self):
        """
        기존 지진 프로젝트 데이터 로드 시뮬레이션
        """
        print("📂 기존 지진 프로젝트 데이터 분석...")
        
        # 기존 데이터 구조 (프로젝트 문서 기반)
        existing_summary = {
            'total_events': 3430,
            'sources': {
                'japan_ohtashi': 1170,
                'korea_kma': 2260
            },
            'magnitude_range': '3.0+',
            'regions': ['Japan', 'Korea'],
            'data_format': '40초 슬라이싱, 3축 가속도',
            'preprocessing': 'Z-score 정규화, (40, 3, 100, 1) 텐서'
        }
        
        print("📊 기존 데이터 현황:")
        print(f"   총 이벤트: {existing_summary['total_events']:,}개")
        print(f"   일본 오타시: {existing_summary['sources']['japan_ohtashi']:,}개")
        print(f"   한국 KMA: {existing_summary['sources']['korea_kma']:,}개")
        print(f"   진도 범위: {existing_summary['magnitude_range']}")
        print(f"   지역: {', '.join(existing_summary['regions'])}")
        
        return existing_summary
    
    def load_cesmd_data(self):
        """
        새로 수집된 CESMD 데이터 로드
        """
        print("\n📂 CESMD 데이터 로드 중...")
        
        # 가장 최신 CESMD 파일 찾기
        cesmd_files = glob.glob("earthquake_project_expanded_*.csv")
        
        if not cesmd_files:
            print("❌ CESMD 데이터 파일을 찾을 수 없습니다.")
            print("   먼저 cesmd_expanded_collector.py를 실행하세요.")
            return None
        
        # 가장 최신 파일 선택
        latest_file = max(cesmd_files, key=os.path.getctime)
        print(f"📁 로드할 파일: {latest_file}")
        
        try:
            self.cesmd_data = pd.read_csv(latest_file)
            print(f"✅ CESMD 데이터 로드 성공: {len(self.cesmd_data)}개 이벤트")
            
            # 기본 통계
            print(f"📊 CESMD 데이터 요약:")
            print(f"   진도 범위: {self.cesmd_data['mag'].min():.1f} ~ {self.cesmd_data['mag'].max():.1f}")
            print(f"   평균 진도: {self.cesmd_data['mag'].mean():.2f}")
            print(f"   데이터 컬럼: {len(self.cesmd_data.columns)}개")
            
            return self.cesmd_data
            
        except Exception as e:
            print(f"❌ CESMD 데이터 로드 실패: {e}")
            return None
    
    def analyze_data_compatibility(self):
        """
        기존 데이터와 CESMD 데이터의 호환성 분석
        """
        if self.cesmd_data is None:
            print("❌ CESMD 데이터가 없습니다.")
            return
        
        print("\n🔍 데이터 호환성 분석")
        print("="*50)
        
        # 컬럼 구조 분석
        print("📋 CESMD 데이터 컬럼 구조:")
        for i, col in enumerate(self.cesmd_data.columns):
            print(f"   {i+1:2d}. {col}")
        
        # 필수 컬럼 확인
        required_columns = ['mag', 'time', 'latitude', 'longitude', 'place', 'net']
        missing_columns = []
        present_columns = []
        
        for col in required_columns:
            if col in self.cesmd_data.columns:
                present_columns.append(col)
            else:
                missing_columns.append(col)
        
        print(f"\n✅ 보유 필수 컬럼: {present_columns}")
        if missing_columns:
            print(f"❌ 누락 필수 컬럼: {missing_columns}")
        
        # 데이터 품질 검사
        print(f"\n🔍 데이터 품질 검사:")
        
        # 결측값 확인
        missing_data = self.cesmd_data.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            print(f"⚠️  결측값 발견: {total_missing}개")
            print("   주요 결측값:")
            for col, missing in missing_data[missing_data > 0].head().items():
                percentage = (missing / len(self.cesmd_data)) * 100
                print(f"     {col}: {missing}개 ({percentage:.1f}%)")
        else:
            print("✅ 결측값 없음")
        
        # 진도 분포 분석
        if 'mag' in self.cesmd_data.columns:
            print(f"\n📊 진도 분포 분석:")
            mag_stats = self.cesmd_data['mag'].describe()
            print(f"   최소값: {mag_stats['min']:.1f}")
            print(f"   25%: {mag_stats['25%']:.1f}")
            print(f"   중간값: {mag_stats['50%']:.1f}")
            print(f"   75%: {mag_stats['75%']:.1f}")
            print(f"   최대값: {mag_stats['max']:.1f}")
            
            # 진도 카테고리별 개수
            mag_categories = pd.cut(self.cesmd_data['mag'], 
                                  bins=[3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], 
                                  labels=['3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0', '5.0-5.5', '5.5-6.0'])
            print(f"\n   진도 카테고리별 분포:")
            for category, count in mag_categories.value_counts().sort_index().items():
                print(f"     {category}: {count}개")
        
        return self.cesmd_data
    
    def create_integrated_dataset(self, existing_summary):
        """
        기존 데이터와 CESMD 데이터 통합
        """
        if self.cesmd_data is None:
            print("❌ 통합할 CESMD 데이터가 없습니다.")
            return None
        
        print("\n🔄 데이터 통합 작업 시작")
        print("="*50)
        
        # CESMD 데이터를 기본으로 사용 (실제 데이터 있음)
        integrated = self.cesmd_data.copy()
        
        # 기존 데이터 호환을 위한 컬럼 추가/수정
        print("🔧 데이터 표준화 중...")
        
        # 1. 지역 정보 표준화
        if 'country' in integrated.columns:
            # 기존 프로젝트와 호환되는 지역 코드 생성
            region_mapping = {
                'US': 'North America',
                'Mexico': 'North America', 
                'Japan': 'East Asia',
                'Korea': 'East Asia',
                'South Pacific Ocean': 'Pacific',
                'North Pacific Ocean': 'Pacific',
                'Philippine Sea': 'Pacific'
            }
            
            integrated['region'] = integrated['country'].map(region_mapping).fillna('Other')
            integrated['region_code'] = integrated['region'].map({
                'East Asia': 0,      # 기존 데이터 (한국, 일본)
                'North America': 1,  # 새로운 데이터 (미국, 멕시코)
                'Pacific': 2,        # 해양 지진
                'Other': 3
            })
        
        # 2. 데이터 소스 표시
        integrated['data_source'] = 'CESMD'
        integrated['is_new_data'] = True
        
        # 3. 시간 정보 표준화
        if 'time' in integrated.columns:
            try:
                integrated['datetime'] = pd.to_datetime(integrated['time'])
                integrated['year'] = integrated['datetime'].dt.year
                integrated['month'] = integrated['datetime'].dt.month
                integrated['day_of_year'] = integrated['datetime'].dt.dayofyear
                integrated['hour'] = integrated['datetime'].dt.hour
            except Exception as e:
                print(f"⚠️ 시간 변환 중 오류: {e}")
        
        # 4. 기존 프로젝트 호환 컬럼 추가
        integrated['class_label'] = 0  # 모든 데이터가 지진 (기존 프로젝트 기준)
        integrated['event_type'] = 'earthquake'
        
        # 5. 통합 메타데이터 추가
        integrated['integration_timestamp'] = datetime.now().isoformat()
        integrated['total_dataset_size'] = existing_summary['total_events'] + len(integrated)
        
        self.integrated_data = integrated
        
        print(f"✅ 데이터 통합 완료")
        print(f"   CESMD 데이터: {len(integrated)}개")
        print(f"   기존 데이터: {existing_summary['total_events']}개")
        print(f"   통합 총계: {existing_summary['total_events'] + len(integrated)}개")
        
        return integrated
    
    def create_comprehensive_analysis(self):
        """
        통합 데이터의 종합 분석
        """
        if self.integrated_data is None:
            print("❌ 분석할 통합 데이터가 없습니다.")
            return
        
        print("\n📊 통합 데이터 종합 분석")
        print("="*50)
        
        df = self.integrated_data
        
        # 1. 기본 통계
        print("📈 기본 통계:")
        print(f"   총 레코드: {len(df):,}개")
        print(f"   총 컬럼: {len(df.columns)}개")
        print(f"   메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 2. 지역별 분포
        if 'region' in df.columns:
            print(f"\n🌍 지역별 분포:")
            region_counts = df['region'].value_counts()
            for region, count in region_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {region}: {count:,}개 ({percentage:.1f}%)")
        
        # 3. 네트워크별 분포
        if 'net' in df.columns:
            print(f"\n🌐 상위 네트워크:")
            network_counts = df['net'].value_counts().head(8)
            for network, count in network_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {network}: {count:,}개 ({percentage:.1f}%)")
        
        # 4. 시간적 분포
        if 'year' in df.columns:
            print(f"\n📅 연도별 분포 (상위 10년):")
            year_counts = df['year'].value_counts().sort_index().tail(10)
            for year, count in year_counts.items():
                print(f"   {year}: {count:,}개")
        
        # 5. 진도 분포 상세 분석
        if 'mag' in df.columns:
            print(f"\n⚡ 진도 분포 분석:")
            print(f"   범위: {df['mag'].min():.1f} ~ {df['mag'].max():.1f}")
            print(f"   평균: {df['mag'].mean():.2f}")
            print(f"   표준편차: {df['mag'].std():.2f}")
            
            # 진도별 구간 분석
            mag_bins = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 10.0]
            mag_labels = ['3.0-3.4', '3.5-3.9', '4.0-4.4', '4.5-4.9', '5.0-5.4', '5.5-5.9', '6.0+']
            
            df['mag_category'] = pd.cut(df['mag'], bins=mag_bins, labels=mag_labels, include_lowest=True)
            mag_dist = df['mag_category'].value_counts().sort_index()
            
            print(f"   진도 구간별 분포:")
            for category, count in mag_dist.items():
                percentage = (count / len(df)) * 100
                print(f"     {category}: {count:,}개 ({percentage:.1f}%)")
        
        return df
    
    def create_visualizations(self):
        """
        통합 데이터 시각화
        """
        if self.integrated_data is None:
            print("❌ 시각화할 데이터가 없습니다.")
            return
        
        print("\n📊 데이터 시각화 생성 중...")
        
        df = self.integrated_data
        
        # 시각화 설정
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 2x3 그리드 생성
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 진도 분포 히스토그램
        ax1 = fig.add_subplot(gs[0, 0])
        if 'mag' in df.columns:
            ax1.hist(df['mag'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('진도 분포', fontsize=14, fontweight='bold')
            ax1.set_xlabel('진도 (Magnitude)')
            ax1.set_ylabel('빈도')
            ax1.grid(True, alpha=0.3)
        
        # 2. 지역별 분포 (파이 차트)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'region' in df.columns:
            region_counts = df['region'].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            ax2.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(region_counts)])
            ax2.set_title('지역별 분포', fontsize=14, fontweight='bold')
        
        # 3. 네트워크별 분포 (상위 10개)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'net' in df.columns:
            network_counts = df['net'].value_counts().head(10)
            bars = ax3.bar(range(len(network_counts)), network_counts.values, color='lightcoral')
            ax3.set_title('상위 네트워크별 분포', fontsize=14, fontweight='bold')
            ax3.set_xlabel('네트워크')
            ax3.set_ylabel('이벤트 수')
            ax3.set_xticks(range(len(network_counts)))
            ax3.set_xticklabels(network_counts.index, rotation=45)
            
            # 막대 위에 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 4. 연도별 추세 (최근 20년)
        ax4 = fig.add_subplot(gs[1, :])
        if 'year' in df.columns:
            recent_years = df[df['year'] >= 2000]
            yearly_counts = recent_years['year'].value_counts().sort_index()
            
            ax4.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6)
            ax4.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3)
            ax4.set_title('연도별 지진 발생 추세 (2000년 이후)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('연도')
            ax4.set_ylabel('지진 발생 횟수')
            ax4.grid(True, alpha=0.3)
        
        # 5. 지리적 분포 (위도-경도 산점도)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'latitude' in df.columns and 'longitude' in df.columns:
            scatter = ax5.scatter(df['longitude'], df['latitude'], 
                                c=df['mag'], cmap='Reds', alpha=0.6, s=20)
            ax5.set_title('지진 발생 위치 (진도별 색상)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('경도')
            ax5.set_ylabel('위도')
            
            # 컬러바 추가
            cbar = plt.colorbar(scatter, ax=ax5)
            cbar.set_label('진도')
        
        # 6. 진도 카테고리별 분포
        ax6 = fig.add_subplot(gs[2, 2])
        if 'mag_category' in df.columns:
            category_counts = df['mag_category'].value_counts().sort_index()
            bars = ax6.bar(range(len(category_counts)), category_counts.values, 
                          color='gold', edgecolor='black')
            ax6.set_title('진도 카테고리별 분포', fontsize=14, fontweight='bold')
            ax6.set_xlabel('진도 범위')
            ax6.set_ylabel('이벤트 수')
            ax6.set_xticks(range(len(category_counts)))
            ax6.set_xticklabels(category_counts.index, rotation=45)
            
            # 막대 위에 값 표시
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.suptitle('통합 지진 데이터셋 종합 분석', fontsize=18, fontweight='bold', y=0.98)
        
        # 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_earthquake_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 통합 분석 시각화 저장: {filename}")
        
        plt.show()
        return filename
    
    def save_integrated_dataset(self, existing_summary):
        """
        통합 데이터셋 최종 저장
        """
        if self.integrated_data is None:
            print("❌ 저장할 통합 데이터가 없습니다.")
            return None
        
        print("\n💾 통합 데이터셋 저장 중...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 메인 데이터셋 저장
        main_filename = f"integrated_earthquake_dataset_{timestamp}.csv"
        self.integrated_data.to_csv(main_filename, index=False)
        print(f"📁 통합 데이터셋 저장: {main_filename}")
        
        # 2. 요약 리포트 생성
        summary_report = {
            'integration_info': {
                'timestamp': timestamp,
                'total_records': len(self.integrated_data),
                'total_columns': len(self.integrated_data.columns)
            },
            'data_sources': {
                'existing_data': {
                    'count': existing_summary['total_events'],
                    'sources': existing_summary['sources']
                },
                'cesmd_data': {
                    'count': len(self.integrated_data),
                    'magnitude_range': f"{self.integrated_data['mag'].min():.1f}-{self.integrated_data['mag'].max():.1f}",
                    'regions': self.integrated_data['region'].value_counts().to_dict() if 'region' in self.integrated_data.columns else {}
                }
            },
            'combined_totals': {
                'total_events': existing_summary['total_events'] + len(self.integrated_data),
                'magnitude_coverage': '3.0-6.0+',
                'geographic_coverage': ['East Asia', 'North America', 'Pacific'],
                'time_span': f"{self.integrated_data['year'].min()}-{self.integrated_data['year'].max()}" if 'year' in self.integrated_data.columns else "Unknown"
            },
            'quality_metrics': {
                'missing_values': self.integrated_data.isnull().sum().sum(),
                'data_completeness': f"{(1 - self.integrated_data.isnull().sum().sum() / (len(self.integrated_data) * len(self.integrated_data.columns))) * 100:.1f}%"
            }
        }
        
        summary_filename = f"integration_report_{timestamp}.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"📋 통합 리포트 저장: {summary_filename}")
        
        # 3. 파일 크기 및 상세 정보
        file_size = os.path.getsize(main_filename)
        print(f"\n📊 저장 결과:")
        print(f"   파일 크기: {file_size / 1024 / 1024:.2f} MB")
        print(f"   레코드 수: {len(self.integrated_data):,}개")
        print(f"   컬럼 수: {len(self.integrated_data.columns)}개")
        print(f"   데이터 완성도: {summary_report['quality_metrics']['data_completeness']}")
        
        return main_filename, summary_filename, summary_report

# 메인 실행 함수
def main():
    print("🚀 지진 데이터 통합 및 분석 시작")
    print("="*60)
    
    integrator = EarthquakeDataIntegrator()
    
    # 1. 기존 데이터 현황 확인
    existing_summary = integrator.load_existing_data()
    
    # 2. CESMD 데이터 로드
    cesmd_data = integrator.load_cesmd_data()
    
    if cesmd_data is None:
        print("❌ CESMD 데이터를 로드할 수 없습니다.")
        return
    
    # 3. 데이터 호환성 분석
    integrator.analyze_data_compatibility()
    
    # 4. 데이터 통합
    integrated_data = integrator.create_integrated_dataset(existing_summary)
    
    if integrated_data is None:
        print("❌ 데이터 통합에 실패했습니다.")
        return
    
    # 5. 종합 분석
    integrator.create_comprehensive_analysis()
    
    # 6. 시각화 생성
    plot_filename = integrator.create_visualizations()
    
    # 7. 최종 저장
    main_file, summary_file, report = integrator.save_integrated_dataset(existing_summary)
    
    # 최종 결과 요약
    print(f"\n🎉 데이터 통합 완료!")
    print(f"📁 생성된 파일:")
    print(f"   - 통합 데이터셋: {main_file}")
    print(f"   - 통합 리포트: {summary_file}")
    print(f"   - 분석 시각화: {plot_filename}")
    
    print(f"\n📊 최종 결과:")
    print(f"   기존 데이터: {existing_summary['total_events']:,}개")
    print(f"   CESMD 데이터: {len(integrated_data):,}개") 
    print(f"   총 데이터: {report['combined_totals']['total_events']:,}개")
    print(f"   증가율: {((len(integrated_data) / existing_summary['total_events']) * 100):.1f}%")
    
    print(f"\n💡 다음 단계:")
    print(f"   1. {main_file}을 ConvLSTM 모델 학습에 사용")
    print(f"   2. 지역별/진도별 성능 검증 수행")
    print(f"   3. 기존 98% → 99%+ 성능 향상 목표")

if __name__ == "__main__":
    main()
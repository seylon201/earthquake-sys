import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from urllib.parse import urlencode

class CESMDDataCollector:
    def __init__(self):
        self.base_url = "https://strongmotioncenter.org/wserv"
        self.session = requests.Session()
        self.collected_data = []
        
    def query_events(self, 
                    minmag=3.0,      # 최소 진도
                    maxmag=7.0,      # 최대 진도  
                    start_date=None, # 시작 날짜 (YYYY-MM-DD)
                    end_date=None,   # 종료 날짜
                    faulttype=None,  # 단층 타입: SS(Strike-slip), NM(Normal), RV(Reverse)
                    format_type='json'):
        """
        지진 이벤트 메타데이터 조회 (날짜 파라미터 제거)
        """
        
        # 기본 파라미터만 사용 (문제가 되는 파라미터들 제거)
        params = {
            'minmag': minmag,
            'maxmag': maxmag,
            'format': format_type,
            'nodata': '404'
        }
        
        # 날짜 파라미터는 제거 (API에서 지원하지 않음)
        # if start_date:
        #     params['starttime'] = start_date
        # if end_date:
        #     params['endtime'] = end_date
        
        # 단층 타입만 안전하게 추가
        if faulttype and faulttype in ['SS', 'NM', 'RV']:
            params['faulttype'] = faulttype
            
        url = f"{self.base_url}/events/query"
        
        try:
            print(f"이벤트 조회 중... (진도 {minmag}-{maxmag})")
            print(f"요청 URL: {url}")
            print(f"파라미터: {params}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            # 상세한 에러 정보 출력
            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                return []
            
            if format_type == 'json':
                try:
                    data = response.json()
                    
                    # FeatureCollection 형태 처리
                    if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                        features = data.get('features', [])
                        print(f"✅ FeatureCollection에서 {len(features)}개 이벤트 발견")
                        
                        # features를 events 형태로 변환
                        events = []
                        for feature in features:
                            properties = feature.get('properties', {})
                            geometry = feature.get('geometry', {})
                            
                            # 좌표 정보 추가 (경도, 위도, 깊이)
                            if geometry.get('coordinates'):
                                coords = geometry['coordinates']
                                properties['longitude'] = coords[0] if len(coords) > 0 else None
                                properties['latitude'] = coords[1] if len(coords) > 1 else None
                                properties['depth'] = coords[2] if len(coords) > 2 else None
                            
                            events.append(properties)
                        
                        return events
                        
                    # 기본 events 형태 처리
                    elif isinstance(data, dict) and 'events' in data:
                        events = data.get('events', [])
                        print(f"✅ {len(events)}개 이벤트 발견")
                        return events
                    else:
                        print(f"❌ 예상과 다른 JSON 구조: {type(data)}")
                        print(f"데이터 키들: {list(data.keys()) if isinstance(data, dict) else 'Not dict'}")
                        return []
                        
                except json.JSONDecodeError:
                    print(f"❌ JSON 파싱 실패. 응답 내용: {response.text[:200]}")
                    return []
            else:
                return response.text
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 이벤트 조회 실패: {e}")
            print(f"전체 URL: {response.url if 'response' in locals() else 'URL 생성 실패'}")
            return []

    def query_stations(self, 
                      event_id=None,
                      network=None,
                      station=None,
                      format_type='json'):
        """
        관측소 메타데이터 조회
        """
        
        params = {
            'format': format_type,
            'nodata': '404'
        }
        
        if event_id:
            params['eventid'] = event_id
        if network:
            params['network'] = network
        if station:
            params['station'] = station
            
        url = f"{self.base_url}/stations/query"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if format_type == 'json':
                data = response.json()
                return data.get('stations', [])
            else:
                return response.text
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 관측소 조회 실패: {e}")
            return []

    def query_records(self, 
                     event_id=None,
                     station=None,
                     network=None,
                     format_type='json'):
        """
        강진동 기록 메타데이터 조회 (format 파라미터 필수)
        """
        
        # format 파라미터는 필수
        params = {
            'format': format_type,
            'nodata': '404'
        }
        
        if event_id:
            params['eventid'] = event_id
        if station:
            params['station'] = station
        if network:
            params['network'] = network
            
        url = f"{self.base_url}/records/query"
        
        try:
            print(f"    🔍 기록 조회: eventid={event_id}, network={network}")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"    ❌ 기록 조회 실패 (HTTP {response.status_code}): {response.text[:100]}")
                return []
            
            if format_type == 'json':
                try:
                    data = response.json()
                    
                    # FeatureCollection 형태 처리
                    if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                        features = data.get('features', [])
                        print(f"    ✅ {len(features)}개 기록 발견")
                        
                        # features를 records 형태로 변환
                        records = []
                        for feature in features:
                            properties = feature.get('properties', {})
                            geometry = feature.get('geometry', {})
                            
                            # 좌표 정보 추가
                            if geometry.get('coordinates'):
                                coords = geometry['coordinates']
                                properties['longitude'] = coords[0] if len(coords) > 0 else None
                                properties['latitude'] = coords[1] if len(coords) > 1 else None
                                properties['depth'] = coords[2] if len(coords) > 2 else None
                            
                            records.append(properties)
                        
                        return records
                        
                    # 기본 records 형태 처리
                    elif isinstance(data, dict) and 'records' in data:
                        records = data.get('records', [])
                        print(f"    ✅ {len(records)}개 기록 발견")
                        return records
                    else:
                        print(f"    ℹ️ 기록 없음 (응답 구조: {type(data)})")
                        return []
                        
                except json.JSONDecodeError:
                    print(f"    ❌ 기록 JSON 파싱 실패")
                    return []
            else:
                return response.text
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ 기록 조회 요청 실패: {str(e)[:100]}")
            return []

    def download_record_data(self, record_id, output_dir="cesmd_data"):
        """
        실제 강진동 데이터 다운로드
        """
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        url = f"{self.base_url}/records/download/{record_id}"
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            filename = f"{output_dir}/record_{record_id}.txt"
            with open(filename, 'wb') as f:
                f.write(response.content)
                
            print(f"✅ 다운로드 완료: {filename}")
            return filename
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 다운로드 실패 (ID: {record_id}): {e}")
            return None

    def bulk_data_collection(self, 
                           mag_ranges=[(3.0, 4.0), (4.0, 5.0), (5.0, 6.0)],
                           years_back=5,
                           max_events_per_range=100,
                           download_records=False):
        """
        대량 데이터 수집 프로세스 (실제 eventid 추출)
        """
        
        print("🚀 CESMD 대량 데이터 수집 시작")
        print("="*50)
        
        all_events = []
        all_records = []
        
        for min_mag, max_mag in mag_ranges:
            print(f"\n📊 진도 {min_mag}-{max_mag} 범위 처리 중...")
            
            # 이벤트 조회 (날짜 제한 없이)
            events = self.query_events(
                minmag=min_mag,
                maxmag=max_mag
            )
            
            # 원하는 개수만큼 제한 (Python에서 처리)
            limited_events = events[:max_events_per_range] if events else []
            
            print(f"📋 수집 대상: {len(limited_events)}개 이벤트 (전체 {len(events)}개 중)")
            
            for i, event in enumerate(limited_events):
                # 실제 eventid 추출 시도
                event_id = None
                possible_id_fields = ['eventid', 'id', 'event_id', 'net_id', 'code']
                
                for field in possible_id_fields:
                    if event.get(field):
                        event_id = event.get(field)
                        break
                
                # eventid가 없으면 다른 방법 시도
                if not event_id:
                    # detail 필드에서 ID 추출 시도
                    detail = event.get('detail', '')
                    net = event.get('net', '')
                    time_str = event.get('time', '')
                    
                    if net and time_str:
                        # 네트워크코드 + 시간으로 임시 ID 생성
                        event_id = f"{net}_{time_str.replace(':', '').replace('-', '').replace(' ', '_')}"
                    else:
                        event_id = f"cesmd_event_{i}_{min_mag}_{max_mag}"
                
                mag = event.get('mag', 0)
                place = event.get('place', 'Unknown')
                
                print(f"  📍 이벤트 {i+1}/{len(limited_events)}: ID={event_id}, 진도={mag}, 위치={place[:30]}")
                
                # 해당 이벤트의 기록 조회 (안전하게)
                try:
                    records = self.query_records(event_id=event_id)
                    
                    # records가 비어있으면 다른 방법으로 시도
                    if not records and event.get('net'):
                        # 네트워크 기반 조회 시도
                        print(f"    🔄 네트워크 {event.get('net')} 기반 재시도...")
                        records = self.query_records(network=event.get('net'))
                    
                    for record in records:
                        record['event_magnitude'] = mag
                        record['event_id'] = event_id
                        record['event_place'] = place
                        all_records.append(record)
                        
                        # 실제 데이터 다운로드 (옵션)
                        if download_records and record.get('recordid'):
                            self.download_record_data(record['recordid'])
                            time.sleep(0.5)  # API 부하 방지
                    
                    print(f"    ➡️ {len(records)}개 기록 발견")
                    
                    if not records:
                        print(f"    ⚠️ 이 이벤트에는 강진동 기록이 없습니다")
                    
                except Exception as e:
                    print(f"    ❌ 기록 조회 실패: {str(e)[:100]}")
                    records = []
                
                all_events.append(event)
                time.sleep(0.3)  # API 부하 방지 (조금 더 길게)
        
        # 결과 저장
        self.save_results(all_events, all_records)
        
        print(f"\n✅ 수집 완료!")
        print(f"   - 총 이벤트: {len(all_events)}개")
        print(f"   - 총 기록: {len(all_records)}개")
        
        # 수집 결과 요약
        if all_events:
            print(f"\n📊 수집 결과 요약:")
            mags = [e.get('mag', 0) for e in all_events if e.get('mag')]
            if mags:
                print(f"   진도 범위: {min(mags):.1f} ~ {max(mags):.1f}")
            
            countries = [e.get('country', 'Unknown') for e in all_events]
            country_counts = {}
            for country in countries:
                country_counts[country] = country_counts.get(country, 0) + 1
            print(f"   주요 국가: {dict(list(country_counts.items())[:3])}")
        
        return all_events, all_records

    def save_results(self, events, records):
        """
        수집 결과를 파일로 저장
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 이벤트 데이터 저장
        if events:
            events_df = pd.DataFrame(events)
            events_file = f"cesmd_events_{timestamp}.csv"
            events_df.to_csv(events_file, index=False)
            print(f"📁 이벤트 데이터 저장: {events_file}")
        
        # 기록 데이터 저장
        if records:
            records_df = pd.DataFrame(records)
            records_file = f"cesmd_records_{timestamp}.csv"
            records_df.to_csv(records_file, index=False)
            print(f"📁 기록 데이터 저장: {records_file}")
        
        # JSON 백업도 저장
        backup_data = {
            'events': events,
            'records': records,
            'collection_time': timestamp
        }
        
        json_file = f"cesmd_backup_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        print(f"📁 JSON 백업 저장: {json_file}")

    def analyze_collected_data(self, records_df):
        """
        수집된 데이터 분석
        """
        
        print("\n📈 수집 데이터 분석 결과")
        print("="*30)
        
        if len(records_df) == 0:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        # 진도별 분포
        if 'event_magnitude' in records_df.columns:
            mag_dist = records_df['event_magnitude'].value_counts().sort_index()
            print(f"🎯 진도별 기록 수:")
            for mag, count in mag_dist.items():
                print(f"   진도 {mag}: {count}개")
        
        # 네트워크별 분포
        if 'network' in records_df.columns:
            network_dist = records_df['network'].value_counts()
            print(f"\n🌐 네트워크별 기록 수:")
            for network, count in network_dist.head().items():
                print(f"   {network}: {count}개")
        
        # 샘플링 레이트 분포
        if 'sampling_rate' in records_df.columns:
            sr_dist = records_df['sampling_rate'].value_counts()
            print(f"\n⏱️ 샘플링 레이트 분포:")
            for sr, count in sr_dist.head().items():
                print(f"   {sr}Hz: {count}개")

    def test_api_connection(self):
        """
        API 연결 테스트 및 기본 확인
        """
        print("🔍 CESMD API 연결 테스트 시작")
        print("="*40)
        
        # 1. 가장 기본적인 요청 테스트 (limit 파라미터 제거)
        basic_url = f"{self.base_url}/events/query"
        basic_params = {
            'format': 'json',
            'minmag': 5.0,
            'maxmag': 6.0
        }
        
        try:
            print("1️⃣ 기본 연결 테스트...")
            response = self.session.get(basic_url, params=basic_params, timeout=15)
            print(f"   상태코드: {response.status_code}")
            print(f"   응답 길이: {len(response.text)} bytes")
            
            if response.status_code == 200:
                print("   ✅ 기본 연결 성공!")
                try:
                    data = response.json()
                    
                    # FeatureCollection 형태 처리
                    if isinstance(data, dict):
                        if data.get('type') == 'FeatureCollection':
                            features = data.get('features', [])
                            print(f"   ✅ FeatureCollection 형태! {len(features)}개 이벤트")
                            
                            # features를 events 형태로 변환
                            events = []
                            for feature in features:
                                properties = feature.get('properties', {})
                                geometry = feature.get('geometry', {})
                                
                                # 좌표 정보 추가
                                if geometry.get('coordinates'):
                                    coords = geometry['coordinates']
                                    properties['longitude'] = coords[0] if len(coords) > 0 else None
                                    properties['latitude'] = coords[1] if len(coords) > 1 else None
                                    properties['depth'] = coords[2] if len(coords) > 2 else None
                                
                                events.append(properties)
                            
                            print(f"   ✅ {len(events)}개 이벤트로 변환 완료")
                            return True
                        elif 'events' in data:
                            print(f"   ✅ 기본 events 형태! {len(data['events'])}개")
                            return True
                        else:
                            print(f"   ⚠️ 예상과 다른 JSON 구조")
                            print(f"   키들: {list(data.keys())}")
                            print(f"   타입: {data.get('type', 'Unknown')}")
                    else:
                        print(f"   ⚠️ 예상과 다른 데이터 타입: {type(data)}")
                        
                except Exception as e:
                    print(f"   ❌ JSON 파싱 실패: {e}")
                    print(f"   원본 응답: {response.text[:200]}...")
            else:
                print(f"   ❌ 연결 실패: {response.status_code}")
                print(f"   에러 내용: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ 연결 오류: {e}")
            
        return False

    def simple_event_query(self, mag_min=5.0, mag_max=6.0):
        """
        단순화된 이벤트 조회 (FeatureCollection 형태 처리)
        """
        print(f"\n🎯 단순 이벤트 조회 (진도 {mag_min}-{mag_max})")
        
        url = f"{self.base_url}/events/query"
        params = {
            'minmag': mag_min,
            'maxmag': mag_max,
            'format': 'json'
        }
        
        print(f"📍 요청 정보:")
        print(f"   URL: {url}")
        print(f"   파라미터: {params}")
        
        try:
            response = self.session.get(url, params=params, timeout=20)
            print(f"📡 응답 정보:")
            print(f"   상태코드: {response.status_code}")
            print(f"   응답 크기: {len(response.text)} bytes")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # FeatureCollection 형태 처리
                    if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                        features = data.get('features', [])
                        print(f"✅ FeatureCollection에서 {len(features)}개 이벤트 발견!")
                        
                        # features를 events 형태로 변환
                        events = []
                        for feature in features:
                            properties = feature.get('properties', {})
                            geometry = feature.get('geometry', {})
                            
                            # 좌표 정보 추가
                            if geometry.get('coordinates'):
                                coords = geometry['coordinates']
                                properties['longitude'] = coords[0] if len(coords) > 0 else None
                                properties['latitude'] = coords[1] if len(coords) > 1 else None
                                properties['depth'] = coords[2] if len(coords) > 2 else None
                            
                            events.append(properties)
                        
                        # 첫 번째 이벤트 샘플 출력
                        if events:
                            sample = events[0]
                            print(f"📋 샘플 이벤트 (첫 번째):")
                            for key, value in list(sample.items())[:8]:  # 더 많은 정보 출력
                                print(f"   {key}: {value}")
                            
                            print(f"\n📊 데이터 요약:")
                            print(f"   총 이벤트 수: {len(events)}")
                            
                            # 진도 분포 확인
                            magnitudes = [e.get('mag', 0) for e in events if e.get('mag')]
                            if magnitudes:
                                print(f"   진도 범위: {min(magnitudes):.1f} ~ {max(magnitudes):.1f}")
                        
                        return events
                        
                    # 기본 events 형태 처리
                    elif isinstance(data, dict) and 'events' in data:
                        events = data.get('events', [])
                        print(f"✅ 기본 형태에서 {len(events)}개 이벤트 발견!")
                        return events
                        
                    else:
                        print(f"❌ 예상과 다른 JSON 구조")
                        print(f"   데이터 타입: {type(data)}")
                        if isinstance(data, dict):
                            print(f"   키들: {list(data.keys())}")
                        return []
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 실패: {e}")
                    print(f"원본 응답: {response.text[:300]}...")
            else:
                print(f"❌ 요청 실패: {response.status_code}")
                print(f"에러 응답: {response.text[:300]}...")
                
        except Exception as e:
            print(f"❌ 요청 오류: {e}")
            
        return []
        """
        수집 결과를 파일로 저장
        """
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 이벤트 데이터 저장
        if events:
            events_df = pd.DataFrame(events)
            events_file = f"cesmd_events_{timestamp}.csv"
            events_df.to_csv(events_file, index=False)
            print(f"📁 이벤트 데이터 저장: {events_file}")
        
        # 기록 데이터 저장
        if records:
            records_df = pd.DataFrame(records)
            records_file = f"cesmd_records_{timestamp}.csv"
            records_df.to_csv(records_file, index=False)
            print(f"📁 기록 데이터 저장: {records_file}")
        
        # JSON 백업도 저장
        backup_data = {
            'events': events,
            'records': records,
            'collection_time': timestamp
        }
        
        json_file = f"cesmd_backup_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(backup_data, f, indent=2)
        print(f"📁 JSON 백업 저장: {json_file}")

    def analyze_collected_data(self, records_df):
        """
        수집된 데이터 분석
        """
        
        print("\n📈 수집 데이터 분석 결과")
        print("="*30)
        
        if len(records_df) == 0:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        # 진도별 분포
        if 'event_magnitude' in records_df.columns:
            mag_dist = records_df['event_magnitude'].value_counts().sort_index()
            print(f"🎯 진도별 기록 수:")
            for mag, count in mag_dist.items():
                print(f"   진도 {mag}: {count}개")
        
        # 네트워크별 분포
        if 'network' in records_df.columns:
            network_dist = records_df['network'].value_counts()
            print(f"\n🌐 네트워크별 기록 수:")
            for network, count in network_dist.head().items():
                print(f"   {network}: {count}개")
        
        # 샘플링 레이트 분포
        if 'sampling_rate' in records_df.columns:
            sr_dist = records_df['sampling_rate'].value_counts()
            print(f"\n⏱️ 샘플링 레이트 분포:")
            for sr, count in sr_dist.head().items():
                print(f"   {sr}Hz: {count}개")


    def explore_event_structure(self, events, max_samples=3):
        """
        이벤트 데이터 구조 탐색 (실제 eventid 찾기)
        """
        print(f"\n🔍 이벤트 데이터 구조 분석 (샘플 {max_samples}개)")
        print("="*40)
        
        for i, event in enumerate(events[:max_samples]):
            print(f"\n📋 이벤트 {i+1} 전체 구조:")
            for key, value in event.items():
                print(f"   {key}: {value}")
            
            # 가능한 ID 필드들 확인
            possible_ids = []
            for key, value in event.items():
                if 'id' in key.lower() or 'code' in key.lower():
                    possible_ids.append(f"{key}: {value}")
            
            if possible_ids:
                print(f"🔑 가능한 ID 필드들:")
                for pid in possible_ids:
                    print(f"   {pid}")
            else:
                print("❌ ID 관련 필드를 찾을 수 없습니다")
        
        return events
if __name__ == "__main__":
    # 데이터 수집기 초기화
    collector = CESMDDataCollector()
    
    # 1단계: API 연결 테스트
    if collector.test_api_connection():
        print("\n" + "="*50)
        
        # 2단계: 단순 조회 테스트 (limit 제거)
        events = collector.simple_event_query(mag_min=5.0, mag_max=6.0)
        
        if events:
            print(f"\n🎉 API 작동 확인! {len(events)}개 이벤트 조회 성공")
            
            # 3단계: 점진적 데이터 수집
            print("\n" + "="*50)
            print("📊 실제 데이터 수집 시작...")
            
            all_events, all_records = collector.bulk_data_collection(
                mag_ranges=[(5.0, 6.0)],          # 우선 높은 진도만
                years_back=1,                      # 최근 1년만
                max_events_per_range=10,           # 10개만 (Python에서 제한)
                download_records=False             # 메타데이터만
            )
            
            if all_events:
                print(f"\n✅ 최종 수집 성공!")
                print(f"   이벤트: {len(all_events)}개")
                print(f"   기록: {len(all_records)}개")
            else:
                print("\n❌ 데이터 수집 실패")
        else:
            print("\n❌ 단순 조회도 실패")
    else:
        print("\n❌ API 연결 실패 - 네트워크나 서비스 상태를 확인하세요")
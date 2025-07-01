#!/usr/bin/env python3
"""
NEW2 3클래스 수정 계획 - industry 데이터 제외 분석
편향 문제 해결을 위한 새로운 클래스 구성
"""

import os
import json
from datetime import datetime

def analyze_current_data_files():
    """현재 NEW2 데이터 파일 분석"""
    
    print("📊 === NEW2 데이터 파일 현황 분석 ===\n")
    
    # NEW2 데이터 파일 목록
    data_files = {
        'kor_quake': ('new2_kor_quake_data.npy', 'new2_kor_quake_labels.npy'),
        'jpn_quake': ('new2_jpn_quake_data.npy', 'new2_jpn_quake_labels.npy'),
        'industry': ('new2_industry_data.npy', 'new2_industry_labels.npy'),
        'motor': ('new2_motor_data.npy', 'new2_motor_labels.npy'),
        'live': ('new2_live_data.npy', 'new2_live_labels.npy'),
        'irregular': ('new2_irregular_data.npy', 'new2_irregular_labels.npy')
    }
    
    file_status = {}
    
    for source, (data_file, label_file) in data_files.items():
        data_exists = os.path.exists(data_file)
        label_exists = os.path.exists(label_file)
        
        status = "✅" if (data_exists and label_exists) else "❌"
        file_status[source] = {
            'data_file': data_file,
            'label_file': label_file,
            'data_exists': data_exists,
            'label_exists': label_exists,
            'both_exist': data_exists and label_exists
        }
        
        if data_exists and label_exists:
            data_size = os.path.getsize(data_file) / (1024*1024)  # MB
            label_size = os.path.getsize(label_file) / (1024*1024)  # MB
            print(f"{status} {source}: 데이터({data_size:.1f}MB) + 라벨({label_size:.1f}MB)")
        else:
            print(f"{status} {source}: 파일 누락 (data: {data_exists}, label: {label_exists})")
    
    return file_status

def plan_revised_3class():
    """수정된 3클래스 구성 계획"""
    
    print("\n🎯 === 수정된 NEW2 3클래스 구성 계획 ===\n")
    
    # 기존 문제점
    print("❌ 기존 문제점:")
    print("   - industry 데이터로 인한 클래스 1(규칙적산업진동) 편향")
    print("   - 모든 실시간 분석이 99.7% 신뢰도로 규칙적산업진동 예측")
    print("   - 지진과 불규칙생활진동 클래스 인식률 저하")
    
    # 수정 방안
    print("\n✅ 수정 방안:")
    print("   - industry 데이터 완전 제외")
    print("   - 규칙적산업진동은 motor 데이터만 사용")
    print("   - 더 균형잡힌 3클래스 구성")
    
    # 새로운 클래스 구성
    revised_mapping = {
        0: {
            'name': '지진',
            'sources': ['kor_quake', 'jpn_quake'],
            'description': '한국 지진 + 일본 지진 데이터'
        },
        1: {
            'name': '규칙적산업진동',
            'sources': ['motor'],
            'description': '모터 진동 데이터만 (industry 제외)'
        },
        2: {
            'name': '불규칙생활진동',
            'sources': ['live', 'irregular'],
            'description': '생활 진동 + 불규칙 진동 데이터'
        }
    }
    
    print("\n📊 새로운 3클래스 구성:")
    for class_id, info in revised_mapping.items():
        icon = '🔴' if class_id == 0 else '🟠' if class_id == 1 else '🟢'
        print(f"   {icon} 클래스 {class_id}: {info['name']}")
        print(f"      소스: {', '.join(info['sources'])}")
        print(f"      설명: {info['description']}")
    
    return revised_mapping

def estimate_dataset_balance(file_status, revised_mapping):
    """수정된 데이터셋 균형성 추정"""
    
    print("\n⚖️ === 수정된 데이터셋 균형성 추정 ===\n")
    
    # 기존 데이터셋의 대략적인 샘플 수 (문서 기반)
    estimated_counts = {
        'kor_quake': 2308,    # 한국 지진
        'jpn_quake': 1564,    # 일본 지진
        'industry': 1110,     # 산업 진동 (제외됨)
        'motor': 1604,        # 모터 진동
        'live': 2135,         # 생활 진동
        'irregular': 700      # 불규칙 진동
    }
    
    print("📈 예상 클래스별 샘플 수:")
    
    total_samples = 0
    class_counts = {}
    
    for class_id, info in revised_mapping.items():
        class_total = 0
        sources_detail = []
        
        for source in info['sources']:
            if source in estimated_counts:
                count = estimated_counts[source]
                class_total += count
                sources_detail.append(f"{source}({count:,}개)")
            else:
                sources_detail.append(f"{source}(파일없음)")
        
        class_counts[class_id] = class_total
        total_samples += class_total
        
        icon = '🔴' if class_id == 0 else '🟠' if class_id == 1 else '🟢'
        print(f"   {icon} 클래스 {class_id} ({info['name']}): {class_total:,}개")
        print(f"      └── {' + '.join(sources_detail)}")
    
    print(f"\n📊 전체 데이터셋 요약:")
    print(f"   총 샘플: {total_samples:,}개 (industry {estimated_counts['industry']:,}개 제외)")
    print(f"   제외된 샘플: {estimated_counts['industry']:,}개")
    
    # 균형성 분석
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        balance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n⚖️ 클래스 균형성:")
        print(f"   최소: {min_count:,}개, 최대: {max_count:,}개")
        print(f"   균형 비율: {balance_ratio:.1f}:1")
        
        if balance_ratio <= 2.0:
            print("   ✅ 양호한 균형 (2:1 이하)")
        elif balance_ratio <= 3.0:
            print("   ⚠️ 보통 균형 (3:1 이하)")
        else:
            print("   ❌ 불균형 (균형 조정 필요)")
    
    return class_counts, total_samples

def create_implementation_plan():
    """구현 계획 생성"""
    
    print("\n🚀 === 구현 계획 ===\n")
    
    steps = [
        {
            'step': 1,
            'title': '수정된 3클래스 데이터셋 생성',
            'tasks': [
                'industry 데이터 제외한 3class_balanced_dataset.py 실행',
                '새로운 균형 데이터셋 생성 (motor만 클래스 1로)',
                '6:2:2 분할로 훈련/검증/테스트 세트 생성'
            ]
        },
        {
            'step': 2,
            'title': 'NEW2 모델 재훈련',
            'tasks': [
                'train_new2_convlstm.py로 새 데이터셋 훈련',
                '성능 지표 분석 및 클래스별 정확도 확인',
                '혼동행렬 분석으로 편향 해결 여부 검증'
            ]
        },
        {
            'step': 3,
            'title': '실시간 시스템 업데이트',
            'tasks': [
                '새 모델로 influx_new2_realtime.py 테스트',
                '다양한 진동 유형에 대한 분류 정확도 검증',
                '오경보율 감소 효과 측정'
            ]
        },
        {
            'step': 4,
            'title': '성능 비교 및 문서화',
            'tasks': [
                '기존 vs 수정된 모델 성능 비교',
                '실제 환경 테스트 결과 업데이트',
                'PROJECT_PLAN.md 업데이트'
            ]
        }
    ]
    
    for step_info in steps:
        print(f"📋 Step {step_info['step']}: {step_info['title']}")
        for task in step_info['tasks']:
            print(f"   ✓ {task}")
        print()
    
    return steps

def save_revision_plan(file_status, revised_mapping, class_counts, steps):
    """수정 계획 저장"""
    
    revision_plan = {
        'timestamp': datetime.now().isoformat(),
        'revision_info': {
            'title': 'NEW2 3클래스 수정 계획',
            'version': 'v2.1',
            'reason': '규칙적산업진동 클래스 편향 문제 해결',
            'excluded_data': 'industry 데이터 (1,110개 샘플)'
        },
        'file_analysis': file_status,
        'revised_class_mapping': revised_mapping,
        'estimated_balance': {
            'class_counts': class_counts,
            'total_samples': sum(class_counts.values()) if class_counts else 0
        },
        'implementation_steps': steps,
        'expected_improvements': {
            'reduced_bias': '규칙적산업진동 클래스 편향 해결',
            'better_balance': '3클래스 간 더 나은 균형',
            'improved_accuracy': '지진 및 불규칙생활진동 인식률 향상',
            'reduced_false_alarms': '오경보율 감소 기대'
        }
    }
    
    with open('NEW2_revision_plan.json', 'w', encoding='utf-8') as f:
        json.dump(revision_plan, f, ensure_ascii=False, indent=2)
    
    print("💾 수정 계획 저장: NEW2_revision_plan.json")
    
    return revision_plan

def main():
    """메인 실행"""
    
    print("🔧 NEW2 3클래스 수정 계획 수립\n")
    print("=" * 60)
    
    try:
        # 1. 현재 데이터 파일 분석
        file_status = analyze_current_data_files()
        
        # 2. 수정된 3클래스 구성 계획
        revised_mapping = plan_revised_3class()
        
        # 3. 데이터셋 균형성 추정
        class_counts, total_samples = estimate_dataset_balance(file_status, revised_mapping)
        
        # 4. 구현 계획 생성
        steps = create_implementation_plan()
        
        # 5. 계획 저장
        plan = save_revision_plan(file_status, revised_mapping, class_counts, steps)
        
        print(f"\n🎉 NEW2 3클래스 수정 계획 완료!")
        print(f"📊 예상 개선 효과:")
        print(f"   🎯 편향 해결: industry 데이터 제외")
        print(f"   ⚖️ 균형 개선: 3클래스 간 더 나은 분포")
        print(f"   🔍 정확도 향상: 지진 탐지 성능 개선 기대")
        
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅' if success else '❌'} 계획 수립 {'완료' if success else '실패'}!")
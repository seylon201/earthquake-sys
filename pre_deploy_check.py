#!/usr/bin/env python3
"""
지진 조기경보 시스템 배포 사전 검증 도구
배포 전 필수 조건들을 자동으로 검사합니다.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import importlib.util

class DeploymentChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.current_dir = Path.cwd()
        
    def log_success(self, message):
        print(f"✅ {message}")
        self.checks_passed += 1
        
    def log_error(self, message):
        print(f"❌ {message}")
        self.checks_failed += 1
        
    def log_warning(self, message):
        print(f"⚠️  {message}")
        self.warnings += 1
        
    def log_info(self, message):
        print(f"ℹ️  {message}")
    
    def check_python_version(self):
        """Python 버전 확인"""
        self.log_info("Python 버전 확인 중...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.log_success(f"Python {version.major}.{version.minor}.{version.micro} 호환됨")
        else:
            self.log_error(f"Python 3.8+ 필요 (현재: {version.major}.{version.minor}.{version.micro})")
    
    def check_required_files(self):
        """필수 파일 존재 확인"""
        self.log_info("필수 파일 확인 중...")
        
        # 핵심 파일들
        core_files = [
            "new2_streamlit_dashboard.py",
            "dashboard_utils.py", 
            "retraining_manager.py",
            "chart_renderer.py"
        ]
        
        # 설정 파일들  
        config_files = [
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml"
        ]
        
        # 모델 파일들
        model_files = [
            "new2_convlstm_3class_best.h5",
            "new2_memory_optimized_best.h5"
        ]
        
        # 핵심 파일 검사
        for file in core_files:
            if (self.current_dir / file).exists():
                self.log_success(f"핵심 파일 발견: {file}")
            else:
                self.log_error(f"핵심 파일 누락: {file}")
        
        # 설정 파일 검사
        for file in config_files:
            if (self.current_dir / file).exists():
                self.log_success(f"설정 파일 발견: {file}")
            else:
                self.log_warning(f"설정 파일 누락: {file}")
        
        # 모델 파일 검사
        model_found = False
        for file in model_files:
            if (self.current_dir / file).exists():
                self.log_success(f"모델 파일 발견: {file}")
                model_found = True
                
        if not model_found:
            self.log_error("모델 파일이 없습니다. 최소 하나의 .h5 모델 파일이 필요합니다.")
    
    def check_python_packages(self):
        """Python 패키지 설치 확인"""
        self.log_info("Python 패키지 확인 중...")
        
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'tensorflow', 
            'plotly', 'scikit-learn', 'matplotlib'
        ]
        
        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    self.log_success(f"패키지 설치됨: {package}")
                else:
                    self.log_warning(f"패키지 누락: {package}")
            except ImportError:
                self.log_warning(f"패키지 누락: {package}")
    
    def check_directories(self):
        """필요한 디렉토리 구조 확인"""
        self.log_info("디렉토리 구조 확인 중...")
        
        required_dirs = [
            "influxLogs",
            "influxLogs/base", 
            "influxLogs/new2_analysis",
            "influxLogs/alerts",
            "data",
            "retrained_models"
        ]
        
        for dir_path in required_dirs:
            full_path = self.current_dir / dir_path
            if full_path.exists():
                self.log_success(f"디렉토리 존재: {dir_path}")
            else:
                self.log_warning(f"디렉토리 누락: {dir_path} (자동 생성 예정)")
                # 디렉토리 자동 생성
                full_path.mkdir(parents=True, exist_ok=True)
                self.log_success(f"디렉토리 생성됨: {dir_path}")
    
    def check_docker_availability(self):
        """Docker 및 Docker Compose 설치 확인"""
        self.log_info("Docker 환경 확인 중...")
        
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log_success(f"Docker 설치됨: {version}")
            else:
                self.log_warning("Docker가 설치되지 않았습니다.")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_warning("Docker가 설치되지 않았습니다.")
        
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.log_success(f"Docker Compose 설치됨: {version}")
            else:
                self.log_warning("Docker Compose가 설치되지 않았습니다.")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_warning("Docker Compose가 설치되지 않았습니다.")
    
    def check_ports(self):
        """포트 사용 가능성 확인"""
        self.log_info("포트 확인 중...")
        
        import socket
        
        ports_to_check = [8501, 8086]  # Streamlit, InfluxDB
        
        for port in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                self.log_warning(f"포트 {port}가 이미 사용 중입니다.")
            else:
                self.log_success(f"포트 {port} 사용 가능")
    
    def check_disk_space(self):
        """디스크 공간 확인"""
        self.log_info("디스크 공간 확인 중...")
        
        import shutil
        
        free_space = shutil.disk_usage(self.current_dir).free
        free_gb = free_space / (1024**3)
        
        if free_gb >= 5.0:
            self.log_success(f"충분한 디스크 공간: {free_gb:.1f}GB")
        elif free_gb >= 2.0:
            self.log_warning(f"디스크 공간 부족: {free_gb:.1f}GB (최소 5GB 권장)")
        else:
            self.log_error(f"심각한 디스크 공간 부족: {free_gb:.1f}GB")
    
    def generate_deployment_report(self):
        """배포 준비 리포트 생성"""
        self.log_info("배포 준비 리포트 생성 중...")
        
        report = {
            "timestamp": str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip()),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "total_checks": self.checks_passed + self.checks_failed,
            "deployment_ready": self.checks_failed == 0
        }
        
        with open(self.current_dir / "deployment_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.log_success("배포 리포트 저장됨: deployment_report.json")
        return report
    
    def run_all_checks(self):
        """모든 검사 실행"""
        print("🔍 지진 조기경보 시스템 배포 사전 검증 시작")
        print("=" * 60)
        
        self.check_python_version()
        print()
        
        self.check_required_files()
        print()
        
        self.check_python_packages()
        print()
        
        self.check_directories()
        print()
        
        self.check_docker_availability()
        print()
        
        self.check_ports()
        print()
        
        self.check_disk_space()
        print()
        
        report = self.generate_deployment_report()
        
        print("=" * 60)
        print("📊 검증 결과 요약")
        print(f"✅ 통과: {self.checks_passed}개")
        print(f"❌ 실패: {self.checks_failed}개") 
        print(f"⚠️  경고: {self.warnings}개")
        print()
        
        if report["deployment_ready"]:
            print("🎉 배포 준비 완료! 다음 명령어로 배포를 시작하세요:")
            print("   Docker 배포: ./deploy.sh dev")
            print("   간단 배포: ./deploy_simple.sh")
            print("   Windows: deploy_windows.bat")
        else:
            print("⚠️  배포 전 문제를 해결하세요:")
            print("   - 실패한 검사 항목들을 확인하고 수정하세요")
            print("   - 필요한 파일들을 준비하세요")
        
        return report["deployment_ready"]

if __name__ == "__main__":
    checker = DeploymentChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파이프 테스트 스크립트
"""

import subprocess
import sys

def test_pipe_input():
    """파이프 입력을 통한 추천 시스템 테스트"""
    
    # 테스트 입력 준비
    test_inputs = [
        "Inception",      # 영화 제목
        "0.8",           # Plot weight
        "0.15",          # Flow weight 
        "0.05",          # Genre weight
        "quit"           # 종료
    ]
    
    # 입력을 문자열로 결합 (각 줄 끝에 \n 추가)
    input_data = "\n".join(test_inputs) + "\n"
    
    try:
        # subprocess를 사용해서 파이프 입력 시뮬레이션
        result = subprocess.run(
            [sys.executable, "weighted_movie_finder.py"],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=60
        )
        
        print("=== STDOUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
            
        print(f"=== Return Code: {result.returncode} ===")
        
    except subprocess.TimeoutExpired:
        print("❌ 프로세스가 타임아웃되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    test_pipe_input()
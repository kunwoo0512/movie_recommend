"""
OpenAI API 키 테스트 스크립트 (.env 파일 지원)
"""

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def test_api_key():
    """
    API 키가 제대로 설정되었는지 확인
    """
    print("[키] OpenAI API 키 테스트")
    print("=" * 40)
    
    # 1. 환경변수 확인
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print(f"[성공] 환경변수에서 API 키 발견: {api_key[:10]}...")
    else:
        print("[오류] 환경변수에 OPENAI_API_KEY가 설정되지 않음")
        print("\n[설정] 설정 방법:")
        print("Windows PowerShell:")
        print('$env:OPENAI_API_KEY="your_api_key_here"')
        print("\nWindows CMD:")
        print('set OPENAI_API_KEY=your_api_key_here')
        print("\n또는 코드에서 직접 설정 가능합니다.")
        return False
    
    # 2. OpenAI 라이브러리 확인
    try:
        import openai
        print("[성공] OpenAI 라이브러리 설치됨")
    except ImportError:
        print("[오류] OpenAI 라이브러리 미설치")
        print("설치 명령: pip install openai")
        return False
    
    # 3. API 연결 테스트 (라이브러리가 있을 때만)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # 간단한 테스트 요청
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        print("[성공] API 연결 성공!")
        print(f"응답: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"[오류] API 연결 실패: {str(e)}")
        if "quota" in str(e).lower():
            print("💳 사용량 한도 초과 또는 결제 정보 필요")
        elif "invalid" in str(e).lower():
            print("[키] API 키가 유효하지 않음")
        return False

def setup_api_key_manual():
    """
    수동으로 API 키 설정
    """
    print("\n[설정] 수동 API 키 설정")
    print("=" * 40)
    
    api_key = input("API 키를 입력하세요 (sk-로 시작): ").strip()
    
    if not api_key.startswith('sk-'):
        print("[오류] 올바른 API 키 형식이 아닙니다 (sk-로 시작해야 함)")
        return None
    
    # 환경변수에 설정
    os.environ['OPENAI_API_KEY'] = api_key
    print("[성공] 현재 세션에 API 키 설정 완료")
    
    return api_key

if __name__ == "__main__":
    print("[AI] OpenAI API 설정 도우미")
    print("=" * 50)
    
    # 1차 테스트
    if not test_api_key():
        print("\n[새로고침] 수동 설정을 시도하시겠습니까? (y/n): ", end="")
        choice = input().lower()
        
        if choice == 'y':
            api_key = setup_api_key_manual()
            if api_key:
                print("\n[새로고침] 다시 테스트 중...")
                test_api_key()
        else:
            print("\n[목록] 설정 완료 후 다시 실행해주세요!")
    
    print("\n[반짝] 테스트 완료!")

"""
GPT API를 사용한 완전한 영화 분석 실행기
"""

import json
import os
from dotenv import load_dotenv
from real_gpt_analyzer import GPTMovieAnalyzer

# .env 파일 로드
load_dotenv()

def analyze_all_movies():
    """
    모든 영화에 대해 GPT 분석 실행
    """
    print("🎬 GPT API를 사용한 영화 분석 시작!")
    print("=" * 50)
    
    # 1. 분석기 초기화
    analyzer = GPTMovieAnalyzer()
    
    if not analyzer.client:
        print("❌ GPT API 초기화 실패. .env 파일의 API 키를 확인해주세요.")
        return
    
    # 2. 영화 데이터 로드
    try:
        with open('movie_plots.json', 'r', encoding='utf-8') as f:
            movies = json.load(f)
        print(f"📚 {len(movies)}개 영화 데이터 로드 완료")
    except FileNotFoundError:
        print("❌ movie_plots.json 파일을 찾을 수 없습니다.")
        return
    
    # 3. 각 영화 분석
    results = {}
    
    for title, movie_data in movies.items():
        print(f"\n🎭 '{title}' 분석 중...")
        
        try:
            analysis = analyzer.analyze_movie(
                title=title,
                plot=movie_data['plot'],
                year=movie_data.get('year')
            )
            
            # 결과에 원본 데이터 추가
            results[title] = {
                **movie_data,  # 원본 영화 데이터
                **analysis,    # GPT 분석 결과
                'success': analysis.get('success', False)
            }
            
            print(f"✅ '{title}' 분석 완료!")
            print(f"   흐름 곡선: {analysis['flow_curve']}")
            print(f"   주요 장르: {dict(list(analysis['genres'].items())[:3])}")
            
        except Exception as e:
            print(f"❌ '{title}' 분석 실패: {str(e)}")
            continue
    
    # 4. 결과 저장
    output_file = 'gpt_api_analysis_results.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 분석 결과가 '{output_file}'에 저장되었습니다!")
        
        # 5. 결과 요약
        print(f"\n📊 분석 요약:")
        print(f"   전체 영화: {len(movies)}개")
        print(f"   성공 분석: {sum(1 for r in results.values() if r.get('success', False))}개")
        print(f"   실패/대체: {sum(1 for r in results.values() if not r.get('success', False))}개")
        
        return results
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {str(e)}")
        return None

def compare_with_ollama():
    """
    GPT와 Ollama 결과 비교
    """
    print("\n🔍 GPT vs Ollama 분석 결과 비교")
    print("=" * 50)
    
    # GPT 결과 로드
    try:
        with open('gpt_api_analysis_results.json', 'r', encoding='utf-8') as f:
            gpt_results = json.load(f)
    except FileNotFoundError:
        print("❌ GPT 분석 결과 파일을 찾을 수 없습니다.")
        return
    
    # Ollama 결과 로드
    try:
        with open('ollama_analysis_results.json', 'r', encoding='utf-8') as f:
            ollama_results = json.load(f)
    except FileNotFoundError:
        print("❌ Ollama 분석 결과 파일을 찾을 수 없습니다.")
        return
    
    # 비교
    for title in gpt_results.keys():
        if title in ollama_results:
            print(f"\n🎬 {title}:")
            print(f"   GPT 흐름:    {gpt_results[title]['flow_curve']}")
            print(f"   Ollama 흐름: {ollama_results[title]['flow_curve']}")
            
            # 장르 비교 (상위 3개)
            gpt_genres = sorted(gpt_results[title]['genres'].items(), 
                              key=lambda x: x[1], reverse=True)[:3]
            ollama_genres = sorted(ollama_results[title]['genres'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            
            print(f"   GPT 장르:    {gpt_genres}")
            print(f"   Ollama 장르: {ollama_genres}")

if __name__ == "__main__":
    # 전체 분석 실행
    results = analyze_all_movies()
    
    if results:
        # 비교 분석
        compare_with_ollama()
        
        print(f"\n🎯 다음 단계:")
        print(f"   1. simple_visualizer.py 실행으로 새로운 차트 생성")
        print(f"   2. http://localhost:8080에서 결과 확인")
        print(f"   3. GPT vs Ollama 성능 비교 분석")

"""
순수 줄거리만 사용한 OpenAI 임베딩 생성
"""
import json
import numpy as np
import openai
import os
from dotenv import load_dotenv
import time

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 임베딩 모델 설정
EMBEDDING_MODEL = "text-embedding-3-small"

def get_openai_embedding(text, model=EMBEDDING_MODEL):
    """OpenAI API로 텍스트 임베딩 생성"""
    try:
        # API 호출 제한 고려 (RPM 제한)
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ 임베딩 생성 실패: {e}")
        return None

def create_plot_only_embedding(movie):
    """순수 줄거리만 사용한 임베딩 생성"""
    
    # 줄거리만 추출
    plot = movie.get('plot', '')
    
    if not plot:
        print(f"  ⚠️ 줄거리 없음, 빈 벡터 사용")
        return [0.0] * 1536  # text-embedding-3-small 차원
    
    # 줄거리만 임베딩 생성
    print(f"  📝 줄거리 임베딩 생성 중...")
    embedding = get_openai_embedding(plot)
    
    if embedding:
        # API 제한 고려 (1분에 1000 요청)
        time.sleep(0.1)  # 100ms 대기
        return embedding
    else:
        print(f"  ❌ 임베딩 실패, 빈 벡터 사용")
        return [0.0] * 1536

def main():
    print("📖 순수 줄거리만 사용한 OpenAI 임베딩 생성")
    print("=" * 50)
    
    # API 키 확인
    if not openai.api_key:
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 추가해주세요.")
        return
    
    # 영화 데이터 로드
    with open('movies_dataset.json', 'r', encoding='utf-8') as f:
        movies = json.load(f)
    
    print(f"📊 처리할 영화 수: {len(movies)}")
    print(f"🎯 사용 모델: {EMBEDDING_MODEL}")
    print(f"💰 예상 비용: ${len(movies) * 0.00002:.4f} (줄거리만 × $0.00002/1K tokens)")
    print(f"🎬 방식: 순수 줄거리(plot) 100% 사용")
    
    print(f"\n🚀 임베딩 생성을 시작합니다...")
    
    embeddings = []
    
    for i, movie in enumerate(movies):
        print(f"\n🎬 [{i+1}/{len(movies)}] '{movie.get('title', 'Unknown')}' 처리 중...")
        
        try:
            embedding = create_plot_only_embedding(movie)
            
            embeddings.append({
                'title': movie.get('title', ''),
                'embedding': embedding
            })
            
            print(f"  ✅ 완료")
            
            # 진행상황 저장 (중간에 중단되어도 복구 가능)
            if (i + 1) % 50 == 0:
                with open(f'movie_embeddings_plot_only_backup_{i+1}.json', 'w', encoding='utf-8') as f:
                    json.dump(embeddings, f, ensure_ascii=False, indent=2)
                print(f"  💾 중간 저장 완료 ({i+1}개)")
        
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            # 오류 시 빈 임베딩으로 처리
            embeddings.append({
                'title': movie.get('title', ''),
                'embedding': [0.0] * 1536
            })
    
    # 최종 저장
    with open('movie_embeddings_plot_only.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 순수 줄거리 임베딩 생성 완료!")
    print(f"📁 저장 파일: movie_embeddings_plot_only.json")
    print(f"📊 총 {len(embeddings)}개 영화 임베딩 생성")
    
    # 간단한 통계
    non_empty_embeddings = sum(1 for emb in embeddings if any(x != 0 for x in emb['embedding']))
    print(f"📈 통계:")
    print(f"   - 성공적으로 생성된 임베딩: {non_empty_embeddings}개")
    print(f"   - 빈 임베딩 (줄거리 없음): {len(embeddings) - non_empty_embeddings}개")

if __name__ == "__main__":
    main()

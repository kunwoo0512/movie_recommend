"""
OpenAI 임베딩 모델을 사용한 영화 임베딩 생성
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
EMBEDDING_MODEL = "text-embedding-3-small"  # 또는 "text-embedding-ada-002"

# 가중치 설정
WEIGHTS = {
    'plot': 0.5,
    'flow_curve': 0.2,
    'genres': 0.2,
    'title': 0.08,
    'director': 0.02
}

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

def get_text_for_openai_embedding(movie):
    """OpenAI 임베딩용 텍스트 구성"""
    
    # 장르: 점수 기반으로 설명적 텍스트 생성
    genres = movie.get('genres', {})
    genre_descriptions = []
    
    for genre, score in genres.items():
        if score >= 8:
            genre_descriptions.append(f"strong {genre.replace('_', ' ')} elements")
        elif score >= 6:
            genre_descriptions.append(f"moderate {genre.replace('_', ' ')} elements")
        elif score >= 4:
            genre_descriptions.append(f"minor {genre.replace('_', ' ')} elements")
    
    genre_text = ", ".join(genre_descriptions) if genre_descriptions else "mixed genre"
    
    # 흐름곡선: 서사 패턴으로 변환
    flow_curve = movie.get('flow_curve', [])
    flow_text = ""
    
    if flow_curve:
        max_intensity = max(flow_curve)
        min_intensity = min(flow_curve)
        avg_intensity = sum(flow_curve) / len(flow_curve)
        
        if max_intensity >= 9:
            flow_text += "high-intensity climax, "
        if min_intensity <= 3:
            flow_text += "slow-paced beginning, "
        if avg_intensity >= 7:
            flow_text += "action-packed storyline, "
        elif avg_intensity <= 4:
            flow_text += "contemplative pacing, "
        
        # 흐름 변화 패턴
        rising_segments = sum(1 for i in range(1, len(flow_curve)) if flow_curve[i] > flow_curve[i-1])
        if rising_segments >= len(flow_curve) * 0.6:
            flow_text += "escalating tension"
        else:
            flow_text += "varied pacing"
    
    # 각 요소별 텍스트 구성
    texts = {
        'plot': movie.get('plot', ''),
        'flow_curve': flow_text,
        'genres': genre_text,
        'title': movie.get('title', ''),
        'director': f"directed by {movie.get('director', 'unknown')}"
    }
    
    return texts

def create_weighted_openai_embedding(texts, weights):
    """가중치를 적용한 OpenAI 임베딩 생성"""
    
    embeddings = []
    
    for key, weight in weights.items():
        text = texts[key]
        if text:
            print(f"  📝 {key} 임베딩 생성 중...")
            embedding = get_openai_embedding(text)
            
            if embedding:
                embeddings.append(np.array(embedding) * weight)
                # API 제한 고려 (1분에 1000 요청)
                time.sleep(0.1)  # 100ms 대기
            else:
                print(f"  ⚠️ {key} 임베딩 실패, 0 벡터 사용")
                embeddings.append(np.zeros(1536))  # text-embedding-3-small 차원
        else:
            embeddings.append(np.zeros(1536))
    
    # 가중합
    if embeddings:
        final_embedding = np.sum(embeddings, axis=0)
        # L2 정규화
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        return final_embedding.tolist()
    else:
        return [0.0] * 1536

def main():
    print("🤖 OpenAI 임베딩을 사용한 영화 임베딩 생성")
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
    print(f"💰 예상 비용: ${len(movies) * 5 * 0.00002:.4f} (5개 요소 × $0.00002/1K tokens)")
    
    # 자동 진행
    print("\n🚀 임베딩 생성을 시작합니다...")
    
    embeddings = []
    
    for i, movie in enumerate(movies):
        print(f"\n🎬 [{i+1}/{len(movies)}] '{movie.get('title', 'Unknown')}' 처리 중...")
        
        try:
            texts = get_text_for_openai_embedding(movie)
            embedding = create_weighted_openai_embedding(texts, WEIGHTS)
            
            embeddings.append({
                'title': movie.get('title', ''),
                'embedding': embedding
            })
            
            print(f"  ✅ 완료")
            
            # 진행상황 저장 (중간에 중단되어도 복구 가능)
            if (i + 1) % 50 == 0:
                with open(f'movie_embeddings_openai_backup_{i+1}.json', 'w', encoding='utf-8') as f:
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
    with open('movie_embeddings_openai.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ OpenAI 임베딩 생성 완료!")
    print(f"📁 저장 파일: movie_embeddings_openai.json")
    print(f"📊 총 {len(embeddings)}개 영화 임베딩 생성")

if __name__ == "__main__":
    main()

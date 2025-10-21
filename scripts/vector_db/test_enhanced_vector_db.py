"""
향상된 임베딩으로 벡터 DB 재구축 및 테스트
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def rebuild_vector_db():
    print("🔧 향상된 임베딩으로 벡터 DB 재구축")
    print("=" * 50)
    
    # 향상된 임베딩 로드
    with open('movie_embeddings_enhanced.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = np.array([item['embedding'] for item in data]).astype('float32')
    titles = [item['title'] for item in data]
    
    print(f"📊 임베딩 정보:")
    print(f"   벡터 개수: {len(embeddings)}")
    print(f"   벡터 차원: {embeddings.shape[1]}")
    
    # FAISS 인덱스 생성
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # 저장
    faiss.write_index(index, 'faiss_movie_index_enhanced.bin')
    
    with open('faiss_movie_meta_enhanced.json', 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 향상된 벡터 DB 저장 완료")
    
    return index, titles

def test_enhanced_search():
    print(f"\n🔎 향상된 검색 테스트")
    print("=" * 30)
    
    # 모델 및 데이터 로드
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index('faiss_movie_index_enhanced.bin')
    
    with open('faiss_movie_meta_enhanced.json', 'r', encoding='utf-8') as f:
        titles = json.load(f)
    
    # 다양한 검색어 테스트
    test_queries = [
        "액션 어드벤처 전쟁 영화",
        "로맨틱 코미디 사랑 이야기", 
        "스릴러 서스펜스 긴장감",
        "공포 호러 무서운",
        "SF 사이파이 미래 과학",
        "배신당한 조직원들에게 복수하는 이야기"
    ]
    
    for query in test_queries:
        print(f"\n🎬 검색어: '{query}'")
        
        # 쿼리 임베딩 (동일한 방식으로)
        query_embedding = model.encode(f"plot: {query}", normalize_embeddings=True)
        
        # 검색
        distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k=5)
        
        print(f"   결과:")
        unique_movies = set()
        result_count = 0
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            movie_title = titles[idx]
            if movie_title not in unique_movies:  # 중복 제거
                unique_movies.add(movie_title)
                result_count += 1
                similarity = 1 / (1 + distance)
                print(f"   {result_count}. {movie_title} (유사도: {similarity:.3f})")
                
                if result_count >= 3:  # 상위 3개만 표시
                    break
    
    # 벡터 다양성 재확인
    print(f"\n📈 벡터 다양성 분석:")
    
    # 랜덤 샘플링으로 벡터들 간 유사도 확인
    sample_indices = np.random.choice(len(titles), min(20, len(titles)), replace=False)
    sample_vectors = np.array([index.reconstruct(i) for i in sample_indices])
    
    # 정규화 후 코사인 유사도 계산
    sample_vectors = sample_vectors / np.linalg.norm(sample_vectors, axis=1, keepdims=True)
    similarities = np.dot(sample_vectors, sample_vectors.T)
    
    # 대각선 제외
    off_diagonal = similarities[np.triu_indices_from(similarities, k=1)]
    
    print(f"   20개 영화 샘플 간 평균 유사도: {off_diagonal.mean():.4f}")
    print(f"   유사도 표준편차: {off_diagonal.std():.4f}")
    print(f"   최대 유사도: {off_diagonal.max():.4f}")
    print(f"   최소 유사도: {off_diagonal.min():.4f}")
    
    if off_diagonal.mean() < 0.5:
        print("   ✅ 벡터 다양성 개선됨")
    else:
        print("   ⚠️  여전히 개선 필요")

if __name__ == "__main__":
    rebuild_vector_db()
    test_enhanced_search()

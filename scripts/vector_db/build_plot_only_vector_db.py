"""
순수 줄거리 임베딩으로 벡터 DB 구축 (전체 영화 정보 포함)
"""
import json
import numpy as np
import faiss

def build_plot_only_vector_db():
    print("📖 순수 줄거리 임베딩으로 벡터 DB 구축 (전체 영화 정보 포함)")
    print("=" * 50)
    
    # 순수 줄거리 임베딩 로드
    with open('movie_embeddings_plot_only.json', 'r', encoding='utf-8') as f:
        embedding_data = json.load(f)
    
    # 전체 영화 데이터셋 로드
    with open('movies_dataset.json', 'r', encoding='utf-8') as f:
        movies_dataset = json.load(f)
    
    # 영화 제목으로 매핑 딕셔너리 생성
    movie_dict = {movie['title']: movie for movie in movies_dataset}
    
    embeddings = []
    movie_metadata = []
    
    # 임베딩과 메타데이터 정리
    for item in embedding_data:
        title = item['title']
        if title in movie_dict:
            embeddings.append(item['embedding'])
            # 임베딩 제외한 전체 영화 정보 저장
            movie_info = movie_dict[title].copy()
            movie_metadata.append(movie_info)
        else:
            print(f"⚠️ 영화 정보 없음: {title}")
    
    embeddings = np.array(embeddings).astype('float32')
    
    print(f"📊 임베딩 정보:")
    print(f"   벡터 개수: {len(embeddings)}")
    print(f"   벡터 차원: {embeddings.shape[1]}")
    print(f"   메타데이터 개수: {len(movie_metadata)}")
    
    # FAISS 인덱스 생성
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # 저장
    faiss.write_index(index, 'faiss_movie_index_plot_only.bin')
    
    with open('faiss_movie_meta_plot_only.json', 'w', encoding='utf-8') as f:
        json.dump(movie_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 순수 줄거리 벡터 DB 저장 완료")
    print(f"   - 인덱스: faiss_movie_index_plot_only.bin")
    print(f"   - 메타데이터: faiss_movie_meta_plot_only.json (전체 영화 정보 포함)")

if __name__ == "__main__":
    build_plot_only_vector_db()

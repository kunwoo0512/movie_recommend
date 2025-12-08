#!/usr/bin/env python3
"""
SentenceBERT 임베딩을 사용한 FAISS 벡터 DB 구축
"""

import json
import numpy as np
import faiss
import os

def build_sentence_bert_vector_db():
    """SentenceBERT 임베딩으로 FAISS 벡터 DB 구축"""
    print("🔧 SentenceBERT 기반 FAISS 벡터 DB 구축")
    print("=" * 60)
    
    # 임베딩 데이터 로드
    embeddings_file = 'movie_embeddings_bert.json'
    try:
        print(f"[로드] {embeddings_file} 로딩...")
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        print(f"[성공] {len(embeddings_data)}개 영화 임베딩 로드 완료")
    except Exception as e:
        print(f"[오류] 임베딩 파일 로드 실패: {e}")
        return
    
    # 임베딩 벡터들 추출
    print("[처리] 임베딩 벡터 추출 중...")
    embeddings = []
    metadata = []
    
    valid_count = 0
    for i, movie in enumerate(embeddings_data):
        embedding = np.array(movie['embedding'], dtype='float32')
        
        # 유효성 검사 - SentenceBERT는 384차원
        if embedding.shape[0] != 384:
            print(f"[경고] {movie['title']}: 임베딩 차원 오류 ({embedding.shape[0]}) - 384차원 기대")
            continue
        
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print(f"[경고] {movie['title']}: 무효한 임베딩 값")
            continue
        
        embeddings.append(embedding)
        metadata.append({
            'title': movie['title'],
            'plot': movie['plot'],
            'year': movie['year'],
            'director': movie['director'],
            'genres': movie['genres'],
            'movie_id': movie['movie_id']
        })
        valid_count += 1
        
        # 진행 상황 출력 (100개마다)
        if (i + 1) % 100 == 0:
            print(f"   처리됨: {i + 1}/{len(embeddings_data)} (유효: {valid_count})")
    
    print(f"[결과] {valid_count}/{len(embeddings_data)}개 유효한 임베딩")
    
    if valid_count == 0:
        print("[오류] 유효한 임베딩이 없습니다.")
        return
    
    # 임베딩 배열 생성
    embeddings_matrix = np.vstack(embeddings).astype('float32')
    print(f"[정보] 임베딩 매트릭스 크기: {embeddings_matrix.shape}")
    
    # FAISS 인덱스 생성
    print("[구축] FAISS 인덱스 생성 중...")
    dimension = embeddings_matrix.shape[1]
    print(f"[정보] 임베딩 차원: {dimension}")
    
    # Inner Product 인덱스 사용 (SentenceBERT는 이미 정규화됨)
    index = faiss.IndexFlatIP(dimension)
    print("[정보] IndexFlatIP 사용 (코사인 유사도)")
    
    # 임베딩 추가
    print("[추가] 임베딩을 인덱스에 추가 중...")
    index.add(embeddings_matrix)
    print(f"[완료] {index.ntotal}개 벡터 추가 완료")
    
    # 파일 저장
    index_file = 'faiss_movie_index_bert.bin'
    meta_file = 'faiss_movie_meta_bert.json'
    
    try:
        print(f"[저장] {index_file} 저장 중...")
        faiss.write_index(index, index_file)
        print(f"[성공] 인덱스 저장 완료")
        
        print(f"[저장] {meta_file} 저장 중...")
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[성공] 메타데이터 저장 완료")
        
    except Exception as e:
        print(f"[오류] 파일 저장 실패: {e}")
        return
    
    # 결과 요약
    print("\n✅ SentenceBERT 기반 벡터 DB 구축 완료!")
    print("=" * 60)
    print(f"📊 통계:")
    print(f"  • 총 영화 수: {index.ntotal}")
    print(f"  • 임베딩 차원: {dimension}")
    print(f"  • 인덱스 파일: {index_file} ({os.path.getsize(index_file)/(1024*1024):.1f}MB)")
    print(f"  • 메타데이터: {meta_file} ({os.path.getsize(meta_file)/(1024*1024):.1f}MB)")
    
    # 테스트
    print(f"\n[테스트] 인덱스 로드 테스트...")
    try:
        test_index = faiss.read_index(index_file)
        print(f"[성공] 인덱스 로드 성공: {test_index.ntotal}개 벡터")
    except Exception as e:
        print(f"[오류] 인덱스 로드 실패: {e}")

if __name__ == "__main__":
    build_sentence_bert_vector_db()
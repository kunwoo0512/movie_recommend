#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

def load_metadata(jsonl_path: Path) -> List[Dict[str, Any]]:
    metas = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def ensure_unit_norm(x: np.ndarray) -> np.ndarray:
    # 코사인=내적을 위해 L2 정규화
    x = x.astype('float32', copy=False)
    faiss.normalize_L2(x)
    return x

def build_faiss_index(embeddings: np.ndarray, out_index: Path, out_idmap: Path):
    dim = embeddings.shape[1]
    # 안전망: 혹시 정규화 안 된 임베딩이 들어오면 여기서 정규화
    if not np.allclose((embeddings**2).sum(axis=1).mean(), 1.0, atol=1e-2):
        embeddings = ensure_unit_norm(embeddings)

    index = faiss.IndexFlatIP(dim)  # 코사인=IP (정규화 전제)
    index.add(embeddings)
    faiss.write_index(index, str(out_index))
    id_map = np.arange(embeddings.shape[0], dtype=np.int64)
    np.save(out_idmap, id_map)
    return index

def query_index(index: faiss.Index, q_vecs: np.ndarray, top_k: int = 5):
    return index.search(q_vecs, top_k)

def aggregate_by_movie(sims: np.ndarray, ids: np.ndarray, metas: List[Dict[str, Any]], 
                      top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """윈도우 결과를 영화별로 집계"""
    # 영화별 점수 집계 (딕셔너리 사용)
    movie_scores = {}
    
    for qi in range(sims.shape[0]):  # 쿼리별 처리
        for sid, score in zip(ids[qi], sims[qi]):
            try:
                meta = metas[int(sid)]
                movie_index = meta.get('movie_index')
                title = meta.get('title', f'Unknown_{movie_index}')
                
                if movie_index not in movie_scores:
                    movie_scores[movie_index] = {
                        'scores': [],
                        'meta': {
                            'title': title,
                            'year': meta.get('year', ''),
                            'director': meta.get('director', ''),
                            'movie_index': movie_index
                        }
                    }
                movie_scores[movie_index]['scores'].append(float(score))
            except (IndexError, KeyError) as e:
                print(f"[경고] 메타데이터 처리 오류: {e}")
                continue
    
    # 영화별 최종 점수 계산 (상위 점수들의 가중 평균)
    final_movies = []
    for movie_index, data in movie_scores.items():
        scores = sorted(data['scores'], reverse=True)
        # 상위 3개 윈도우의 가중 평균 (1.0, 0.7, 0.5)
        weights = [1.0, 0.7, 0.5]
        final_score = sum(s * w for s, w in zip(scores[:3], weights[:len(scores)])) / sum(weights[:len(scores)])
        
        final_movies.append({
            'score': final_score,
            'meta': data['meta']
        })
    
    # 점수순 정렬 후 top_k 선택
    final_movies.sort(key=lambda x: x['score'], reverse=True)
    final_movies = final_movies[:top_k]
    
    # numpy 배열 형태로 변환
    final_sims = np.array([[movie['score'] for movie in final_movies]], dtype='float32')
    final_ids = np.array([[i for i in range(len(final_movies))]], dtype='int64')
    final_metas = [movie['meta'] for movie in final_movies]
    
    return final_sims, final_ids, final_metas
    """윈도우 점수를 영화별로 집계"""
    from collections import defaultdict
    
    # 영화별 점수 수집
    movie_scores = defaultdict(list)
    movie_info = {}
    
    for sid, sim in zip(ids[0], sims[0]):
        meta = metas[int(sid)]
        movie_idx = meta.get('movie_index', meta.get('title'))  # movie_index 또는 title로 그룹핑
        
        movie_scores[movie_idx].append(float(sim))
        
        # 영화 정보 저장 (첫 번째 윈도우 정보 사용)
        if movie_idx not in movie_info:
            movie_info[movie_idx] = {
                'title': meta.get('title', 'Unknown'),
                'year': meta.get('year', ''),
                'director': meta.get('director', ''),
                'movie_index': movie_idx
            }
    
    # 영화별 최종 점수 계산 (상위 3개 윈도우 평균)
    movie_final_scores = []
    for movie_idx, scores in movie_scores.items():
        # 상위 3개 점수의 가중 평균 (첫 번째가 가장 높은 가중치)
        scores = sorted(scores, reverse=True)
        if len(scores) >= 3:
            final_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
        elif len(scores) == 2:
            final_score = (scores[0] * 0.7 + scores[1] * 0.3)
        else:
            final_score = scores[0]
        
        movie_final_scores.append((final_score, movie_idx))
    
    # 점수순 정렬
    movie_final_scores.sort(reverse=True)
    
    # 결과 재구성
    final_sims = np.array([[score for score, _ in movie_final_scores]], dtype='float32')
    final_ids = np.array([[movie_idx for _, movie_idx in movie_final_scores]], dtype='int64')
    final_metas = [movie_info[movie_idx] for _, movie_idx in movie_final_scores]
    
    return final_sims, final_ids, final_metas

def print_results(sims, ids, metas: List[Dict[str, Any]]):
    for qi in range(ids.shape[0]):
        print(f"\n[Query {qi}] Top-{ids.shape[1]} results")
        for rank, (sid, sim) in enumerate(zip(ids[qi], sims[qi]), start=1):
            # sid는 이제 집계된 영화의 인덱스
            m = metas[int(sid)]
            print(f"  {rank:>2}. score={sim:.4f} | {m.get('title')} ({m.get('year')}) | dir={m.get('director')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--build', action='store_true', help='FAISS 인덱스 생성')
    ap.add_argument('--demo_query', type=str, default='')
    ap.add_argument('--top_k', type=int, default=5)
    # ✅ 임베딩/쿼리 모델 통일을 위한 옵션 (임베딩 만들 때 쓴 것과 동일하게!)
    ap.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    emb_path = data_dir / 'embeddings.npy'
    meta_path = data_dir / 'metadata.jsonl'
    index_path = data_dir / 'index.faiss'
    idmap_path = data_dir / 'id_map.npy'

    embeddings = np.load(emb_path)          # (N, D)
    metas = load_metadata(meta_path)

    if args.build:
        print('[1/3] Building FAISS index')
        build_faiss_index(embeddings, index_path, idmap_path)
        print(f'✅ Saved index to {index_path} and id_map to {idmap_path}')

    if args.demo_query:
        print('[2/3] Encoding demo query')
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)

        with torch.inference_mode():
            q = model.encode([args.demo_query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
        q = ensure_unit_norm(q)  # 쿼리도 정규화

        # 차원 검증(모델 통일 확인)
        if q.shape[1] != embeddings.shape[1]:
            raise ValueError(f"Dim mismatch: query({q.shape[1]}) vs corpus({embeddings.shape[1]}). "
                             f"--model을 임베딩 생성 시 사용한 것과 동일하게 맞추세요.")

        print('[3/3] Searching')
        index = faiss.read_index(str(index_path))
        sims, ids = query_index(index, q, top_k=args.top_k * 3)  # 더 많이 검색해서 집계

        # 윈도우 점수를 영화별로 집계
        print('[4/4] Aggregating by movie')
        final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, metas, top_k=args.top_k)
        
        print_results(final_sims, final_ids, final_metas)

if __name__ == '__main__':
    main()
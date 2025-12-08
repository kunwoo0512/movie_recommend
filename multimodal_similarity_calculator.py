#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Movie Similarity Calculator (기존 방식 기반)

줄거리(청킹 후 평균) + 흐름곡선(384차원 확장) + 장르(384차원 확장)을 
사용자 지정 가중치로 실시간 결합하여 유사 영화 검색
"""

import os
import json
import numpy as np
import faiss
import re
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pathlib import Path

class MultimodalSimilarityCalculator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movie_metadata = None
        self.sbert_model = None
        self.embedding_dim = 384
        self.initialized = False
        
        # 미리 계산된 임베딩 캐시
        self.plot_embeddings_cache = None
        self.flow_embeddings_cache = None
        self.genre_embeddings_cache = None
        
    def load_all_data(self):
        """모든 데이터와 모델 로드"""
        try:
            print("🚀 멀티모달 시스템 초기화 중...")
            
            # SentenceBERT 모델 로드
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"📱 디바이스: {device}")
            
            self.sbert_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2', 
                device=device
            )
            print("🤖 SentenceBERT 모델 로드 완료")
            
            # 영화 메타데이터 로드 (plot 포함)
            metadata_path = os.path.join(self.data_dir, "multimodal_metadata.jsonl")
            if not os.path.exists(metadata_path):
                print(f"❌ 메타데이터를 찾을 수 없습니다: {metadata_path}")
                return False
                
            self.movie_metadata = []
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.movie_metadata.append(json.loads(line.strip()))
            
            print(f"📊 {len(self.movie_metadata)}개 영화 메타데이터 로드")
            
            # 각 모달리티별 임베딩 미리 계산
            self._precompute_embeddings()
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def create_plot_chunks(self, plot: str, window_size: int = 2, stride: int = 1) -> List[str]:
        """줄거리를 청킹"""
        sentences = self.split_sentences(plot)
        
        if len(sentences) <= window_size:
            return [plot]
        
        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = ' '.join(sentences[i:i + window_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_plot_embedding(self, plot: str) -> np.ndarray:
        """줄거리 임베딩 생성 (청킹 후 평균) - 기존 방식"""
        chunks = self.create_plot_chunks(plot)
        
        # 각 청크를 임베딩
        with torch.inference_mode():
            chunk_embeddings = self.sbert_model.encode(
                chunks, 
                convert_to_numpy=True, 
                normalize_embeddings=True
            )
        
        # 청크들의 평균으로 영화 대표 임베딩 생성
        if len(chunk_embeddings.shape) == 1:
            chunk_embeddings = chunk_embeddings.reshape(1, -1)
            
        movie_plot_embedding = np.mean(chunk_embeddings, axis=0)
        
        # L2 정규화
        norm = np.linalg.norm(movie_plot_embedding)
        if norm > 0:
            movie_plot_embedding = movie_plot_embedding / norm
            
        return movie_plot_embedding.astype('float32')
    
    def get_flow_embedding(self, flow_curve: List[float]) -> np.ndarray:
        """흐름곡선 임베딩 생성 (384차원 확장) - 기존 방식"""
        # 흐름곡선을 0-1로 정규화
        flow_array = np.array(flow_curve, dtype='float32')
        flow_normalized = flow_array / 10.0  # 0-10 스케일을 0-1로
        
        # 10개 값을 384차원으로 확장
        # 10개 값을 38번 반복 (380차원) + 평균값 4개
        expanded_flow = np.tile(flow_normalized, 38)  # 380차원
        
        # 나머지 4차원은 평균값으로 채움
        avg_flow = np.mean(flow_normalized)
        flow_padding = np.full(4, avg_flow, dtype='float32')
        
        flow_embedding = np.concatenate([expanded_flow, flow_padding])
        
        # L2 정규화
        norm = np.linalg.norm(flow_embedding)
        if norm > 0:
            flow_embedding = flow_embedding / norm
            
        return flow_embedding
    
    def get_genre_embedding(self, genres: Dict[str, int]) -> np.ndarray:
        """장르 점수 임베딩 생성 (384차원 확장) - 기존 방식"""
        # 장르 순서 고정
        genre_names = ['action', 'thriller', 'romance', 'drama', 'comedy', 'sci_fi', 'horror']
        
        # 장르 점수 벡터 생성
        genre_scores = []
        for genre in genre_names:
            score = genres.get(genre, 0) / 10.0  # 0-10을 0-1로 정규화
            genre_scores.append(score)
        
        genre_array = np.array(genre_scores, dtype='float32')
        
        # 7개 장르를 384차원으로 확장
        # 7개 값을 54번 반복 (378차원) + 평균값 6개
        expanded_genre = np.tile(genre_array, 54)  # 378차원 (7 * 54)
        
        # 나머지 6차원은 평균값으로 채움
        avg_genre = np.mean(genre_array)
        genre_padding = np.full(6, avg_genre, dtype='float32')
        
        genre_embedding = np.concatenate([expanded_genre, genre_padding])
        
        # L2 정규화
        norm = np.linalg.norm(genre_embedding)
        if norm > 0:
            genre_embedding = genre_embedding / norm
            
        return genre_embedding
    
    def _precompute_embeddings(self):
        """모든 영화의 각 모달리티별 임베딩 미리 계산"""
        print("📊 모든 영화의 임베딩 사전 계산 중...")
        
        plot_embeddings = []
        flow_embeddings = []
        genre_embeddings = []
        
        for i, movie in enumerate(self.movie_metadata):
            if i % 100 == 0:
                print(f"   진행률: {i}/{len(self.movie_metadata)}")
            
            # 각 모달리티별 임베딩 계산
            plot_emb = self.get_plot_embedding(movie['plot'])
            flow_emb = self.get_flow_embedding(movie['flow_curve'])
            genre_emb = self.get_genre_embedding(movie['genres'])
            
            plot_embeddings.append(plot_emb)
            flow_embeddings.append(flow_emb)
            genre_embeddings.append(genre_emb)
        
        # NumPy 배열로 변환
        self.plot_embeddings_cache = np.array(plot_embeddings, dtype='float32')
        self.flow_embeddings_cache = np.array(flow_embeddings, dtype='float32')
        self.genre_embeddings_cache = np.array(genre_embeddings, dtype='float32')
        
        print("✅ 임베딩 사전 계산 완료")
    
    def find_movie_by_title(self, title: str):
        """영화 제목으로 찾기"""
        title_lower = title.lower().strip()
        
        for i, movie in enumerate(self.movie_metadata):
            movie_title = movie.get('title', '').lower().strip()
            if title_lower == movie_title:
                return movie, i
                
            # 연도 포함된 경우 처리
            if '(' in movie_title and ')' in movie_title:
                title_without_year = movie_title.split('(')[0].strip()
                if title_lower == title_without_year:
                    return movie, i
        
        return None, -1
    
    def create_combined_embedding(self, movie_idx: int, w_plot: float, w_flow: float, w_genre: float) -> np.ndarray:
        """사용자 지정 가중치로 멀티모달 임베딩 생성"""
        # 각 모달리티 임베딩 가져오기
        plot_emb = self.plot_embeddings_cache[movie_idx]
        flow_emb = self.flow_embeddings_cache[movie_idx]
        genre_emb = self.genre_embeddings_cache[movie_idx]
        
        # 가중 결합 (기존 방식과 동일)
        total_weight = w_plot + w_flow + w_genre
        combined_embedding = (
            plot_emb * (w_plot / total_weight) +
            flow_emb * (w_flow / total_weight) + 
            genre_emb * (w_genre / total_weight)
        )
        
        # 최종 정규화
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
            
        return combined_embedding
    
    def calculate_weighted_similarity(self, movie_title: str, w_plot: float = 0.65, 
                                    w_flow: float = 0.25, w_genre: float = 0.10, 
                                    top_k: int = 10):
        """가중치 조절 가능한 유사 영화 검색 (기존 방식 기반)"""
        if not self.initialized:
            if not self.load_all_data():
                raise RuntimeError("데이터 로드 실패")
        
        print(f"🔍 '{movie_title}' 유사 영화 검색 (가중치: plot={w_plot:.2f}, flow={w_flow:.2f}, genre={w_genre:.2f})")
        
        # 대상 영화 찾기
        target_movie, target_idx = self.find_movie_by_title(movie_title)
        if target_movie is None:
            raise ValueError(f"영화를 찾을 수 없습니다: '{movie_title}'")
        
        print(f"🎯 대상 영화: {target_movie['title']} ({target_movie['year']})")
        
        # 모든 영화의 가중 결합 임베딩 생성
        print("⚖️ 가중치 기반 임베딩 생성 중...")
        all_embeddings = []
        for i in range(len(self.movie_metadata)):
            combined_emb = self.create_combined_embedding(i, w_plot, w_flow, w_genre)
            all_embeddings.append(combined_emb)
        
        embeddings_array = np.array(all_embeddings, dtype='float32')
        
        # FAISS 인덱스 생성 (실시간)
        index = faiss.IndexFlatIP(self.embedding_dim)  # 내적 기반 (정규화된 벡터에서 코사인 유사도)
        index.add(embeddings_array)
        
        # 대상 영화 임베딩으로 검색
        query_embedding = embeddings_array[target_idx:target_idx+1]
        similarities, indices = index.search(query_embedding, top_k + 1)  # 자기 자신 포함
        
        # 결과 정리
        similar_movies = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            # 자기 자신 제외
            if idx == target_idx:
                continue
            
            if len(similar_movies) >= top_k:
                break
                
            movie_meta = self.movie_metadata[idx]
            similar_movies.append({
                'rank': len(similar_movies) + 1,
                'title': movie_meta['title'],
                'year': movie_meta['year'],
                'director': movie_meta['director'],
                'similarity_score': float(sim),
                'genres': movie_meta['genres'],
                'flow_curve': movie_meta['flow_curve'],
                'plot': movie_meta['plot'][:200] + "..." if len(movie_meta['plot']) > 200 else movie_meta['plot'],
                'poster': movie_meta.get('poster', ''),
                'weights_used': {
                    'plot': w_plot / (w_plot + w_flow + w_genre),
                    'flow': w_flow / (w_plot + w_flow + w_genre),
                    'genre': w_genre / (w_plot + w_flow + w_genre)
                }
            })
        
        return similar_movies

# Factory function
def get_weighted_calculator(data_dir: str = "data"):
    """MultimodalSimilarityCalculator 인스턴스 반환"""
    return MultimodalSimilarityCalculator(data_dir)

# 테스트
if __name__ == "__main__":
    calculator = get_weighted_calculator()
    if calculator.load_all_data():
        print("\n🧪 Inception 테스트...")
        results = calculator.calculate_weighted_similarity("Inception", top_k=5)
        
        for movie in results:
            print(f"\n{movie['rank']}. {movie['title']} ({movie['year']})")
            print(f"   감독: {movie['director']}")
            print(f"   유사도: {movie['similarity_score']:.4f}")
            weights = movie['weights_used']
            print(f"   가중치: plot={weights['plot']:.2f}, flow={weights['flow']:.2f}, genre={weights['genre']:.2f}")
    else:
        print("❌ 시스템 초기화 실패")
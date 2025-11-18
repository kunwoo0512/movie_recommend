#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
분리된 임베딩 가중치 조절 유틸리티
기존 시스템에 가중치 조절 기능을 추가하기 위한 헬퍼 클래스
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import torch

class WeightedSearchHelper:
    def __init__(self, separated_embeddings_dir: str = "data/separated_embeddings"):
        """가중치 조절 헬퍼 초기화"""
        self.embeddings_dir = Path(separated_embeddings_dir)
        self.model = None
        
        # 임베딩들
        self.plot_embeddings = None
        self.flow_embeddings = None
        self.genre_embeddings = None
        
        # FAISS 인덱스들
        self.plot_index = None
        self.flow_index = None
        self.genre_index = None
        
        # 메타데이터
        self.metadata = None
        self.genre_info = None
        
        # 기본 가중치
        self.default_weights = {'plot': 0.6, 'flow': 0.3, 'genre': 0.1}
        
    def load_model(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """SBERT 모델 로드"""
        if self.model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(model_name, device=device)
        return self.model
    
    def load_separated_embeddings(self) -> bool:
        """분리된 임베딩들 로드"""
        try:
            # 임베딩 로드
            self.plot_embeddings = np.load(self.embeddings_dir / "plot_embeddings.npy")
            self.flow_embeddings = np.load(self.embeddings_dir / "flow_embeddings.npy")
            self.genre_embeddings = np.load(self.embeddings_dir / "genre_embeddings.npy")
            
            # FAISS 인덱스 로드
            self.plot_index = faiss.read_index(str(self.embeddings_dir / "plot_index.faiss"))
            self.flow_index = faiss.read_index(str(self.embeddings_dir / "flow_index.faiss"))
            self.genre_index = faiss.read_index(str(self.embeddings_dir / "genre_index.faiss"))
            
            # 메타데이터 로드
            self.metadata = []
            with open(self.embeddings_dir / "movie_metadata.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    self.metadata.append(json.loads(line))
            
            with open(self.embeddings_dir / "genre_info.json", 'r', encoding='utf-8') as f:
                self.genre_info = json.load(f)
            
            return True
        except Exception as e:
            print(f"❌ 분리된 임베딩 로드 실패: {e}")
            return False
    
    def encode_query_text(self, query: str) -> np.ndarray:
        """쿼리 텍스트를 임베딩으로 변환 (줄거리용)"""
        if self.model is None:
            self.load_model()
        
        with torch.inference_mode():
            embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].astype('float32')
    
    def encode_query_flow(self, query: str) -> np.ndarray:
        """쿼리에서 흐름 관련 정보 추출 후 임베딩"""
        # 흐름 관련 키워드 분석
        flow_keywords = {
            '긴장': '매우 긴장감 넘치는',
            '스릴': '매우 긴장감 넘치는', 
            '잔잔': '잔잔하고 평온한',
            '평온': '잔잔하고 평온한',
            '액션': '기복이 심한',
            '드라마': '일정한 흐름의'
        }
        
        flow_description = "적당히 긴장감 있는 일정한 흐름의"
        for keyword, description in flow_keywords.items():
            if keyword in query:
                flow_description = description
                break
        
        flow_text = f"{flow_description} 영화"
        return self.encode_query_text(flow_text)
    
    def encode_query_genre(self, query: str) -> np.ndarray:
        """쿼리에서 장르 정보 추출 후 임베딩"""
        # 장르 키워드 매핑
        genre_keywords = {
            '액션': 'action', '코미디': 'comedy', '드라마': 'drama',
            '공포': 'horror', '무서운': 'horror', '좀비': 'horror',
            '로맨스': 'romance', '사랑': 'romance', '연애': 'romance',
            'SF': 'sci_fi', '미래': 'sci_fi', '우주': 'sci_fi', '과학': 'sci_fi',
            '스릴러': 'thriller', '범죄': 'thriller', '수사': 'thriller'
        }
        
        detected_genres = []
        for keyword, genre in genre_keywords.items():
            if keyword in query:
                detected_genres.append(genre)
        
        if detected_genres:
            genre_text = f"강한 {' '.join(detected_genres)} 영화"
        else:
            genre_text = "일반적인 영화"
        
        return self.encode_query_text(genre_text)
    
    def search_with_weights(self, query: str, 
                          w_plot: float = None, w_flow: float = None, w_genre: float = None,
                          top_k: int = 20) -> List[Dict[str, Any]]:
        """가중치를 적용한 검색"""
        
        # 분리된 임베딩이 로드되지 않았으면 로드
        if self.plot_embeddings is None:
            if not self.load_separated_embeddings():
                return []
        
        # 가중치 정규화
        if all(w is None for w in [w_plot, w_flow, w_genre]):
            w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        else:
            weights = np.array([w_plot or 0, w_flow or 0, w_genre or 0])
            if weights.sum() > 0:
                weights = weights / weights.sum()
                w_plot, w_flow, w_genre = weights
            else:
                w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        
        # 각 모달리티별 쿼리 임베딩
        query_plot = self.encode_query_text(query)
        query_flow = self.encode_query_flow(query)
        query_genre = self.encode_query_genre(query)
        
        # 각 모달리티별 유사도 계산
        plot_scores = np.dot(self.plot_embeddings, query_plot)
        flow_scores = np.dot(self.flow_embeddings, query_flow)
        genre_scores = np.dot(self.genre_embeddings, query_genre)
        
        # 가중합 계산
        final_scores = (w_plot * plot_scores + w_flow * flow_scores + w_genre * genre_scores)
        
        # 상위 K개 결과 선택
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # 결과 포맷팅
        results = []
        for i, idx in enumerate(top_indices):
            movie_meta = self.metadata[idx]
            result = {
                'rank': i + 1,
                'movie_id': movie_meta['movie_id'],
                'title': movie_meta['title'],
                'year': movie_meta.get('year', ''),
                'director': movie_meta.get('director', ''),
                'final_score': float(final_scores[idx]),
                'component_scores': {
                    'plot': float(plot_scores[idx]),
                    'flow': float(flow_scores[idx]),
                    'genre': float(genre_scores[idx])
                },
                'weights_used': {
                    'plot': w_plot,
                    'flow': w_flow,
                    'genre': w_genre
                }
            }
            results.append(result)
        
        return results
    
    def find_similar_movies_weighted(self, target_movie_title: str,
                                   w_plot: float = None, w_flow: float = None, w_genre: float = None,
                                   top_k: int = 10) -> List[Dict[str, Any]]:
        """가중치를 적용한 유사 영화 검색"""
        
        # 분리된 임베딩이 로드되지 않았으면 로드
        if self.plot_embeddings is None:
            if not self.load_separated_embeddings():
                return []
        
        # 대상 영화 찾기
        target_idx = None
        for i, meta in enumerate(self.metadata):
            if target_movie_title.lower() in meta['title'].lower():
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        # 가중치 정규화
        if all(w is None for w in [w_plot, w_flow, w_genre]):
            w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        else:
            weights = np.array([w_plot or 0, w_flow or 0, w_genre or 0])
            if weights.sum() > 0:
                weights = weights / weights.sum()
                w_plot, w_flow, w_genre = weights
            else:
                w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        
        # 대상 영화의 임베딩들
        target_plot = self.plot_embeddings[target_idx]
        target_flow = self.flow_embeddings[target_idx]
        target_genre = self.genre_embeddings[target_idx]
        
        # 각 모달리티별 유사도 계산
        plot_scores = np.dot(self.plot_embeddings, target_plot)
        flow_scores = np.dot(self.flow_embeddings, target_flow)
        genre_scores = np.dot(self.genre_embeddings, target_genre)
        
        # 가중합 계산
        final_scores = (w_plot * plot_scores + w_flow * flow_scores + w_genre * genre_scores)
        
        # 자기 자신 제외하고 상위 K개 선택
        final_scores[target_idx] = -1  # 자기 자신 점수를 낮춤
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # 결과 포맷팅
        results = []
        for i, idx in enumerate(top_indices):
            if idx == target_idx:  # 혹시나 자기 자신이 포함되면 스킵
                continue
                
            movie_meta = self.metadata[idx]
            result = {
                'rank': len(results) + 1,
                'movie_id': movie_meta['movie_id'],
                'title': movie_meta['title'],
                'year': movie_meta.get('year', ''),
                'director': movie_meta.get('director', ''),
                'similarity_score': float(final_scores[idx]),
                'component_scores': {
                    'plot': float(plot_scores[idx]),
                    'flow': float(flow_scores[idx]),
                    'genre': float(genre_scores[idx])
                },
                'weights_used': {
                    'plot': w_plot,
                    'flow': w_flow,
                    'genre': w_genre
                },
                'genres': movie_meta.get('genres', {}),
                'flow_curve': movie_meta.get('flow_curve', [])
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results

# 전역 헬퍼 인스턴스
_weighted_helper = None

def get_weighted_helper():
    """전역 헬퍼 인스턴스 반환"""
    global _weighted_helper
    if _weighted_helper is None:
        _weighted_helper = WeightedSearchHelper()
    return _weighted_helper
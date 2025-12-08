#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Multimodal Movie Similarity Calculator

모든 모달리티(줄거리/흐름/장르)를 통일된 데이터셋으로 새로 생성하여
사용자 지정 가중치로 개별 점수를 계산하고 최종 추천 제공
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

class UnifiedMultimodalCalculator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movie_metadata = None  # flow_curve, genres 포함
        self.plot_metadata = None   # plot 포함 
        self.sbert_model = None
        self.embedding_dim = 384
        
        # 개별 FAISS 인덱스들
        self.plot_index = None
        self.flow_index = None
        self.genre_index = None
        
        self.initialized = False
        
    def load_all_data(self):
        """모든 데이터와 모델 로드"""
        try:
            print("[System] 통합 멀티모달 시스템 초기화 중...")
            
            # PyTorch meta tensor 문제 해결을 위한 강제 설정
            import torch
            import os
            
            # CUDA 완전 비활성화
            torch.cuda.is_available = lambda: False
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # SentenceBERT 모델 로드
            device = 'cpu'  # 강제로 CPU 사용
            print(f"[Device] 디바이스: {device} (강제 CPU 모드)")
            
            try:
                # meta tensor 문제 해결을 위해 CPU에서만 로드
                print("[Model] SentenceBERT 모델을 CPU에서 로드 중...")
                
                # 환경변수로 CPU 강제 설정
                import sentence_transformers
                
                self.sbert_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2', 
                    device='cpu'  # 항상 CPU에서 로드
                )
                
                # 모델을 명시적으로 CPU로 이동
                self.sbert_model.to('cpu')
                
                print("[Model] SentenceBERT 모델 CPU에서 로드 완료")
                    
            except Exception as e:
                print(f"[Error] SentenceBERT 모델 로드 실패: {str(e)}")
                # 폴백: 더 가벼운 모델 사용
                print("[Fallback] 기본 모델로 재시도...")
                try:
                    self.sbert_model = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    self.sbert_model.to('cpu')
                    print("[Model] 기본 SentenceBERT 모델 로드 완료")
                except Exception as e2:
                    print(f"[Error] 기본 모델도 실패: {str(e2)}")
                    # 가장 단순한 모델 시도
                    self.sbert_model = SentenceTransformer(
                        'sentence-transformers/all-MiniLM-L6-v2',
                        device='cpu'
                    )
                    self.sbert_model.to('cpu')
                    print("[Model] 최소 SentenceBERT 모델 로드 완료")
            
            # 영화 메타데이터 로드 (흐름/장르용)
            metadata_path = os.path.join(self.data_dir, "separated_embeddings", "movie_metadata.jsonl")
            if not os.path.exists(metadata_path):
                print(f"❌ 영화 메타데이터를 찾을 수 없습니다: {metadata_path}")
                return False
                
            self.movie_metadata = []
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.movie_metadata.append(json.loads(line.strip()))
            
            # 줄거리 메타데이터 로드
            plot_metadata_path = os.path.join(self.data_dir, "multimodal_metadata.jsonl")
            if not os.path.exists(plot_metadata_path):
                print(f"❌ 줄거리 메타데이터를 찾을 수 없습니다: {plot_metadata_path}")
                return False
                
            self.plot_metadata = []
            with open(plot_metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.plot_metadata.append(json.loads(line.strip()))
            
            print(f"[Data] {len(self.movie_metadata)}개 영화 (흐름/장르), {len(self.plot_metadata)}개 영화 (줄거리)")
            
            # 데이터 정합성 확인
            if len(self.movie_metadata) != len(self.plot_metadata):
                print(f"⚠️ 데이터 수 불일치: {len(self.movie_metadata)} vs {len(self.plot_metadata)}")
            
            # 인덱스들 생성 또는 로드
            self._load_or_create_indices()
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_or_create_indices(self):
        """인덱스들 로드 또는 새로 생성"""
        base_path = os.path.join(self.data_dir, "unified_indices")
        os.makedirs(base_path, exist_ok=True)
        
        plot_index_path = os.path.join(base_path, "plot_index.faiss")
        flow_index_path = os.path.join(base_path, "flow_index.faiss") 
        genre_index_path = os.path.join(base_path, "genre_index.faiss")
        
        # 모든 인덱스가 존재하는지 확인
        all_exist = all([
            os.path.exists(plot_index_path),
            os.path.exists(flow_index_path),
            os.path.exists(genre_index_path)
        ])
        
        if all_exist:
            print("[Loading] 기존 통합 인덱스들 로딩 중...")
            self.plot_index = faiss.read_index(plot_index_path)
            self.flow_index = faiss.read_index(flow_index_path)
            self.genre_index = faiss.read_index(genre_index_path)
            print("[Success] 모든 인덱스 로드 완료")
        else:
            print("[Create] 통합 인덱스들 새로 생성 중...")
            self._create_all_indices(base_path)
    
    def split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def create_plot_chunks(self, plot: str, window_size: int = 2, stride: int = 1) -> List[str]:
        """줄거리를 청킹 (기존 방식과 동일)"""
        sentences = self.split_sentences(plot)
        
        if len(sentences) <= window_size:
            return [plot]
        
        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = ' '.join(sentences[i:i + window_size])
            chunks.append(chunk)
        
        return chunks
    
    def get_plot_embedding(self, plot: str) -> np.ndarray:
        """줄거리 임베딩 생성 (청킹 후 평균)"""
        chunks = self.create_plot_chunks(plot)
        
        # 각 청크를 임베딩
        with torch.inference_mode():
            chunk_embeddings = self.sbert_model.encode(
                chunks, 
                convert_to_numpy=True, 
                normalize_embeddings=True,
                batch_size=32  # 배치 크기 제한
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
        """흐름곡선 임베딩 생성 (384차원 확장)"""
        flow_array = np.array(flow_curve, dtype='float32')
        flow_normalized = flow_array / 10.0  # 0-10 스케일을 0-1로
        
        # 10개 값을 384차원으로 확장
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
        """장르 점수 임베딩 생성 (384차원 확장)"""
        genre_names = ['action', 'thriller', 'romance', 'drama', 'comedy', 'sci_fi', 'horror']
        
        # 장르 점수 벡터 생성
        genre_scores = []
        for genre in genre_names:
            score = genres.get(genre, 0) / 10.0  # 0-10을 0-1로 정규화
            genre_scores.append(score)
        
        genre_array = np.array(genre_scores, dtype='float32')
        
        # 7개 장르를 384차원으로 확장
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
    
    def _create_all_indices(self, base_path: str):
        """모든 인덱스 새로 생성"""
        print("[Create] 모든 임베딩 계산 및 인덱스 생성 중...")
        
        plot_embeddings = []
        flow_embeddings = []
        genre_embeddings = []
        
        # 각 영화별로 임베딩 계산
        total_movies = len(self.movie_metadata)
        for i in range(total_movies):
            if i % 50 == 0:
                print(f"   진행률: {i}/{total_movies}")
            
            # 줄거리 임베딩 (multimodal_metadata에서)
            plot_meta = self.plot_metadata[i]
            plot_emb = self.get_plot_embedding(plot_meta['plot'])
            plot_embeddings.append(plot_emb)
            
            # 흐름/장르 임베딩 (movie_metadata에서)
            movie_meta = self.movie_metadata[i]
            flow_emb = self.get_flow_embedding(movie_meta['flow_curve'])
            genre_emb = self.get_genre_embedding(movie_meta['genres'])
            
            flow_embeddings.append(flow_emb)
            genre_embeddings.append(genre_emb)
        
        # NumPy 배열로 변환
        plot_array = np.array(plot_embeddings, dtype='float32')
        flow_array = np.array(flow_embeddings, dtype='float32')
        genre_array = np.array(genre_embeddings, dtype='float32')
        
        print(f"[Complete] 임베딩 계산 완료: Plot({plot_array.shape}), Flow({flow_array.shape}), Genre({genre_array.shape})")
        
        # FAISS 인덱스 생성 (내적 기반)
        self.plot_index = faiss.IndexFlatIP(self.embedding_dim)
        self.flow_index = faiss.IndexFlatIP(self.embedding_dim)
        self.genre_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 임베딩 추가
        self.plot_index.add(plot_array)
        self.flow_index.add(flow_array)
        self.genre_index.add(genre_array)
        
        # 인덱스 저장
        faiss.write_index(self.plot_index, os.path.join(base_path, "plot_index.faiss"))
        faiss.write_index(self.flow_index, os.path.join(base_path, "flow_index.faiss"))
        faiss.write_index(self.genre_index, os.path.join(base_path, "genre_index.faiss"))
        
        print("[Save] 모든 인덱스 저장 완료")
    
    def find_movie_by_title(self, title: str):
        """영화 제목으로 찾기"""
        title_lower = title.lower().strip()
        
        # 입력 제목에서 연도 분리
        input_title = title_lower
        input_year = None
        if '(' in title_lower and ')' in title_lower:
            parts = title_lower.split('(')
            input_title = parts[0].strip()
            year_part = parts[1].split(')')[0].strip()
            if year_part.isdigit():
                input_year = year_part
        
        for i, movie in enumerate(self.movie_metadata):
            movie_title = movie.get('title', '').lower().strip()
            movie_year = str(movie.get('year', '')).strip()
            
            # 정확한 매칭 (제목만)
            if input_title == movie_title:
                # 연도까지 지정된 경우 연도도 확인
                if input_year and input_year != movie_year:
                    continue
                return movie, i
            
            # 영화 제목에서 연도 제거하고 비교
            if '(' in movie_title and ')' in movie_title:
                title_without_year = movie_title.split('(')[0].strip()
                if input_title == title_without_year:
                    # 연도까지 지정된 경우 연도도 확인
                    if input_year and input_year != movie_year:
                        continue
                    return movie, i
        
        return None, -1
    
    def search_plot_similarity(self, movie_idx: int, k: int = 10):
        """줄거리 기반 유사도 검색"""
        similarities, indices = self.plot_index.search(
            np.array([self.plot_index.reconstruct(movie_idx)]), k + 1
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != movie_idx and len(results) < k:  # 자기 자신 제외
                movie = self.movie_metadata[idx]
                results.append({
                    'index': idx,
                    'title': movie['title'],
                    'year': movie['year'],
                    'similarity': float(sim)
                })
        
        return results
    
    def search_flow_similarity(self, movie_idx: int, k: int = 10):
        """흐름곡선 기반 유사도 검색"""
        similarities, indices = self.flow_index.search(
            np.array([self.flow_index.reconstruct(movie_idx)]), k + 1
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != movie_idx and len(results) < k:  # 자기 자신 제외
                movie = self.movie_metadata[idx]
                results.append({
                    'index': idx,
                    'title': movie['title'],
                    'year': movie['year'],
                    'similarity': float(sim)
                })
        
        return results
    
    def search_genre_similarity(self, movie_idx: int, k: int = 10):
        """장르 기반 유사도 검색"""
        similarities, indices = self.genre_index.search(
            np.array([self.genre_index.reconstruct(movie_idx)]), k + 1
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != movie_idx and len(results) < k:  # 자기 자신 제외
                movie = self.movie_metadata[idx]
                results.append({
                    'index': idx,
                    'title': movie['title'],
                    'year': movie['year'],
                    'similarity': float(sim)
                })
        
        return results
    
    def calculate_weighted_similarity(self, movie_title: str, w_plot: float = 0.8, 
                                    w_flow: float = 0.1, w_genre: float = 0.1, 
                                    top_k: int = 10):
        """가중치 기반 유사 영화 검색 - 모든 후보에 대해 직접 계산"""
        if not self.initialized:
            if not self.load_all_data():
                raise RuntimeError("시스템 초기화 실패")
        
        print(f"[Search] '{movie_title}' 유사 영화 검색 (가중치: plot={w_plot:.2f}, flow={w_flow:.2f}, genre={w_genre:.2f})")
        
        # 대상 영화 찾기
        target_movie, target_idx = self.find_movie_by_title(movie_title)
        if target_movie is None:
            raise ValueError(f"영화를 찾을 수 없습니다: '{movie_title}'")
        
        print(f"🎯 대상 영화: {target_movie['title']} ({target_movie['year']}) - 인덱스: {target_idx}")
        
        # 대상 영화의 임베딩 벡터들 가져오기
        target_plot_vec = self.plot_index.reconstruct(target_idx)
        target_flow_vec = self.flow_index.reconstruct(target_idx)
        target_genre_vec = self.genre_index.reconstruct(target_idx)
        
        print(f"[Embeddings] 대상 영화 임베딩 벡터 준비 완료")
        print(f"[Debug] 벡터 크기: plot={target_plot_vec.shape}, flow={target_flow_vec.shape}, genre={target_genre_vec.shape}")
        
        # 모든 영화에 대해 직접 유사도 계산
        all_movies_scores = []
        total_movies = len(self.movie_metadata)
        
        print(f"[Calculating] 전체 {total_movies}개 영화에 대해 유사도 계산 중...")
        
        for i in range(total_movies):
            if i == target_idx:  # 자기 자신 제외
                continue
                
            # 각 모달리티 유사도 직접 계산
            candidate_plot_vec = self.plot_index.reconstruct(i)
            candidate_flow_vec = self.flow_index.reconstruct(i)
            candidate_genre_vec = self.genre_index.reconstruct(i)
            
            # 코사인 유사도 계산 (내적 - 정규화된 벡터이므로)
            plot_similarity = float(np.dot(target_plot_vec, candidate_plot_vec))
            flow_similarity = float(np.dot(target_flow_vec, candidate_flow_vec))
            genre_similarity = float(np.dot(target_genre_vec, candidate_genre_vec))
            
            # 가중치 적용한 최종 점수
            total_weight = w_plot + w_flow + w_genre
            weighted_score = (
                w_plot * plot_similarity +
                w_flow * flow_similarity +
                w_genre * genre_similarity
            ) / total_weight
            
            # 메타데이터와 함께 저장
            movie_meta = self.movie_metadata[i]
            plot_meta = self.plot_metadata[i]
            
            all_movies_scores.append({
                'index': i,
                'title': movie_meta['title'],
                'year': movie_meta['year'],
                'director': movie_meta['director'],
                'similarity_score': weighted_score,
                'component_scores': {
                    'plot': plot_similarity,
                    'flow': flow_similarity,
                    'genre': genre_similarity
                },
                'weights_used': {
                    'plot': w_plot / total_weight,
                    'flow': w_flow / total_weight,
                    'genre': w_genre / total_weight
                },
                'genres': movie_meta['genres'],
                'flow_curve': movie_meta['flow_curve'],
                'plot': plot_meta['plot'][:200] + "..." if len(plot_meta['plot']) > 200 else plot_meta['plot'],
                'poster': movie_meta.get('poster', '')
            })
        
        # 최종 점수로 정렬
        all_movies_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 상위 top_k개 선택 및 순위 부여
        final_results = all_movies_scores[:top_k]
        for i, movie in enumerate(final_results):
            movie['rank'] = i + 1
        
        print(f"✅ 계산 완료: 총 {len(all_movies_scores)}개 후보 중 상위 {len(final_results)}개 선택")
        print(f"📊 점수 범위: {final_results[-1]['similarity_score']:.4f} ~ {final_results[0]['similarity_score']:.4f}")
        
        # 상위 3개 결과의 상세 정보 출력
        print(f"[Top 3 Results]")
        for i, movie in enumerate(final_results[:3]):
            comp = movie['component_scores']
            print(f"  {i+1}. {movie['title']} ({movie['year']}) - 최종점수: {movie['similarity_score']:.4f}")
            print(f"     세부점수: plot={comp['plot']:.3f}, flow={comp['flow']:.3f}, genre={comp['genre']:.3f}")
        
        return final_results

# Factory function
def get_weighted_calculator(data_dir: str = "data"):
    """UnifiedMultimodalCalculator 인스턴스 반환"""
    return UnifiedMultimodalCalculator(data_dir)

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
            comp = movie['component_scores']
            print(f"   세부: plot={comp['plot']:.3f}, flow={comp['flow']:.3f}, genre={comp['genre']:.3f}")
    else:
        print("❌ 시스템 초기화 실패")
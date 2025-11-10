#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사용자 조절 가능한 멀티모달 임베딩 시스템
각 모달리티를 분리 저장하여 실시간 가중치 조절 가능
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer
import torch
import faiss
import time

class FlexibleMultimodalSystem:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """유연한 멀티모달 시스템 초기화"""
        print("🚀 유연한 멀티모달 시스템 초기화")
        
        # SentenceBERT 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sbert_model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = 384
        
        # 분리된 임베딩들
        self.plot_embeddings = None
        self.flow_embeddings = None  
        self.genre_embeddings = None
        self.metadata = None
        
        # 기본 가중치
        self.default_weights = {
            'plot': 0.65,
            'flow': 0.25,
            'genre': 0.10
        }
        
    def create_separate_embeddings(self, input_file: str, output_dir: str = "data"):
        """각 모달리티별로 분리된 임베딩 생성"""
        print("\n🎬 영화 데이터 로딩...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        print(f"📊 총 {len(movies)}개 영화 발견")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plot_embeddings = []
        flow_embeddings = []
        genre_embeddings = []
        movie_metadata = []
        
        print("\n🔄 분리된 임베딩 생성 중...")
        start_time = time.time()
        
        for i, movie in enumerate(movies):
            try:
                if (i + 1) % 50 == 0 or i + 1 == len(movies):
                    elapsed = time.time() - start_time
                    print(f"   [{i+1:3d}/{len(movies)}] {movie.get('title', 'Unknown')} ({elapsed:.1f}초)")
                
                # 1. 줄거리 임베딩
                plot_emb = self._get_plot_embedding(movie['plot'])
                plot_embeddings.append(plot_emb)
                
                # 2. 흐름곡선 임베딩
                flow_emb = self._get_flow_embedding(movie['flow_curve'])
                flow_embeddings.append(flow_emb)
                
                # 3. 장르 임베딩
                genre_emb = self._get_genre_embedding(movie['genres'])
                genre_embeddings.append(genre_emb)
                
                # 메타데이터
                metadata = {
                    'movie_index': i,
                    'title': movie.get('title', 'Unknown'),
                    'year': movie.get('year', 'Unknown'),
                    'director': movie.get('director', 'Unknown'),
                    'plot': movie.get('plot', ''),
                    'flow_curve': movie.get('flow_curve', []),
                    'genres': movie.get('genres', {}),
                    'poster': movie.get('poster', '')
                }
                movie_metadata.append(metadata)
                
            except Exception as e:
                print(f"⚠️ 오류 - {movie.get('title', 'Unknown')}: {e}")
                continue
        
        # numpy 배열로 변환
        plot_array = np.vstack(plot_embeddings).astype('float32')
        flow_array = np.vstack(flow_embeddings).astype('float32') 
        genre_array = np.vstack(genre_embeddings).astype('float32')
        
        # 분리된 파일로 저장
        np.save(output_path / 'plot_embeddings.npy', plot_array)
        np.save(output_path / 'flow_embeddings.npy', flow_array)
        np.save(output_path / 'genre_embeddings.npy', genre_array)
        
        # 메타데이터 저장
        with open(output_path / 'flexible_metadata.jsonl', 'w', encoding='utf-8') as f:
            for meta in movie_metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\\n')
        
        print(f"\\n✅ 분리된 임베딩 저장 완료!")
        print(f"   📁 줄거리: {output_path / 'plot_embeddings.npy'}")
        print(f"   📁 흐름곡선: {output_path / 'flow_embeddings.npy'}")
        print(f"   📁 장르: {output_path / 'genre_embeddings.npy'}")
        print(f"   📁 메타데이터: {output_path / 'flexible_metadata.jsonl'}")
        
        return plot_array, flow_array, genre_array, movie_metadata
    
    def load_separate_embeddings(self, data_dir: str = "data"):
        """분리된 임베딩들 로드"""
        data_path = Path(data_dir)
        
        self.plot_embeddings = np.load(data_path / 'plot_embeddings.npy')
        self.flow_embeddings = np.load(data_path / 'flow_embeddings.npy')
        self.genre_embeddings = np.load(data_path / 'genre_embeddings.npy')
        
        # 메타데이터 로드
        self.metadata = []
        with open(data_path / 'flexible_metadata.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print(f"📁 분리된 임베딩 로드 완료:")
        print(f"   줄거리: {self.plot_embeddings.shape}")
        print(f"   흐름곡선: {self.flow_embeddings.shape}")
        print(f"   장르: {self.genre_embeddings.shape}")
        
    def combine_embeddings_with_weights(self, plot_weight: float, flow_weight: float, genre_weight: float) -> np.ndarray:
        """사용자 지정 가중치로 임베딩 결합"""
        # 가중치 정규화
        total_weight = plot_weight + flow_weight + genre_weight
        plot_w = plot_weight / total_weight
        flow_w = flow_weight / total_weight  
        genre_w = genre_weight / total_weight
        
        print(f"⚖️ 가중치: 줄거리({plot_w:.2f}) + 흐름곡선({flow_w:.2f}) + 장르({genre_w:.2f})")
        
        # 실시간 결합
        combined = (
            self.plot_embeddings * plot_w +
            self.flow_embeddings * flow_w +
            self.genre_embeddings * genre_w
        )
        
        # L2 정규화
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 0 나누기 방지
        combined = combined / norms
        
        return combined.astype('float32')
    
    def find_similar_movies_flexible(self, movie_title: str, movie_year: str = None,
                                   plot_weight: float = 0.65, flow_weight: float = 0.25, genre_weight: float = 0.10,
                                   top_k: int = 10) -> List[Dict[str, Any]]:
        """사용자 지정 가중치로 유사 영화 검색"""
        
        # 대상 영화 찾기
        target_index = self._find_movie_index(movie_title, movie_year)
        if target_index is None:
            raise ValueError(f"영화를 찾을 수 없습니다: {movie_title}")
        
        print(f"🎯 대상 영화: {self.metadata[target_index]['title']} ({self.metadata[target_index]['year']})")
        
        # 사용자 지정 가중치로 모든 임베딩 결합
        combined_embeddings = self.combine_embeddings_with_weights(plot_weight, flow_weight, genre_weight)
        
        # FAISS 인덱스 생성 (실시간)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(combined_embeddings)
        
        # 대상 영화의 결합 임베딩으로 검색
        query_vector = combined_embeddings[target_index:target_index+1]
        similarities, indices = index.search(query_vector, top_k + 1)  # +1 for excluding self
        
        # 결과 정리
        similar_movies = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == target_index:  # 자기 자신 제외
                continue
            
            if len(similar_movies) >= top_k:
                break
                
            movie_meta = self.metadata[idx]
            similar_movies.append({
                'rank': len(similar_movies) + 1,
                'title': movie_meta['title'],
                'year': movie_meta['year'], 
                'director': movie_meta['director'],
                'similarity_score': float(sim),
                'genres': movie_meta['genres'],
                'flow_curve': movie_meta['flow_curve'],
                'plot': movie_meta['plot'][:200] + "..." if len(movie_meta['plot']) > 200 else movie_meta['plot'],
                'poster': movie_meta['poster']
            })
        
        return similar_movies
    
    def _find_movie_index(self, title: str, year: str = None) -> int:
        """영화 인덱스 찾기"""
        for i, meta in enumerate(self.metadata):
            if title.lower() in meta['title'].lower():
                if year is None or meta['year'] == year:
                    return i
        return None
    
    # 기존 임베딩 생성 메서드들 (create_multimodal_embeddings.py에서 복사)
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _create_plot_chunks(self, plot: str, window_size: int = 2, stride: int = 1) -> List[str]:
        sentences = self._split_sentences(plot)
        
        if len(sentences) <= window_size:
            return [plot]
        
        chunks = []
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = ' '.join(sentences[i:i + window_size])
            chunks.append(chunk)
        
        return chunks
    
    def _get_plot_embedding(self, plot: str) -> np.ndarray:
        chunks = self._create_plot_chunks(plot)
        
        with torch.inference_mode():
            chunk_embeddings = self.sbert_model.encode(
                chunks, convert_to_numpy=True, normalize_embeddings=True
            )
        
        movie_plot_embedding = np.mean(chunk_embeddings, axis=0)
        
        norm = np.linalg.norm(movie_plot_embedding)
        if norm > 0:
            movie_plot_embedding = movie_plot_embedding / norm
            
        return movie_plot_embedding.astype('float32')
    
    def _get_flow_embedding(self, flow_curve: List[float]) -> np.ndarray:
        flow_array = np.array(flow_curve, dtype='float32')
        flow_normalized = flow_array / 10.0
        
        expanded_flow = np.tile(flow_normalized, 38)
        avg_flow = np.mean(flow_normalized) 
        flow_padding = np.full(4, avg_flow, dtype='float32')
        
        flow_embedding = np.concatenate([expanded_flow, flow_padding])
        
        norm = np.linalg.norm(flow_embedding)
        if norm > 0:
            flow_embedding = flow_embedding / norm
            
        return flow_embedding
    
    def _get_genre_embedding(self, genres: Dict[str, int]) -> np.ndarray:
        genre_names = ['action', 'thriller', 'romance', 'drama', 'comedy', 'sci_fi', 'horror']
        
        genre_scores = []
        for genre in genre_names:
            score = genres.get(genre, 0) / 10.0
            genre_scores.append(score)
        
        genre_array = np.array(genre_scores, dtype='float32')
        
        expanded_genre = np.tile(genre_array, 54)
        avg_genre = np.mean(genre_array)
        genre_padding = np.full(6, avg_genre, dtype='float32')
        
        genre_embedding = np.concatenate([expanded_genre, genre_padding])
        
        norm = np.linalg.norm(genre_embedding)
        if norm > 0:
            genre_embedding = genre_embedding / norm
            
        return genre_embedding

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="유연한 멀티모달 영화 추천")
    parser.add_argument('--build', action='store_true', help='분리된 임베딩 생성')
    parser.add_argument('--movie', type=str, help='검색할 영화 제목')
    parser.add_argument('--year', type=str, help='영화 연도')
    parser.add_argument('--plot_weight', type=float, default=0.65, help='줄거리 가중치')
    parser.add_argument('--flow_weight', type=float, default=0.25, help='흐름곡선 가중치')
    parser.add_argument('--genre_weight', type=float, default=0.10, help='장르 가중치')
    parser.add_argument('--top_k', type=int, default=10, help='추천 영화 수')
    
    args = parser.parse_args()
    
    try:
        system = FlexibleMultimodalSystem()
        
        if args.build:
            # 분리된 임베딩 생성
            input_file = "data/processed/movies_dataset.json"
            system.create_separate_embeddings(input_file)
        else:
            # 임베딩 로드
            system.load_separate_embeddings()
            
            if args.movie:
                # 단일 검색
                similar_movies = system.find_similar_movies_flexible(
                    args.movie, args.year,
                    args.plot_weight, args.flow_weight, args.genre_weight,
                    args.top_k
                )
                
                # 결과 출력
                print(f"\\n🎬 '{args.movie}'와 유사한 영화들 (가중치 조정)")
                print("=" * 80)
                
                for movie in similar_movies:
                    print(f"\\n{movie['rank']:2d}. {movie['title']} ({movie['year']})")
                    print(f"    감독: {movie['director']}")
                    print(f"    유사도: {movie['similarity_score']:.3f}")
            else:
                # 대화형 모드
                print("\\n🎬 유연한 멀티모달 추천 시스템")
                print("=" * 50)
                print("• 가중치를 실시간으로 조절할 수 있습니다")
                print("• 'quit' 입력시 종료")
                print("=" * 50)
                
                while True:
                    try:
                        movie_title = input("\\n🔍 영화 제목 > ").strip()
                        if movie_title.lower() in ['quit', 'exit']:
                            break
                        
                        # 가중치 입력
                        print("가중치 입력 (기본값: 줄거리 0.65, 흐름곡선 0.25, 장르 0.10)")
                        plot_w = float(input("줄거리 가중치 > ") or "0.65")
                        flow_w = float(input("흐름곡선 가중치 > ") or "0.25") 
                        genre_w = float(input("장르 가중치 > ") or "0.10")
                        
                        similar_movies = system.find_similar_movies_flexible(
                            movie_title, None, plot_w, flow_w, genre_w, 5
                        )
                        
                        # 결과 출력
                        print(f"\\n🎬 '{movie_title}'와 유사한 영화들")
                        for movie in similar_movies:
                            print(f"{movie['rank']}. {movie['title']} ({movie['year']}) - {movie['similarity_score']:.3f}")
                        
                    except Exception as e:
                        print(f"❌ 오류: {e}")
                    except KeyboardInterrupt:
                        break
    
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
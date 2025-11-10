#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티모달 영화 임베딩 생성기
줄거리 (65%) + 흐름곡선 (25%) + 장르 (10%) 가중 결합
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import re
from sentence_transformers import SentenceTransformer
import torch
from sklearn.preprocessing import MinMaxScaler
import faiss
import time

class MultimodalEmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """멀티모달 임베딩 생성기 초기화"""
        print("🚀 멀티모달 임베딩 생성기 초기화")
        
        # SentenceBERT 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📱 디바이스: {self.device}")
        
        self.sbert_model = SentenceTransformer(model_name, device=self.device)
        print(f"🤖 SentenceBERT 모델 로드: {model_name}")
        
        # 임베딩 차원
        self.embedding_dim = 384
        
        # 가중치 설정
        self.weights = {
            'plot': 0.65,
            'flow': 0.25, 
            'genre': 0.10
        }
        print(f"⚖️ 가중치: {self.weights}")
        
    def split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 마침표, 느낌표, 물음표로 문장 분할
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def create_plot_chunks(self, plot: str, window_size: int = 2, stride: int = 1) -> List[str]:
        """줄거리를 청킹 (기존 방식과 동일)"""
        sentences = self.split_sentences(plot)
        
        if len(sentences) <= window_size:
            return [plot]  # 문장이 적으면 전체 반환
        
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
                normalize_embeddings=True
            )
        
        # 청크들의 평균으로 영화 대표 임베딩 생성
        movie_plot_embedding = np.mean(chunk_embeddings, axis=0)
        
        # L2 정규화
        norm = np.linalg.norm(movie_plot_embedding)
        if norm > 0:
            movie_plot_embedding = movie_plot_embedding / norm
            
        return movie_plot_embedding.astype('float32')
    
    def get_flow_embedding(self, flow_curve: List[float]) -> np.ndarray:
        """흐름곡선 임베딩 생성"""
        # 흐름곡선을 0-1로 정규화
        flow_array = np.array(flow_curve, dtype='float32')
        flow_normalized = flow_array / 10.0  # 0-10 스케일을 0-1로
        
        # 단순한 방법: 흐름곡선을 반복해서 384차원으로 확장
        # 10개 값을 38.4번 반복 (384 / 10 = 38.4)
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
        """장르 점수 임베딩 생성"""
        # 장르 순서 고정
        genre_names = ['action', 'thriller', 'romance', 'drama', 'comedy', 'sci_fi', 'horror']
        
        # 장르 점수 벡터 생성
        genre_scores = []
        for genre in genre_names:
            score = genres.get(genre, 0) / 10.0  # 0-10을 0-1로 정규화
            genre_scores.append(score)
        
        genre_array = np.array(genre_scores, dtype='float32')
        
        # 7개 장르를 384차원으로 확장
        # 7개 값을 54.85번 반복 (384 / 7 ≈ 54.85)
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
    
    def create_multimodal_embedding(self, movie_data: Dict[str, Any]) -> np.ndarray:
        """멀티모달 임베딩 생성 (Weighted Sum 방식)"""
        # 1. 각 모달리티별 임베딩 생성
        plot_emb = self.get_plot_embedding(movie_data['plot'])
        flow_emb = self.get_flow_embedding(movie_data['flow_curve'])
        genre_emb = self.get_genre_embedding(movie_data['genres'])
        
        # 2. 가중 결합
        combined_embedding = (
            plot_emb * self.weights['plot'] +
            flow_emb * self.weights['flow'] + 
            genre_emb * self.weights['genre']
        )
        
        # 3. 최종 정규화
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
            
        return combined_embedding
    
    def process_all_movies(self, input_file: str, output_dir: str = "data"):
        """모든 영화의 멀티모달 임베딩 생성"""
        print("\n🎬 영화 데이터 로딩...")
        
        # 입력 파일 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        print(f"📊 총 {len(movies)}개 영화 발견")
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 임베딩 저장을 위한 리스트
        all_embeddings = []
        movie_metadata = []
        
        print("\n🔄 멀티모달 임베딩 생성 중...")
        start_time = time.time()
        
        for i, movie in enumerate(movies):
            try:
                # 진행률 출력
                if (i + 1) % 50 == 0 or i + 1 == len(movies):
                    elapsed = time.time() - start_time
                    print(f"   [{i+1:3d}/{len(movies)}] {movie.get('title', 'Unknown')} "
                          f"({elapsed:.1f}초 경과)")
                
                # 멀티모달 임베딩 생성
                multimodal_emb = self.create_multimodal_embedding(movie)
                all_embeddings.append(multimodal_emb)
                
                # 메타데이터 저장
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
        embeddings_array = np.vstack(all_embeddings).astype('float32')
        
        print(f"\n💾 임베딩 저장 중...")
        print(f"   형태: {embeddings_array.shape}")
        print(f"   크기: {embeddings_array.nbytes / 1024 / 1024:.2f} MB")
        
        # 파일 저장
        np.save(output_path / 'multimodal_embeddings.npy', embeddings_array)
        
        # 메타데이터 저장 (JSONL 형식)
        with open(output_path / 'multimodal_metadata.jsonl', 'w', encoding='utf-8') as f:
            for meta in movie_metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        print(f"✅ 완료!")
        print(f"   📁 임베딩: {output_path / 'multimodal_embeddings.npy'}")
        print(f"   📁 메타데이터: {output_path / 'multimodal_metadata.jsonl'}")
        
        return embeddings_array, movie_metadata

def main():
    """메인 실행 함수"""
    try:
        # 임베딩 생성기 초기화
        generator = MultimodalEmbeddingGenerator()
        
        # 영화 데이터셋 경로
        input_file = "data/processed/movies_dataset.json"
        
        if not Path(input_file).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
            return
        
        # 멀티모달 임베딩 생성
        embeddings, metadata = generator.process_all_movies(input_file)
        
        print(f"\n🎉 멀티모달 임베딩 생성 완료!")
        print(f"   영화 수: {len(metadata)}")
        print(f"   임베딩 차원: {embeddings.shape[1]}")
        print(f"   가중치: plot(65%) + flow(25%) + genre(10%)")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
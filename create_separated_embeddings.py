#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기본 SBERT를 사용한 분리된 임베딩 생성
줄거리, 흐름곡선, 장르를 별도 임베딩으로 분리하여 저장
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class SeparatedEmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """분리된 임베딩 생성기 초기화"""
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🎯 사용 디바이스: {self.device}")
        
    def load_model(self):
        """SBERT 모델 로드"""
        print(f"🤖 SBERT 모델 로딩: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("✅ 모델 로딩 완료")
        
    def load_movie_data(self, data_file: str = "data/processed/movies_dataset.json") -> List[Dict[str, Any]]:
        """영화 데이터 로드"""
        print(f"📁 영화 데이터 로드 중: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        print(f"✅ {len(movies)}개 영화 데이터 로드 완료")
        return movies
    
    def generate_plot_embeddings(self, movies: List[Dict[str, Any]]) -> np.ndarray:
        """줄거리 임베딩 생성"""
        print("📝 줄거리 임베딩 생성 중...")
        
        plot_texts = []
        for movie in movies:
            # 청크된 줄거리들을 결합
            if 'chunks' in movie and movie['chunks']:
                # 청크들을 하나로 결합 (기존 방식과 동일)
                combined_plot = " ".join(chunk['text'] for chunk in movie['chunks'])
            else:
                # 전체 줄거리 사용
                combined_plot = movie.get('plot', '')
            
            plot_texts.append(combined_plot)
        
        # 배치 임베딩 생성
        embeddings = self.model.encode(
            plot_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"✅ 줄거리 임베딩 완료: {embeddings.shape}")
        return embeddings
    
    def generate_flow_embeddings(self, movies: List[Dict[str, Any]]) -> np.ndarray:
        """흐름곡선 임베딩 생성"""
        print("📈 흐름곡선 임베딩 생성 중...")
        
        flow_texts = []
        for movie in movies:
            flow_curve = movie.get('flow_curve', [])
            
            if flow_curve:
                # 흐름곡선을 텍스트로 변환 (기존 방식)
                flow_description = self._flow_to_text(flow_curve)
            else:
                flow_description = "평균적인 영화 흐름"
            
            flow_texts.append(flow_description)
        
        # 배치 임베딩 생성
        embeddings = self.model.encode(
            flow_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"✅ 흐름곡선 임베딩 완료: {embeddings.shape}")
        return embeddings
    
    def _flow_to_text(self, flow_curve: List[float]) -> str:
        """흐름곡선을 텍스트로 변환"""
        if not flow_curve:
            return "평균적인 영화 흐름"
        
        avg_tension = sum(flow_curve) / len(flow_curve)
        max_tension = max(flow_curve)
        min_tension = min(flow_curve)
        
        # 흐름 특성 분석
        descriptions = []
        
        if avg_tension > 7:
            descriptions.append("매우 긴장감 넘치는")
        elif avg_tension > 5:
            descriptions.append("적당히 긴장감 있는")
        else:
            descriptions.append("잔잔하고 평온한")
        
        if max_tension - min_tension > 5:
            descriptions.append("기복이 심한")
        else:
            descriptions.append("일정한 흐름의")
        
        # 클라이맥스 위치 분석
        max_idx = flow_curve.index(max_tension)
        if max_idx < len(flow_curve) * 0.3:
            descriptions.append("초반 클라이맥스")
        elif max_idx > len(flow_curve) * 0.7:
            descriptions.append("후반 클라이맥스")
        else:
            descriptions.append("중반 클라이맥스")
        
        return " ".join(descriptions) + " 영화"
    
    def generate_genre_embeddings(self, movies: List[Dict[str, Any]]) -> np.ndarray:
        """장르 임베딩 생성"""
        print("🎭 장르 임베딩 생성 중...")
        
        genre_texts = []
        for movie in movies:
            genres = movie.get('genres', {})
            
            if genres:
                # 장르를 텍스트로 변환 (기존 방식)
                genre_description = self._genres_to_text(genres)
            else:
                genre_description = "일반적인 영화"
            
            genre_texts.append(genre_description)
        
        # 배치 임베딩 생성
        embeddings = self.model.encode(
            genre_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"✅ 장르 임베딩 완료: {embeddings.shape}")
        return embeddings
    
    def _genres_to_text(self, genres: Dict[str, float]) -> str:
        """장르 딕셔너리를 텍스트로 변환"""
        if not genres:
            return "일반적인 영화"
        
        # 상위 장르들 선택 (점수 기준)
        sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)
        top_genres = sorted_genres[:3]  # 상위 3개 장르
        
        genre_descriptions = []
        for genre, score in top_genres:
            if score > 7:
                intensity = "매우 강한"
            elif score > 5:
                intensity = "강한"
            else:
                intensity = "약간의"
            
            genre_descriptions.append(f"{intensity} {genre}")
        
        return " ".join(genre_descriptions) + " 영화"
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """FAISS 인덱스 생성"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product (정규화된 벡터용)
        index.add(embeddings.astype('float32'))
        return index
    
    def save_separated_embeddings(self, movies: List[Dict[str, Any]], 
                                plot_embeddings: np.ndarray,
                                flow_embeddings: np.ndarray, 
                                genre_embeddings: np.ndarray,
                                output_dir: str = "data/separated_embeddings"):
        """분리된 임베딩들을 파일로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 분리된 임베딩 저장 중: {output_path}")
        
        # 임베딩 저장
        np.save(output_path / "plot_embeddings.npy", plot_embeddings)
        np.save(output_path / "flow_embeddings.npy", flow_embeddings)
        np.save(output_path / "genre_embeddings.npy", genre_embeddings)
        
        # FAISS 인덱스 생성 및 저장
        plot_index = self.create_faiss_index(plot_embeddings)
        flow_index = self.create_faiss_index(flow_embeddings)
        genre_index = self.create_faiss_index(genre_embeddings)
        
        faiss.write_index(plot_index, str(output_path / "plot_index.faiss"))
        faiss.write_index(flow_index, str(output_path / "flow_index.faiss"))
        faiss.write_index(genre_index, str(output_path / "genre_index.faiss"))
        
        # 메타데이터 저장
        metadata = []
        for i, movie in enumerate(movies):
            metadata.append({
                'index': i,
                'movie_id': movie.get('movie_id', f"movie_{i}"),
                'title': movie.get('title', ''),
                'year': movie.get('year', ''),
                'director': movie.get('director', ''),
                'genres': movie.get('genres', {}),
                'flow_curve': movie.get('flow_curve', []),
                'poster': movie.get('poster', '')
            })
        
        with open(output_path / "movie_metadata.jsonl", 'w', encoding='utf-8') as f:
            for meta in metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        # 장르 정보 저장
        all_genres = set()
        for movie in movies:
            if 'genres' in movie:
                all_genres.update(movie['genres'].keys())
        
        genre_info = {
            'genre_types': sorted(list(all_genres)),
            'total_movies': len(movies),
            'embedding_dim': plot_embeddings.shape[1]
        }
        
        with open(output_path / "genre_info.json", 'w', encoding='utf-8') as f:
            json.dump(genre_info, f, ensure_ascii=False, indent=2)
        
        print("✅ 모든 분리된 임베딩 저장 완료!")
        print(f"   📝 줄거리: plot_embeddings.npy, plot_index.faiss")
        print(f"   📈 흐름곡선: flow_embeddings.npy, flow_index.faiss") 
        print(f"   🎭 장르: genre_embeddings.npy, genre_index.faiss")
        print(f"   📋 메타데이터: movie_metadata.jsonl, genre_info.json")

def main():
    """메인 실행 함수"""
    print("🚀 기본 SBERT 기반 분리된 임베딩 생성 시작")
    print("=" * 60)
    
    # 생성기 초기화
    generator = SeparatedEmbeddingGenerator()
    generator.load_model()
    
    # 데이터 로드
    movies = generator.load_movie_data()
    
    # 분리된 임베딩 생성
    plot_embeddings = generator.generate_plot_embeddings(movies)
    flow_embeddings = generator.generate_flow_embeddings(movies)
    genre_embeddings = generator.generate_genre_embeddings(movies)
    
    # 저장
    generator.save_separated_embeddings(
        movies, plot_embeddings, flow_embeddings, genre_embeddings
    )
    
    print("\n🎉 분리된 임베딩 생성 완료!")
    print(f"출력 디렉토리: data/separated_embeddings/")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파인튜닝된 분리 임베딩 시스템 (사용자 가중치 조절)
3개 타입의 임베딩을 분리 저장하여 실시간 가중치 조절 가능:
- 줄거리 임베딩 (파인튜닝된 SBERT)
- 흐름곡선 임베딩 
- 장르 임베딩
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re
from sentence_transformers import SentenceTransformer
import torch
import faiss
import time

class FlexibleMultimodalSystem:
    def __init__(self):
        """파인튜닝된 분리 임베딩 시스템 초기화"""
        print("🚀 파인튜닝된 분리 임베딩 시스템 초기화")
        
        # 파인튜닝된 SBERT 모델 로드 (쿼리 임베딩용)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_finetuned_model()
        self.embedding_dim = 384
        
        # 분리된 임베딩들과 메타데이터
        self.plot_embeddings = None
        self.flow_embeddings = None  
        self.genre_embeddings = None
        self.movie_metadata = None
        self.genre_info = None
        
        # FAISS 인덱스들
        self.plot_index = None
        self.flow_index = None
        self.genre_index = None
        
        # 기본 가중치
        self.default_weights = {
            'plot': 0.6,
            'flow': 0.3,
            'genre': 0.1
        }
        
    def _load_finetuned_model(self):
        """파인튜닝된 SBERT 모델 로드"""
        print("🤖 파인튜닝된 SBERT 모델 로딩 중...")
        
        possible_paths = [
            "models/movie-finetuned-sbert",
            "models/finetuned_sbert", 
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        model_path = None
        for path in possible_paths:
            if Path(path).exists() and (not Path(path).is_dir() or any(Path(path).iterdir())):
                model_path = path
                break
        
        if not model_path:
            model_path = "paraphrase-multilingual-MiniLM-L12-v2"
            
        self.sbert_model = SentenceTransformer(model_path, device=self.device)
        print(f"✅ 모델 로드 완료: {model_path}")
        
    def load_separated_embeddings(self, embeddings_dir: str = "data/separated_embeddings"):
        """분리된 임베딩들을 로드"""
        print(f"\n📂 분리된 임베딩 로드 중: {embeddings_dir}")
        
        embeddings_path = Path(embeddings_dir)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"임베딩 디렉토리를 찾을 수 없습니다: {embeddings_dir}")
        
        # 1. 줄거리 임베딩 로드
        plot_file = embeddings_path / "plot_embeddings.npy"
        if plot_file.exists():
            self.plot_embeddings = np.load(plot_file)
            print(f"📝 줄거리 임베딩 로드: {self.plot_embeddings.shape}")
        else:
            raise FileNotFoundError(f"줄거리 임베딩 파일을 찾을 수 없습니다: {plot_file}")
            
        # 2. 흐름곡선 임베딩 로드
        flow_file = embeddings_path / "flow_embeddings.npy"
        if flow_file.exists():
            self.flow_embeddings = np.load(flow_file)
            print(f"📈 흐름곡선 임베딩 로드: {self.flow_embeddings.shape}")
        else:
            raise FileNotFoundError(f"흐름곡선 임베딩 파일을 찾을 수 없습니다: {flow_file}")
            
        # 3. 장르 임베딩 로드
        genre_file = embeddings_path / "genre_embeddings.npy"
        if genre_file.exists():
            self.genre_embeddings = np.load(genre_file)
            print(f"🎭 장르 임베딩 로드: {self.genre_embeddings.shape}")
        else:
            raise FileNotFoundError(f"장르 임베딩 파일을 찾을 수 없습니다: {genre_file}")
            
        # 4. 메타데이터 로드
        metadata_file = embeddings_path / "movie_metadata.jsonl"
        if metadata_file.exists():
            self.movie_metadata = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.movie_metadata.append(json.loads(line))
            print(f"📋 영화 메타데이터 로드: {len(self.movie_metadata)}개")
        else:
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_file}")
            
        # 5. 장르 정보 로드
        genre_info_file = embeddings_path / "genre_info.json"
        if genre_info_file.exists():
            with open(genre_info_file, 'r', encoding='utf-8') as f:
                self.genre_info = json.load(f)
            print(f"📊 장르 정보 로드: {self.genre_info['dimension']}개 장르")
        
        # 6. FAISS 인덱스 로드
        self._load_faiss_indices(embeddings_path)
        
        print(f"✅ 모든 분리된 임베딩 로드 완료!")
        
    def _load_faiss_indices(self, embeddings_path: Path):
        """FAISS 인덱스들 로드"""
        print("🔍 FAISS 인덱스 로딩 중...")
        
        # 줄거리 인덱스
        plot_index_file = embeddings_path / "plot_index.faiss"
        if plot_index_file.exists():
            self.plot_index = faiss.read_index(str(plot_index_file))
            print(f"📝 줄거리 인덱스 로드: {self.plot_index.ntotal}개 벡터")
        
        # 흐름곡선 인덱스
        flow_index_file = embeddings_path / "flow_index.faiss"
        if flow_index_file.exists():
            self.flow_index = faiss.read_index(str(flow_index_file))
            print(f"📈 흐름곡선 인덱스 로드: {self.flow_index.ntotal}개 벡터")
        
        # 장르 인덱스
        genre_index_file = embeddings_path / "genre_index.faiss"
        if genre_index_file.exists():
            self.genre_index = faiss.read_index(str(genre_index_file))
            print(f"🎭 장르 인덱스 로드: {self.genre_index.ntotal}개 벡터")
    
    def encode_query_text(self, query: str) -> np.ndarray:
        """쿼리 텍스트를 파인튜닝된 모델로 임베딩"""
        embedding = self.sbert_model.encode([query], normalize_embeddings=True)[0]
        return embedding.astype(np.float32)
    
    def encode_query_flow(self, flow_description: str) -> np.ndarray:
        """
        흐름 관련 쿼리를 흐름곡선 벡터로 변환
        
        간단한 키워드 매핑으로 시작:
        - "긴장감", "스릴" -> 높은 값
        - "잔잔한", "평온" -> 낮은 값  
        - "변화무쌍" -> 변동이 큰 패턴
        """
        # 기본 흐름곡선 (중간값 5)
        base_flow = [5] * 20
        
        # 키워드 기반 조정
        if any(word in flow_description for word in ["긴장", "스릴", "액션"]):
            base_flow = [7, 8, 9, 8, 7, 8, 9, 8, 7, 8, 9, 7, 8, 9, 8, 7, 8, 9, 8, 7]
        elif any(word in flow_description for word in ["잔잔", "평온", "드라마"]):
            base_flow = [4, 3, 4, 3, 4, 3, 4, 5, 4, 3, 4, 3, 4, 5, 4, 3, 4, 3, 4, 3]
        elif any(word in flow_description for word in ["변화", "반전", "롤러코스터"]):
            base_flow = [2, 8, 3, 9, 1, 7, 4, 8, 2, 9, 3, 7, 1, 8, 4, 9, 2, 7, 3, 8]
        
        # 384차원으로 확장
        if len(base_flow) < 384:
            repeat_times = 384 // len(base_flow) + 1
            extended_flow = (base_flow * repeat_times)[:384]
        else:
            extended_flow = base_flow[:384]
        
        # 정규화
        normalized = np.array(extended_flow, dtype=float)
        normalized = (normalized - 5.0) / 5.0  # [-1, 1] 범위로
        
        # L2 정규화
        norm = np.linalg.norm(normalized)
        if norm > 0:
            normalized = normalized / norm
            
        return normalized.astype(np.float32)
    
    def encode_query_genre(self, genre_description: str) -> np.ndarray:
        """
        장르 관련 쿼리를 장르 벡터로 변환
        
        키워드 기반으로 장르 점수 할당
        """
        if not self.genre_info:
            # 기본 장르 벡터
            return np.zeros(384, dtype=np.float32)
        
        genre_types = self.genre_info['genre_types']
        genre_scores = {}
        
        # 키워드 매핑
        genre_keywords = {
            'action': ['액션', '전투', '싸움', '폭발', '추격'],
            'comedy': ['코미디', '웃긴', '유머', '재미', '개그'],
            'drama': ['드라마', '감동', '인간', '삶', '가족'],
            'horror': ['공포', '무서운', '좀비', '귀신', '스릴러'],
            'romance': ['로맨스', '사랑', '연애', '결혼', '로맨틱'],
            'sci_fi': ['SF', '미래', '우주', '로봇', '인공지능', '과학'],
            'thriller': ['스릴러', '긴장', '미스터리', '범죄', '수사']
        }
        
        # 쿼리에서 장르 추출
        for genre, keywords in genre_keywords.items():
            if genre in genre_types:
                score = 0
                for keyword in keywords:
                    if keyword in genre_description:
                        score += 2
                genre_scores[genre] = min(score, 10)  # 최대 10점
        
        # 기본 장르 벡터 생성
        genre_vector = []
        for genre in genre_types:
            score = genre_scores.get(genre, 0)
            genre_vector.append(score)
        
        # 384차원으로 확장
        if len(genre_vector) < 384:
            genre_vector.extend([0] * (384 - len(genre_vector)))
        else:
            genre_vector = genre_vector[:384]
        
        # 정규화
        normalized = np.array(genre_vector, dtype=float)
        normalized = (normalized - 5.0) / 5.0  # [-1, 1] 범위로
        
        # L2 정규화
        norm = np.linalg.norm(normalized)
        if norm > 0:
            normalized = normalized / norm
            
        return normalized.astype(np.float32)
    
    def search_with_flexible_weights(
        self, 
        query: str,
        w_plot: float = None,
        w_flow: float = None, 
        w_genre: float = None,
        top_k: int = 20,
        flow_hint: str = "",
        genre_hint: str = ""
    ) -> List[Dict[str, Any]]:
        """
        사용자 지정 가중치로 멀티모달 검색 수행
        
        Args:
            query: 기본 검색 쿼리
            w_plot: 줄거리 가중치 (0~1)
            w_flow: 흐름곡선 가중치 (0~1) 
            w_genre: 장르 가중치 (0~1)
            top_k: 반환할 결과 수
            flow_hint: 흐름 관련 힌트 ("긴장감 있는", "잔잔한" 등)
            genre_hint: 장르 관련 힌트 ("액션", "로맨스" 등)
        """
        
        if not all([self.plot_embeddings is not None, 
                   self.flow_embeddings is not None,
                   self.genre_embeddings is not None]):
            raise ValueError("분리된 임베딩들이 로드되지 않았습니다. load_separated_embeddings()를 먼저 실행하세요.")
        
        # 가중치 정규화
        if all(w is None for w in [w_plot, w_flow, w_genre]):
            w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        else:
            # 입력된 가중치들의 합이 1이 되도록 정규화
            weights = np.array([w_plot or 0, w_flow or 0, w_genre or 0])
            if weights.sum() > 0:
                weights = weights / weights.sum()
                w_plot, w_flow, w_genre = weights
            else:
                w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']
        
        print(f"🔍 검색 실행: plot={w_plot:.2f}, flow={w_flow:.2f}, genre={w_genre:.2f}")
        
        # 각 모달리티별 쿼리 임베딩
        query_plot = self.encode_query_text(query)
        query_flow = self.encode_query_flow(flow_hint or query)
        query_genre = self.encode_query_genre(genre_hint or query)
        
        # 각 모달리티별 유사도 계산
        plot_scores = np.dot(self.plot_embeddings, query_plot)
        flow_scores = np.dot(self.flow_embeddings, query_flow) 
        genre_scores = np.dot(self.genre_embeddings, query_genre)
        
        # 가중합 계산
        final_scores = (w_plot * plot_scores + 
                       w_flow * flow_scores + 
                       w_genre * genre_scores)
        
        # 상위 K개 결과 선택
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # 결과 포맷팅
        results = []
        for i, idx in enumerate(top_indices):
            movie_meta = self.movie_metadata[idx]
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
    
    def print_search_results(self, results: List[Dict[str, Any]], query: str):
        """검색 결과를 보기 좋게 출력"""
        print(f"\n🔎 검색 결과: '{query}'")
        print("=" * 80)
        
        for result in results:
            print(f"\n🎬 순위 {result['rank']}: {result['title']} ({result['year']})")
            print(f"   감독: {result['director']}")
            print(f"   🔢 최종 점수: {result['final_score']:.4f}")
            print(f"   📊 세부 점수: 줄거리={result['component_scores']['plot']:.3f}, "
                  f"흐름={result['component_scores']['flow']:.3f}, "
                  f"장르={result['component_scores']['genre']:.3f}")
            print(f"   ⚖️ 사용된 가중치: 줄거리={result['weights_used']['plot']:.2f}, "
                  f"흐름={result['weights_used']['flow']:.2f}, "
                  f"장르={result['weights_used']['genre']:.2f}")


def demonstrate_flexible_search():
    """유연한 검색 시스템 데모"""
    print("🎭 파인튜닝된 분리 임베딩 검색 시스템 데모")
    print("=" * 60)
    
    # 시스템 초기화
    system = FlexibleMultimodalSystem()
    
    # 분리된 임베딩 로드
    system.load_separated_embeddings()
    
    # 테스트 쿼리들
    test_queries = [
        {
            'query': '꿈과 현실을 오가는 영화',
            'flow_hint': '심리적 긴장감',
            'genre_hint': 'SF 스릴러',
            'weights': [
                {'plot': 0.8, 'flow': 0.1, 'genre': 0.1, 'name': '줄거리 중심'},
                {'plot': 0.4, 'flow': 0.4, 'genre': 0.2, 'name': '흐름 중시'},
                {'plot': 0.3, 'flow': 0.2, 'genre': 0.5, 'name': '장르 중심'}
            ]
        },
        {
            'query': '좀비 아포칼립스 생존',
            'flow_hint': '긴장감 넘치는 액션',
            'genre_hint': '공포 액션',
            'weights': [
                {'plot': 0.6, 'flow': 0.3, 'genre': 0.1, 'name': '기본 균형'},
                {'plot': 0.2, 'flow': 0.6, 'genre': 0.2, 'name': '흐름 중심'}
            ]
        }
    ]
    
    # 각 쿼리별 테스트
    for test in test_queries:
        for weight_config in test['weights']:
            print(f"\n{'='*80}")
            print(f"🔍 쿼리: {test['query']}")
            print(f"⚖️ 가중치 설정: {weight_config['name']}")
            print(f"📝 흐름 힌트: {test['flow_hint']}")
            print(f"🎭 장르 힌트: {test['genre_hint']}")
            
            results = system.search_with_flexible_weights(
                query=test['query'],
                w_plot=weight_config['plot'],
                w_flow=weight_config['flow'],
                w_genre=weight_config['genre'],
                top_k=10,
                flow_hint=test['flow_hint'],
                genre_hint=test['genre_hint']
            )
            
            system.print_search_results(results[:5], test['query'])  # 상위 5개만 출력
            
            print(f"\n💡 분석: {weight_config['name']} 설정으로 검색")
            time.sleep(1)  # 출력 간격


def interactive_search():
    """대화형 검색 인터페이스"""
    print("🎮 대화형 유연 검색 시스템")
    print("=" * 60)
    
    system = FlexibleMultimodalSystem()
    system.load_separated_embeddings()
    
    print("\n📖 사용법:")
    print("- 검색어 입력 (예: '꿈과 현실을 오가는 영화')")
    print("- 가중치는 0-1 사이 값으로 입력 (빈칸시 기본값 사용)")
    print("- 'quit' 입력시 종료")
    
    while True:
        try:
            print(f"\n{'='*50}")
            query = input("🔍 검색어를 입력하세요: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 검색을 종료합니다.")
                break
                
            if not query:
                print("❌ 검색어를 입력해주세요.")
                continue
            
            # 가중치 입력
            print("\n⚖️ 가중치 설정 (엔터키로 기본값 사용):")
            
            w_plot_input = input(f"  📝 줄거리 가중치 [기본값: {system.default_weights['plot']}]: ").strip()
            w_plot = float(w_plot_input) if w_plot_input else system.default_weights['plot']
            
            w_flow_input = input(f"  📈 흐름곡선 가중치 [기본값: {system.default_weights['flow']}]: ").strip()
            w_flow = float(w_flow_input) if w_flow_input else system.default_weights['flow']
            
            w_genre_input = input(f"  🎭 장르 가중치 [기본값: {system.default_weights['genre']}]: ").strip()
            w_genre = float(w_genre_input) if w_genre_input else system.default_weights['genre']
            
            # 힌트 입력
            flow_hint = input("  📈 흐름 힌트 (선택사항): ").strip()
            genre_hint = input("  🎭 장르 힌트 (선택사항): ").strip()
            
            # 검색 실행
            results = system.search_with_flexible_weights(
                query=query,
                w_plot=w_plot,
                w_flow=w_flow, 
                w_genre=w_genre,
                top_k=15,
                flow_hint=flow_hint,
                genre_hint=genre_hint
            )
            
            system.print_search_results(results[:10], query)
            
        except KeyboardInterrupt:
            print(f"\n\n👋 검색을 종료합니다.")
            break
        except ValueError as e:
            print(f"❌ 가중치 입력 오류: {e}")
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demonstrate_flexible_search()
    elif len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_search()
    else:
        print("🎭 파인튜닝된 분리 임베딩 검색 시스템")
        print("=" * 60)
        print("📖 사용법:")
        print("  python flexible_multimodal_system.py --demo       # 데모 실행")
        print("  python flexible_multimodal_system.py --interactive # 대화형 실행")
        print("\n💡 또는 코드에서 직접 사용:")
        print("  system = FlexibleMultimodalSystem()")
        print("  system.load_separated_embeddings()")
        print("  results = system.search_with_flexible_weights(query='원하는 검색어', w_plot=0.6, w_flow=0.3, w_genre=0.1)")
        
        # 간단한 테스트 실행
        try:
            system = FlexibleMultimodalSystem()
            system.load_separated_embeddings()
            
            print(f"\n✅ 시스템 로드 완료!")
            print(f"📊 로드된 데이터: {len(system.movie_metadata)}개 영화")
            print(f"📝 줄거리 임베딩: {system.plot_embeddings.shape}")
            print(f"📈 흐름곡선 임베딩: {system.flow_embeddings.shape}")
            print(f"🎭 장르 임베딩: {system.genre_embeddings.shape}")
            
            # 간단한 검색 예시
            print(f"\n🔍 간단한 검색 예시:")
            results = system.search_with_flexible_weights(
                query="꿈과 현실을 오가는 영화",
                w_plot=0.7, w_flow=0.2, w_genre=0.1,
                top_k=3
            )
            
            for result in results:
                print(f"  🎬 {result['title']} (점수: {result['final_score']:.3f})")
                
        except Exception as e:
            print(f"❌ 시스템 로드 실패: {e}")
            print("💡 먼저 generate_separated_embeddings.py를 실행해서 분리된 임베딩을 생성하세요.")
        
        # 청킹된 데이터를 영화별로 그룹화
        print("📋 기존 청킹 데이터 로딩...")
        movie_chunks = {}
        with open(chunk_metadata_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                meta = json.loads(line)
                movie_id = (meta['title'], meta['year'])
                if movie_id not in movie_chunks:
                    movie_chunks[movie_id] = []
                movie_chunks[movie_id].append((i, chunk_embeddings[i]))
        
        print(f"📊 총 {len(movies)}개 영화 발견, {len(movie_chunks)}개 영화의 청킹 데이터 사용 가능")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plot_embeddings = []
        flow_embeddings = []
        genre_embeddings = []
        movie_metadata = []
        
        print("\n🔄 분리된 임베딩 생성 중...")
        start_time = time.time()
        chunk_used = 0
        new_generated = 0
        
        for i, movie in enumerate(movies):
            try:
                if (i + 1) % 50 == 0 or i + 1 == len(movies):
                    elapsed = time.time() - start_time
                    print(f"   [{i+1:3d}/{len(movies)}] {movie.get('title', 'Unknown')} ({elapsed:.1f}초)")
                
                # 1. 줄거리 임베딩 (기존 청킹 데이터 우선 사용)
                movie_id = (movie.get('title', 'Unknown'), movie.get('year', 'Unknown'))
                if movie_id in movie_chunks:
                    # 기존 청킹된 임베딩들의 평균 사용
                    chunk_embeds = [emb for _, emb in movie_chunks[movie_id]]
                    plot_emb = np.mean(chunk_embeds, axis=0).astype('float32')
                    chunk_used += 1
                else:
                    # 청킹 데이터가 없는 경우에만 새로 생성
                    plot_emb = self._get_plot_embedding(movie['plot'])
                    new_generated += 1
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
        print(f"   📁 줄거리: {output_path / 'plot_embeddings.npy'} ({plot_array.shape})")
        print(f"   📁 흐름곡선: {output_path / 'flow_embeddings.npy'} ({flow_array.shape})")
        print(f"   📁 장르: {output_path / 'genre_embeddings.npy'} ({genre_array.shape})")
        print(f"   📁 메타데이터: {output_path / 'flexible_metadata.jsonl'}")
        print(f"\\n📊 줄거리 임베딩 통계:")
        print(f"   • 기존 청킹 데이터 사용: {chunk_used}개")
        print(f"   • 새로 생성: {new_generated}개")
        print(f"   • 총 처리: {len(plot_embeddings)}개")
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
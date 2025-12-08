#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
영화 유사도 기반 추천 시스템
멀티모달 임베딩 (줄거리 + 흐름곡선 + 장르)을 사용한 유사 영화 찾기
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

class MovieSimilarityRecommender:
    def __init__(self, data_dir: str = "data"):
        """영화 유사도 추천 시스템 초기화"""
        self.data_dir = Path(data_dir)
        self.index = None
        self.metadata = None
        self.movie_id_to_index = {}
        self.index_to_movie_id = {}
        
    def load_data(self):
        """멀티모달 임베딩과 메타데이터 로드"""
        print("📁 멀티모달 데이터 로딩 중...")
        
        # 임베딩 로드
        embedding_path = self.data_dir / 'multimodal_embeddings.npy'
        if not embedding_path.exists():
            raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {embedding_path}")
        
        embeddings = np.load(embedding_path)
        print(f"   임베딩 형태: {embeddings.shape}")
        
        # 메타데이터 로드
        metadata_path = self.data_dir / 'multimodal_metadata.jsonl'
        if not metadata_path.exists():
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        print(f"   영화 수: {len(self.metadata)}")
        
        # 영화 ID 매핑 생성
        for i, meta in enumerate(self.metadata):
            movie_id = f"{meta['title']}_{meta['year']}"
            self.movie_id_to_index[movie_id] = i
            self.index_to_movie_id[i] = movie_id
        
        return embeddings
    
    def build_index(self, save_index: bool = True):
        """FAISS 인덱스 생성"""
        print("🔨 FAISS 인덱스 생성 중...")
        
        embeddings = self.load_data()
        
        # FAISS 인덱스 생성 (내적 기반 - 정규화된 벡터에서 코사인 유사도)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # 임베딩 추가
        self.index.add(embeddings.astype('float32'))
        
        print(f"   인덱스 생성 완료: {self.index.ntotal}개 벡터")
        
        # 인덱스 저장
        if save_index:
            index_path = self.data_dir / 'multimodal_index.faiss'
            faiss.write_index(self.index, str(index_path))
            print(f"   인덱스 저장: {index_path}")
        
        return self.index
    
    def load_index(self):
        """저장된 FAISS 인덱스 로드"""
        index_path = self.data_dir / 'multimodal_index.faiss'
        
        if index_path.exists():
            print("📂 저장된 인덱스 로딩 중...")
            self.index = faiss.read_index(str(index_path))
            self.load_data()  # 메타데이터만 로드
            print(f"   인덱스 로드 완료: {self.index.ntotal}개 벡터")
        else:
            print("⚠️ 저장된 인덱스가 없습니다. 새로 생성합니다.")
            self.build_index()
    
    def find_movie_by_title(self, title: str, year: Optional[str] = None) -> Optional[int]:
        """영화 제목으로 인덱스 찾기"""
        if year:
            movie_id = f"{title}_{year}"
            return self.movie_id_to_index.get(movie_id)
        
        # 연도 없이 검색 (첫 번째 매치)
        for movie_id, index in self.movie_id_to_index.items():
            if title.lower() in movie_id.lower():
                return index
        
        return None
    
    def get_similar_movies(self, movie_title: str, movie_year: Optional[str] = None, 
                          top_k: int = 10, exclude_self: bool = True) -> List[Dict[str, Any]]:
        """유사한 영화 찾기"""
        if self.index is None:
            self.load_index()
        
        # 대상 영화 찾기
        movie_index = self.find_movie_by_title(movie_title, movie_year)
        if movie_index is None:
            available_movies = [meta['title'] for meta in self.metadata[:10]]
            raise ValueError(f"영화를 찾을 수 없습니다: '{movie_title}' (연도: {movie_year})\n"
                           f"사용 가능한 영화 예시: {available_movies}")
        
        target_movie = self.metadata[movie_index]
        print(f"🎯 대상 영화: {target_movie['title']} ({target_movie['year']})")
        
        # 해당 영화의 임베딩으로 유사도 검색
        # top_k + 1: 자기 자신도 포함되므로
        search_k = top_k + 1 if exclude_self else top_k
        
        # 단일 벡터 검색을 위해 임베딩 추출
        embeddings = np.load(self.data_dir / 'multimodal_embeddings.npy')
        query_vector = embeddings[movie_index:movie_index+1].astype('float32')
        
        similarities, indices = self.index.search(query_vector, search_k)
        
        # 결과 정리
        similar_movies = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            # 자기 자신 제외
            if exclude_self and idx == movie_index:
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
    
    def display_similar_movies(self, similar_movies: List[Dict[str, Any]], target_title: str):
        """유사 영화 결과 출력"""
        print(f"\n🎬 '{target_title}'와 유사한 영화들")
        print("=" * 80)
        
        if not similar_movies:
            print("❌ 유사한 영화를 찾을 수 없습니다.")
            return
        
        for movie in similar_movies:
            print(f"\n{movie['rank']:2d}. {movie['title']} ({movie['year']})")
            print(f"    감독: {movie['director']}")
            print(f"    유사도: {movie['similarity_score']:.3f}")
            
            # 장르 정보
            genres = movie['genres']
            if genres:
                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                genre_str = ", ".join([f"{genre}({score})" for genre, score in top_genres])
                print(f"    주요 장르: {genre_str}")
            
            # 흐름곡선 요약
            flow = movie['flow_curve']
            if flow:
                avg_tension = sum(flow) / len(flow)
                max_tension = max(flow)
                print(f"    긴장도: 평균 {avg_tension:.1f}, 최대 {max_tension}")
            
            # 줄거리 미리보기
            print(f"    줄거리: {movie['plot']}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="영화 유사도 추천 시스템")
    parser.add_argument('--build', action='store_true', help='FAISS 인덱스 새로 생성')
    parser.add_argument('--movie', type=str, help='유사 영화를 찾을 영화 제목')
    parser.add_argument('--year', type=str, help='영화 연도 (선택사항)')
    parser.add_argument('--top_k', type=int, default=10, help='추천할 영화 수')
    parser.add_argument('--data_dir', type=str, default='data', help='데이터 디렉토리')
    
    args = parser.parse_args()
    
    try:
        # 추천 시스템 초기화
        recommender = MovieSimilarityRecommender(args.data_dir)
        
        # 인덱스 빌드 또는 로드
        if args.build:
            recommender.build_index()
        else:
            recommender.load_index()
        
        # 대화형 모드 또는 단일 검색
        if args.movie:
            # 단일 영화 검색
            similar_movies = recommender.get_similar_movies(
                args.movie, args.year, args.top_k
            )
            recommender.display_similar_movies(similar_movies, args.movie)
        else:
            # 대화형 모드 선택
            print("\n🎬 영화 유사도 추천 시스템")
            print("=" * 50)
            print("1. 기존 추천 (멀티모달 고정 가중치)")
            print("2. 가중치 조절 추천 (분리 임베딩)")
            print("=" * 50)
            
            mode = input("추천 모드를 선택하세요 (1 또는 2): ").strip()
            
            if mode == "2":
                weighted_similarity_search(args.data_dir)
            else:
                basic_similarity_search(recommender)
                
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        import traceback
        traceback.print_exc()

def basic_similarity_search(recommender):
    """기존 유사도 검색"""
    print("\n🎬 기존 영화 유사도 추천")
    print("=" * 50)
    print("• 영화 제목을 입력하면 유사한 영화를 추천해드립니다")
    print("• 'quit' 입력시 종료")
    print("=" * 50)
    
    while True:
        try:
            movie_title = input("\n🔍 영화 제목을 입력하세요 > ").strip()
            
            if movie_title.lower() in ['quit', 'exit', '종료']:
                print("👋 시스템을 종료합니다.")
                break
            
            if not movie_title:
                print("❌ 영화 제목을 입력해주세요.")
                continue
            
            # 유사 영화 검색
            similar_movies = recommender.get_similar_movies(movie_title, top_k=10)
            recommender.display_similar_movies(similar_movies, movie_title)
            
        except ValueError as e:
            print(f"❌ {e}")
        except KeyboardInterrupt:
            print("\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

def weighted_similarity_search(data_dir):
    """가중치 조절 유사도 검색 (OpenAI 임베딩 기반)"""
    from weighted_search_utils_openai import get_openai_weighted_helper
    
    print("\n🎭 가중치 조절 영화 유사도 추천 (OpenAI)")
    print("=" * 50)
    print("• 영화 제목을 입력하면 유사한 영화를 추천해드립니다")
    print("• 실시간 가중치 조절 (plot/flow/genre)")
    print("• OpenAI 임베딩 (1536차원) 사용")
    print("• 'quit' 입력시 종료")
    print("=" * 50)
    
    # OpenAI 가중치 헬퍼 초기화
    helper = get_openai_weighted_helper()
    if not helper.load_separated_embeddings():
        print("❌ OpenAI 분리된 임베딩을 로드할 수 없습니다.")
        print("먼저 python create_separated_embeddings_openai.py 를 실행하세요.")
        return
    
    print("✅ OpenAI 가중치 조절 시스템 로드 완료!")
    
    while True:
        try:
            print("\n" + "="*50)
            movie_title = input("🔍 영화 제목을 입력하세요: ").strip()
            
            if movie_title.lower() in ['quit', 'exit', '종료']:
                print("👋 시스템을 종료합니다.")
                break
                
            if not movie_title:
                print("❌ 영화 제목을 입력해주세요.")
                continue
            
            # 가중치 입력 (선택사항)
            print("\n⚖️ 가중치 설정 (엔터키로 기본값 사용):")
            w_plot_input = input("  📝 줄거리 가중치 [기본값: 0.6]: ").strip()
            w_flow_input = input("  📈 흐름곡선 가중치 [기본값: 0.3]: ").strip()
            w_genre_input = input("  🎭 장르 가중치 [기본값: 0.1]: ").strip()
            
            # 가중치 파싱
            w_plot = float(w_plot_input) if w_plot_input else None
            w_flow = float(w_flow_input) if w_flow_input else None
            w_genre = float(w_genre_input) if w_genre_input else None
            
            # 유사 영화 검색
            results = helper.find_similar_movies_weighted(
                target_movie_title=movie_title,
                w_plot=w_plot,
                w_flow=w_flow, 
                w_genre=w_genre,
                top_k=10
            )
            
            if not results:
                print("❌ 해당 영화를 찾을 수 없거나 유사한 영화가 없습니다.")
                continue
            
            # 결과 출력
            print(f"\n🔎 '{movie_title}'와 유사한 영화들")
            print("="*80)
            
            for movie in results:
                print(f"\n🎬 순위 {movie['rank']}: {movie['title']} ({movie['year']})")
                print(f"   감독: {movie['director']}")
                print(f"   🔢 유사도: {movie['similarity_score']:.4f}")
                comp_scores = movie['component_scores']
                print(f"   📊 세부 점수: 줄거리={comp_scores['plot']:.3f}, "
                      f"흐름={comp_scores['flow']:.3f}, 장르={comp_scores['genre']:.3f}")
                
                # 장르 정보 (상위 3개)
                genres = movie.get('genres', {})
                if genres:
                    top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                    genre_str = ", ".join([f"{genre}({score})" for genre, score in top_genres])
                    print(f"   🎭 주요 장르: {genre_str}")
            
            weights_used = results[0]['weights_used']
            print(f"\n💡 사용된 가중치: 줄거리={weights_used['plot']:.2f}, "
                  f"흐름={weights_used['flow']:.2f}, 장르={weights_used['genre']:.2f}")
                    
        except ValueError as e:
            print(f"❌ 가중치 오류: {e}. 0.0~1.0 사이의 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 검색 실패: {e}")

if __name__ == "__main__":
    main()
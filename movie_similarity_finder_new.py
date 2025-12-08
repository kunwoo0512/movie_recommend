#!/usr/bin/env python3#!/usr/bin/env python3

# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-

""""""

가중치 기반 영화 유사도 추천 시스템영화 유사도 기반 추천 시스템 (파인튜닝된 분리 임베딩)

줄거리(청킹) + 흐름곡선 + 장르 임베딩을 사용한 유사 영화 찾기분리된 줄거리 + 흐름곡선 + 장르 임베딩을 사용한 유사 영화 찾기

"""가중치 조절 가능

"""

import argparse

from weighted_similarity_calculator import get_weighted_calculatorimport json

import numpy as np

class MovieSimilarityRecommender:import faiss

    def __init__(self, data_dir: str = "data"):from pathlib import Path

        """가중치 기반 영화 유사도 추천 시스템 초기화"""from typing import List, Dict, Any, Optional

        self.data_dir = data_dirimport argparse

        self.calculator = get_weighted_calculator()

        self.initialized = False# 새로운 분리 임베딩 시스템 임포트

        from flexible_multimodal_system_new import FlexibleMultimodalSystem

    def load_data(self):

        """데이터 로드"""class MovieSimilarityRecommender:

        print("📁 가중치 기반 추천 시스템 로딩 중...")    def __init__(self, data_dir: str = "data"):

                """영화 유사도 추천 시스템 초기화"""

        if self.calculator.load_all_data():        self.data_dir = Path(data_dir)

            self.initialized = True        

            print("✅ 추천 시스템 초기화 완료!")        # 새로운 분리 임베딩 시스템 사용

        else:        self.flexible_system = FlexibleMultimodalSystem()

            raise RuntimeError("데이터 로드 실패")        

            # 기본 가중치 설정

    def get_similar_movies(self, movie_title: str, movie_year: str = None,        self.default_weights = {

                          w_plot: float = 0.65, w_flow: float = 0.25, w_genre: float = 0.10,            'plot': 0.6,

                          top_k: int = 10) -> list:            'flow': 0.3,

        """가중치를 적용한 유사 영화 검색"""            'genre': 0.1

        if not self.initialized:        }

            self.load_data()        

            def load_data(self):

        # 연도가 주어진 경우 제목에 포함        """분리된 임베딩과 메타데이터 로드"""

        search_title = movie_title        print("📁 파인튜닝된 분리 임베딩 로딩 중...")

        if movie_year:        

            search_title = f"{movie_title} ({movie_year})"        # 분리된 임베딩 로드

                self.flexible_system.load_separated_embeddings("data/separated_embeddings")

        return self.calculator.calculate_weighted_similarity(        

            movie_title=search_title,        print(f"   📝 줄거리 임베딩: {self.flexible_system.plot_embeddings.shape}")

            w_plot=w_plot,        print(f"   📈 흐름곡선 임베딩: {self.flexible_system.flow_embeddings.shape}")

            w_flow=w_flow,        print(f"   🎭 장르 임베딩: {self.flexible_system.genre_embeddings.shape}")

            w_genre=w_genre,        print(f"   🎬 영화 수: {len(self.flexible_system.movie_metadata)}")

            top_k=top_k        

        )        print("✅ 모든 데이터 로드 완료!")

            

    def display_similar_movies(self, similar_movies: list, target_title: str):    def find_movie_by_title(self, title: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:

        """유사 영화 결과 출력"""        """제목으로 영화 찾기"""

        print(f"\n🎬 '{target_title}'와 유사한 영화들")        if not self.flexible_system.movie_metadata:

        print("=" * 90)            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행하세요.")

                

        if not similar_movies:        # 제목 정규화 (대소문자 구분 없이)

            print("❌ 유사한 영화를 찾을 수 없습니다.")        title_lower = title.lower()

            return        

                for i, meta in enumerate(self.flexible_system.movie_metadata):

        for movie in similar_movies:            if title_lower in meta['title'].lower():

            print(f"\n{movie['rank']:2d}. {movie['title']} ({movie['year']})")                if year is None or meta.get('year') == year:

            print(f"    감독: {movie['director']}")                    return {

            print(f"    🔢 최종 유사도: {movie['similarity_score']:.4f}")                        'index': i,

                                    'movie_id': meta['movie_id'],

            # 세부 점수                        'title': meta['title'],

            comp = movie['component_scores']                        'year': meta.get('year', ''),

            weights = movie['weights_used']                        'director': meta.get('director', '')

            print(f"    📊 세부 점수: 줄거리={comp['plot']:.3f}(×{weights['plot']:.2f}), "                    }

                  f"흐름={comp['flow']:.3f}(×{weights['flow']:.2f}), "        

                  f"장르={comp['genre']:.3f}(×{weights['genre']:.2f})")        return None

                

            # 장르 정보    def find_similar_movies_flexible(

            genres = movie.get('genres', {})        self, 

            if genres:        movie_title: str, 

                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]        top_k: int = 10,

                genre_str = ", ".join([f"{genre}({score})" for genre, score in top_genres])        w_plot: float = None,

                print(f"    🎭 주요 장르: {genre_str}")        w_flow: float = None,

                    w_genre: float = None,

            # 줄거리 미리보기        year: Optional[str] = None

            if movie.get('plot'):    ) -> List[Dict[str, Any]]:

                print(f"    📝 줄거리: {movie['plot']}")        """

        가중치 조절 가능한 유사 영화 찾기

def interactive_mode(recommender):        

    """대화형 모드"""        Args:

    print("\n🎬 가중치 조절 영화 유사도 추천 시스템")            movie_title: 기준 영화 제목

    print("=" * 70)            top_k: 반환할 결과 수

    print("• 영화 제목을 입력하면 유사한 영화를 추천해드립니다")            w_plot: 줄거리 가중치

    print("• 줄거리, 흐름곡선, 장르의 가중치를 조절할 수 있습니다")            w_flow: 흐름곡선 가중치  

    print("• 'quit' 또는 'exit' 입력시 종료")            w_genre: 장르 가중치

    print("=" * 70)            year: 영화 연도 (선택사항)

            """

    while True:        

        try:        # 기준 영화 찾기

            print("\n" + "="*50)        base_movie = self.find_movie_by_title(movie_title, year)

            movie_title = input("🔍 영화 제목을 입력하세요: ").strip()        if not base_movie:

                        raise ValueError(f"영화를 찾을 수 없습니다: {movie_title}")

            if movie_title.lower() in ['quit', 'exit', '종료']:        

                print("👋 시스템을 종료합니다.")        base_index = base_movie['index']

                break        print(f"🎬 기준 영화: {base_movie['title']} ({base_movie['year']})")

                        

            if not movie_title:        # 가중치 정규화

                print("❌ 영화 제목을 입력해주세요.")        if all(w is None for w in [w_plot, w_flow, w_genre]):

                continue            w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']

                    else:

            # 가중치 입력            weights = np.array([w_plot or 0, w_flow or 0, w_genre or 0])

            print("\n⚖️ 가중치 설정 (엔터키로 기본값 사용):")            if weights.sum() > 0:

            w_plot_input = input("  📝 줄거리 가중치 [기본값: 0.65]: ").strip()                weights = weights / weights.sum()

            w_flow_input = input("  📈 흐름곡선 가중치 [기본값: 0.25]: ").strip()                w_plot, w_flow, w_genre = weights

            w_genre_input = input("  🎭 장르 가중치 [기본값: 0.10]: ").strip()            else:

                            w_plot, w_flow, w_genre = self.default_weights['plot'], self.default_weights['flow'], self.default_weights['genre']

            # 가중치 파싱        

            try:        print(f"⚖️ 사용된 가중치: plot={w_plot:.2f}, flow={w_flow:.2f}, genre={w_genre:.2f}")

                w_plot = float(w_plot_input) if w_plot_input else 0.65        

                w_flow = float(w_flow_input) if w_flow_input else 0.25        # 기준 영화의 각 모달리티별 임베딩

                w_genre = float(w_genre_input) if w_genre_input else 0.10        base_plot = self.flexible_system.plot_embeddings[base_index]

                        base_flow = self.flexible_system.flow_embeddings[base_index]

                # 가중치 유효성 검사        base_genre = self.flexible_system.genre_embeddings[base_index]

                if w_plot < 0 or w_flow < 0 or w_genre < 0:        

                    print("❌ 가중치는 0 이상이어야 합니다.")        # 모든 영화와의 각 모달리티별 유사도 계산

                    continue        plot_similarities = np.dot(self.flexible_system.plot_embeddings, base_plot)

                            flow_similarities = np.dot(self.flexible_system.flow_embeddings, base_flow)

                total_weight = w_plot + w_flow + w_genre        genre_similarities = np.dot(self.flexible_system.genre_embeddings, base_genre)

                if total_weight == 0:        

                    print("❌ 최소 하나의 가중치는 0보다 커야 합니다.")        # 가중합 계산

                    continue        final_similarities = (w_plot * plot_similarities + 

                                                 w_flow * flow_similarities + 

            except ValueError:                             w_genre * genre_similarities)

                print("❌ 가중치는 숫자여야 합니다.")        

                continue        # 기준 영화 제외하고 상위 K개 선택

                    final_similarities[base_index] = -999  # 자기 자신 제외

            # 유사 영화 검색        top_indices = np.argsort(final_similarities)[::-1][:top_k]

            print(f"\n🔍 검색 중... (가중치: {w_plot:.2f}, {w_flow:.2f}, {w_genre:.2f})")        

                    # 결과 포맷팅

            similar_movies = recommender.get_similar_movies(        results = []

                movie_title=movie_title,        for i, idx in enumerate(top_indices):

                w_plot=w_plot,            meta = self.flexible_system.movie_metadata[idx]

                w_flow=w_flow,            result = {

                w_genre=w_genre,                'rank': i + 1,

                top_k=10                'movie_id': meta['movie_id'],

            )                'title': meta['title'],

                            'year': meta.get('year', ''),

            # 결과 출력                'director': meta.get('director', ''),

            recommender.display_similar_movies(similar_movies, movie_title)                'final_score': float(final_similarities[idx]),

                            'component_scores': {

        except ValueError as e:                    'plot': float(plot_similarities[idx]),

            print(f"❌ {e}")                    'flow': float(flow_similarities[idx]), 

        except KeyboardInterrupt:                    'genre': float(genre_similarities[idx])

            print("\n👋 시스템을 종료합니다.")                },

            break                'weights_used': {

        except Exception as e:                    'plot': w_plot,

            print(f"❌ 오류 발생: {e}")                    'flow': w_flow,

                    'genre': w_genre

def main():                }

    """메인 실행 함수"""            }

    parser = argparse.ArgumentParser(description="가중치 기반 영화 유사도 추천 시스템")            results.append(result)

    parser.add_argument('--movie', type=str, help='유사 영화를 찾을 영화 제목')        

    parser.add_argument('--year', type=str, help='영화 연도 (선택사항)')        return results

    parser.add_argument('--plot_weight', type=float, default=0.65, help='줄거리 가중치')    

    parser.add_argument('--flow_weight', type=float, default=0.25, help='흐름곡선 가중치')    def find_similar_movies(self, movie_title: str, top_k: int = 10, year: Optional[str] = None) -> List[Dict[str, Any]]:

    parser.add_argument('--genre_weight', type=float, default=0.10, help='장르 가중치')        """기본 가중치로 유사 영화 찾기 (하위 호환성)"""

    parser.add_argument('--top_k', type=int, default=10, help='추천할 영화 수')        return self.find_similar_movies_flexible(

    parser.add_argument('--data_dir', type=str, default='data', help='데이터 디렉토리')            movie_title=movie_title, 

                top_k=top_k,

    args = parser.parse_args()            year=year

            )

    try:    

        # 추천 시스템 초기화    def print_similar_movies(self, results: List[Dict[str, Any]], base_title: str):

        recommender = MovieSimilarityRecommender(args.data_dir)        """유사 영화 결과를 보기 좋게 출력"""

                print(f"\n🎯 '{base_title}'와 유사한 영화들")

        if args.movie:        print("=" * 80)

            # 단일 영화 검색        

            similar_movies = recommender.get_similar_movies(        for result in results:

                movie_title=args.movie,            print(f"\n🎬 순위 {result['rank']}: {result['title']} ({result['year']})")

                movie_year=args.year,            print(f"   감독: {result['director']}")

                w_plot=args.plot_weight,            print(f"   🔢 최종 점수: {result['final_score']:.4f}")

                w_flow=args.flow_weight,            print(f"   📊 세부 점수: 줄거리={result['component_scores']['plot']:.3f}, "

                w_genre=args.genre_weight,                  f"흐름={result['component_scores']['flow']:.3f}, "

                top_k=args.top_k                  f"장르={result['component_scores']['genre']:.3f}")

            )            print(f"   ⚖️ 사용된 가중치: 줄거리={result['weights_used']['plot']:.2f}, "

            recommender.display_similar_movies(similar_movies, args.movie)                  f"흐름={result['weights_used']['flow']:.2f}, "

        else:                  f"장르={result['weights_used']['genre']:.2f}")

            # 대화형 모드

            interactive_mode(recommender)

            def interactive_similarity_search():

    except Exception as e:    """대화형 유사 영화 찾기"""

        print(f"❌ 시스템 초기화 실패: {e}")    print("🎭 영화 유사도 기반 추천 시스템 (파인튜닝된 분리 임베딩)")

        import traceback    print("=" * 70)

        traceback.print_exc()    print("• 줄거리 + 흐름곡선 + 장르 임베딩 활용")

    print("• 가중치 조절 가능")

if __name__ == "__main__":    print("• 'quit' 또는 'exit' 입력시 종료")

    main()    print("=" * 70)
    
    try:
        # 시스템 초기화
        recommender = MovieSimilarityRecommender()
        recommender.load_data()
        
        while True:
            try:
                print(f"\n{'='*50}")
                movie_title = input("🎬 기준 영화 제목을 입력하세요: ").strip()
                
                if movie_title.lower() in ['quit', 'exit', 'q']:
                    print("👋 종료합니다.")
                    break
                    
                if not movie_title:
                    print("❌ 영화 제목을 입력해주세요.")
                    continue
                
                # 영화가 존재하는지 먼저 확인
                base_movie = recommender.find_movie_by_title(movie_title)
                if not base_movie:
                    print(f"❌ '{movie_title}'를 찾을 수 없습니다.")
                    print("💡 일부 제목만 입력해도 검색 가능합니다.")
                    continue
                
                print(f"✅ 찾은 영화: {base_movie['title']} ({base_movie['year']})")
                
                # 추천할 영화 수 입력
                top_k_input = input("📊 추천받을 영화 수 [기본값: 10]: ").strip()
                top_k = int(top_k_input) if top_k_input else 10
                
                # 가중치 입력
                print("\n⚖️ 가중치 설정 (엔터키로 기본값 사용):")
                
                w_plot_input = input(f"  📝 줄거리 가중치 [기본값: {recommender.default_weights['plot']}]: ").strip()
                w_plot = float(w_plot_input) if w_plot_input else None
                
                w_flow_input = input(f"  📈 흐름곡선 가중치 [기본값: {recommender.default_weights['flow']}]: ").strip()
                w_flow = float(w_flow_input) if w_flow_input else None
                
                w_genre_input = input(f"  🎭 장르 가중치 [기본값: {recommender.default_weights['genre']}]: ").strip()
                w_genre = float(w_genre_input) if w_genre_input else None
                
                # 유사 영화 검색
                print("\n🔍 유사 영화 검색 중...")
                results = recommender.find_similar_movies_flexible(
                    movie_title=movie_title,
                    top_k=top_k,
                    w_plot=w_plot,
                    w_flow=w_flow,
                    w_genre=w_genre
                )
                
                # 결과 출력
                recommender.print_similar_movies(results, base_movie['title'])
                
            except KeyboardInterrupt:
                print(f"\n\n👋 종료합니다.")
                break
            except ValueError as e:
                print(f"❌ 입력 오류: {e}")
            except Exception as e:
                print(f"❌ 검색 중 오류 발생: {e}")
                
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        print("💡 먼저 generate_separated_embeddings.py를 실행해서 분리된 임베딩을 생성하세요.")


def demo_similarity_search():
    """유사도 검색 데모"""
    print("🎭 영화 유사도 검색 데모")
    print("=" * 50)
    
    recommender = MovieSimilarityRecommender()
    recommender.load_data()
    
    # 데모용 영화들
    demo_movies = [
        {'title': 'Inception', 'weights': [{'plot': 0.7, 'flow': 0.2, 'genre': 0.1, 'name': '스토리 중심'}]},
        {'title': 'Matrix', 'weights': [{'plot': 0.3, 'flow': 0.4, 'genre': 0.3, 'name': '균형 잡힌 분석'}]},
        {'title': 'Avengers', 'weights': [{'plot': 0.2, 'flow': 0.6, 'genre': 0.2, 'name': '액션 흐름 중심'}]}
    ]
    
    for demo in demo_movies:
        for weight_config in demo['weights']:
            print(f"\n{'='*60}")
            print(f"🎬 기준 영화: {demo['title']}")
            print(f"⚖️ 가중치 설정: {weight_config['name']}")
            
            try:
                results = recommender.find_similar_movies_flexible(
                    movie_title=demo['title'],
                    top_k=5,
                    w_plot=weight_config['plot'],
                    w_flow=weight_config['flow'],
                    w_genre=weight_config['genre']
                )
                
                # 상위 3개만 출력
                for result in results[:3]:
                    print(f"\n   🎬 {result['title']} ({result['year']})")
                    print(f"      점수: {result['final_score']:.3f} | 줄거리: {result['component_scores']['plot']:.3f}")
                    
            except Exception as e:
                print(f"   ❌ {demo['title']} 검색 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영화 유사도 추천 시스템")
    parser.add_argument('--demo', action='store_true', help='데모 실행')
    parser.add_argument('--interactive', action='store_true', help='대화형 모드')
    parser.add_argument('--build', action='store_true', help='임베딩 빌드 (하위 호환성)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_similarity_search()
    elif args.interactive:
        interactive_similarity_search()
    elif args.build:
        print("💡 빌드는 generate_separated_embeddings.py에서 수행됩니다.")
        print("python generate_separated_embeddings.py 를 실행하세요.")
    else:
        print("🎭 영화 유사도 추천 시스템")
        print("=" * 40)
        print("📖 사용법:")
        print("  python movie_similarity_finder.py --interactive  # 대화형 모드")
        print("  python movie_similarity_finder.py --demo        # 데모 실행")
        print()
        
        # 기본 실행: 간단한 테스트
        try:
            recommender = MovieSimilarityRecommender()
            recommender.load_data()
            
            print("✅ 시스템 로드 완료!")
            print(f"📊 로드된 영화 수: {len(recommender.flexible_system.movie_metadata)}")
            
            # 간단한 예시
            print("\n🔍 간단한 검색 예시: 'Inception'과 유사한 영화")
            try:
                results = recommender.find_similar_movies('Inception', top_k=3)
                for result in results:
                    print(f"  🎬 {result['title']} (점수: {result['final_score']:.3f})")
            except Exception as e:
                print(f"  ❌ 예시 검색 실패: {e}")
                
        except Exception as e:
            print(f"❌ 시스템 로드 실패: {e}")
            print("💡 먼저 generate_separated_embeddings.py를 실행해서 분리된 임베딩을 생성하세요.")
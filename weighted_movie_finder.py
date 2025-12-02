#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Movie Similarity Recommender System

Uses chunked plot + flow curves + genre embeddings for weighted similarity calculation.
"""

import argparse
import os
import time
import re
from unified_multimodal_calculator import get_weighted_calculator

class LLMRecommendationExplainer:
    """LLM을 사용한 영화 추천 이유 설명"""
    
    def __init__(self, api_key: str = None):
        """LLM 설명기 초기화"""
        if not api_key:
            # 환경변수에서 API 키 로드 시도
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
            except ImportError:
                pass
        
        try:
            import openai
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.available = True
                print("✅ LLM 설명 기능 활성화")
            else:
                self.available = False
                print("⚠️ LLM 설명 기능 비활성화 (API 키 없음)")
        except ImportError:
            print("⚠️ LLM 설명 기능 비활성화 (openai 패키지 없음)")
            self.available = False
        except Exception as e:
            print(f"⚠️ LLM 설명 기능 비활성화: {e}")
            self.available = False
    
    def explain_recommendation(self, target_movie: dict, recommended_movie: dict, 
                              similarity_scores: dict = None, weights: dict = None) -> str:
        """추천 이유 설명 생성 (가중치와 유사도 점수 기반)"""
        if not self.available:
            return "LLM 설명 기능을 사용할 수 없습니다."
        
        try:
            # 유사도 점수 분석
            plot_score = similarity_scores.get('plot', 0) if similarity_scores else 0
            flow_score = similarity_scores.get('flow', 0) if similarity_scores else 0
            genre_score = similarity_scores.get('genre', 0) if similarity_scores else 0
            
            # 가중치 정보
            plot_weight = weights.get('plot', 0.8) if weights else 0.8
            flow_weight = weights.get('flow', 0.1) if weights else 0.1
            genre_weight = weights.get('genre', 0.1) if weights else 0.1
            
            # 가장 높은 유사도를 가진 요소 찾기
            similarity_factors = {
                'plot': plot_score,
                'flow': flow_score, 
                'genre': genre_score
            }
            
            # 가중 점수 계산
            weighted_factors = {
                'plot': plot_score * plot_weight,
                'flow': flow_score * flow_weight,
                'genre': genre_score * genre_weight
            }
            
            main_factor = max(weighted_factors, key=weighted_factors.get)
            
            prompt = self._create_focused_explanation_prompt(
                target_movie, recommended_movie, main_factor, 
                similarity_factors, weights
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 영화 추천 전문가입니다. 주어진 유사도 분석 결과를 바탕으로 간결하고 구체적인 추천 이유를 한국어로 설명해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # 토큰 수를 늘려서 텍스트가 잘리지 않도록 함
                temperature=0.3
            )
            
            explanation = response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            return f"설명 생성 중 오류: {str(e)[:50]}..."
    
    def _create_focused_explanation_prompt(self, target_movie: dict, recommended_movie: dict, 
                                         main_factor: str, similarity_scores: dict, weights: dict) -> str:
        """주요 유사도 요소에 집중한 설명 프롬프트 생성"""
        target_title = target_movie.get('title', 'Unknown')
        target_year = target_movie.get('year', 'Unknown')
        
        rec_title = recommended_movie.get('title', 'Unknown')
        rec_year = recommended_movie.get('year', 'Unknown')
        
        # 주요 유사도 요소에 따른 맞춤형 프롬프트
        if main_factor == 'plot':
            target_plot = target_movie.get('plot', '')[:200]
            rec_plot = recommended_movie.get('plot', '')[:200]
            
            prompt = f"""
원본 영화: {target_title} ({target_year})
줄거리: {target_plot}

추천 영화: {rec_title} ({rec_year})  
줄거리: {rec_plot}

줄거리 유사도: {similarity_scores.get('plot', 0):.2f}
가중치: 줄거리 {weights.get('plot', 0.8):.1f}, 흐름 {weights.get('flow', 0.1):.1f}, 장르 {weights.get('genre', 0.1):.1f}

두 영화의 줄거리에서 어떤 부분이 유사한지 구체적으로 설명해주세요. (2-3문장으로 간결하게)
"""
        
        elif main_factor == 'flow':
            prompt = f"""
원본 영화: {target_title} ({target_year})
추천 영화: {rec_title} ({rec_year})

흐름곡선 유사도: {similarity_scores.get('flow', 0):.2f}
가중치: 줄거리 {weights.get('plot', 0.8):.1f}, 흐름 {weights.get('flow', 0.1):.1f}, 장르 {weights.get('genre', 0.1):.1f}

두 영화의 스토리 전개 패턴이나 감정 흐름이 어떻게 유사한지 설명해주세요. (2-3문장으로 간결하게)
"""
        
        else:  # genre
            target_genres = list(target_movie.get('genres', {}).keys())[:3]
            rec_genres = list(recommended_movie.get('genres', {}).keys())[:3]
            
            prompt = f"""
원본 영화: {target_title} ({target_year})
장르: {', '.join(target_genres) if target_genres else '정보 없음'}

추천 영화: {rec_title} ({rec_year})
장르: {', '.join(rec_genres) if rec_genres else '정보 없음'}

장르 유사도: {similarity_scores.get('genre', 0):.2f}
가중치: 줄거리 {weights.get('plot', 0.8):.1f}, 흐름 {weights.get('flow', 0.1):.1f}, 장르 {weights.get('genre', 0.1):.1f}

두 영화의 공통 장르나 스타일이 어떻게 유사한지 설명해주세요. (2-3문장으로 간결하게)
"""
        
        return prompt
    
class MovieSimilarityRecommender:
    def __init__(self, data_dir: str = "data", enable_llm: bool = False, api_key: str = None):
        """Initialize unified multimodal movie similarity recommender"""
        self.data_dir = data_dir
        self.calculator = get_weighted_calculator()
        self.initialized = False
        
        # LLM 설명 기능 초기화
        self.enable_llm = enable_llm
        self.llm_explainer = None
        if enable_llm:
            self.llm_explainer = LLMRecommendationExplainer(api_key)
            self.enable_llm = self.llm_explainer.available
        
    def load_data(self):
        """Load all necessary data"""
        print("📁 Loading unified multimodal recommendation system...")
        
        if self.calculator.load_all_data():
            self.initialized = True
            print("✅ Unified multimodal recommendation system initialized successfully!")
        else:
            raise RuntimeError("Failed to load data")
    
    def get_similar_movies(self, movie_title: str, movie_year: str = None,
                          w_plot: float = 0.8, w_flow: float = 0.1, w_genre: float = 0.1,
                          top_k: int = 10) -> list:
        """Get similar movies using unified multimodal similarity"""
        if not self.initialized:
            self.load_data()
        
        # Include year in title if provided
        search_title = movie_title
        if movie_year:
            search_title = f"{movie_title} ({movie_year})"
        
        return self.calculator.calculate_weighted_similarity(
            movie_title=search_title,
            w_plot=w_plot,
            w_flow=w_flow,
            w_genre=w_genre,
            top_k=top_k
        )
    
    def display_similar_movies(self, similar_movies: list, target_title: str):
        """Display similar movie results with optional LLM explanations"""
        print(f"\n🎬 Movies similar to '{target_title}'")
        print("=" * 90)
        
        if not similar_movies:
            print("❌ No similar movies found.")
            return
        
        # 타겟 영화 정보 가져오기 (LLM 설명용)
        target_movie = None
        if self.enable_llm and self.llm_explainer:
            target_movie, _ = self.calculator.find_movie_by_title(target_title)
        
        for i, movie in enumerate(similar_movies):
            print(f"\n{movie['rank']:2d}. {movie['title']} ({movie['year']})")
            print(f"    Director: {movie['director']}")
            print(f"    🔢 Final similarity: {movie['similarity_score']:.4f}")
            
            # Component scores
            comp = movie['component_scores']
            weights = movie['weights_used']
            print(f"    📊 Components: plot={comp['plot']:.3f}(×{weights['plot']:.2f}), "
                  f"flow={comp['flow']:.3f}(×{weights['flow']:.2f}), "
                  f"genre={comp['genre']:.3f}(×{weights['genre']:.2f})")
            
            # LLM 추천 이유 설명
            if self.enable_llm and self.llm_explainer and target_movie:
                print(f"    🤖 LLM 분석 중... ({i+1}/{len(similar_movies)})")
                explanation = self.llm_explainer.explain_recommendation(target_movie, movie)
                print(f"    💡 Why recommended: {explanation}")
                
                # API 호출 간격 (rate limiting 방지)
                if i < len(similar_movies) - 1:  # 마지막이 아니면 대기
                    time.sleep(1)
            
            # Genre information
            genres = movie.get('genres', {})
            if genres:
                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                genre_str = ", ".join([f"{genre}({score})" for genre, score in top_genres])
                print(f"    🎭 Top genres: {genre_str}")
            
            # Flow curve info
            flow = movie.get('flow_curve', [])
            if flow:
                avg_tension = sum(flow) / len(flow)
                max_tension = max(flow)
                print(f"    📈 Flow: avg={avg_tension:.1f}, max={max_tension}")
            
            # Plot preview
            if movie.get('plot'):
                print(f"    📝 Plot: {movie['plot']}")
        
        # LLM 사용 여부 안내
        if self.enable_llm:
            print(f"\n💭 LLM 설명: {'활성화됨' if self.llm_explainer.available else '비활성화됨'}")

def interactive_mode(recommender):
    """Interactive mode for movie recommendations"""
    print("\n🎬 Unified Multimodal Movie Similarity Recommender")
    print("=" * 70)
    print("• Enter a movie title to get similar movie recommendations")
    print("• You can adjust weights for plot, flow curves, and genres")
    print("• All embeddings are newly created with unified dataset")
    if recommender.enable_llm:
        print("• 🤖 LLM explanations enabled")
    print("• Type 'quit' or 'exit' to terminate")
    print("=" * 70)
    
    while True:
        try:
            print("\n" + "="*50)
            movie_title = input("🔍 Enter movie title: ").strip()
            
            if movie_title.lower() in ['quit', 'exit']:
                print("👋 Terminating system.")
                break
                
            if not movie_title:
                print("❌ Please enter a movie title.")
                continue
            
            # Weight input
            print("\n⚖️ Weight settings (press Enter for defaults):")
            w_plot_input = input("  📝 Plot weight [default: 0.8]: ").strip()
            w_flow_input = input("  📈 Flow weight [default: 0.1]: ").strip()
            w_genre_input = input("  🎭 Genre weight [default: 0.1]: ").strip()
            
            # Parse weights
            try:
                w_plot = float(w_plot_input) if w_plot_input else 0.8
                w_flow = float(w_flow_input) if w_flow_input else 0.1
                w_genre = float(w_genre_input) if w_genre_input else 0.1
                
                # Validate weights
                if w_plot < 0 or w_flow < 0 or w_genre < 0:
                    print("❌ Weights must be non-negative.")
                    continue
                    
                total_weight = w_plot + w_flow + w_genre
                if total_weight == 0:
                    print("❌ At least one weight must be positive.")
                    continue
                    
            except ValueError:
                print("❌ Weights must be numbers.")
                continue
            
            # Search for similar movies
            print(f"\n🔍 Searching... (weights: {w_plot:.2f}, {w_flow:.2f}, {w_genre:.2f})")
            
            similar_movies = recommender.get_similar_movies(
                movie_title=movie_title,
                w_plot=w_plot,
                w_flow=w_flow,
                w_genre=w_genre,
                top_k=10
            )
            
            # Display results
            recommender.display_similar_movies(similar_movies, movie_title)
            
        except ValueError as e:
            print(f"❌ {e}")
        except KeyboardInterrupt:
            print("\n👋 Terminating system.")
            break
        except Exception as e:
            print(f"❌ Error occurred: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Unified Multimodal Movie Similarity Recommender")
    parser.add_argument('--movie', type=str, help='Movie title to find similar movies for')
    parser.add_argument('--year', type=str, help='Movie year (optional)')
    parser.add_argument('--plot_weight', type=float, default=0.8, help='Plot weight')
    parser.add_argument('--flow_weight', type=float, default=0.1, help='Flow weight')
    parser.add_argument('--genre_weight', type=float, default=0.1, help='Genre weight')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    # LLM 관련 인수 추가
    parser.add_argument('--llm', action='store_true', help='Enable LLM explanations')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (or use OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    try:
        # Initialize recommender with LLM option
        recommender = MovieSimilarityRecommender(
            data_dir=args.data_dir,
            enable_llm=args.llm,
            api_key=args.api_key
        )
        
        if args.movie:
            # Single movie search
            similar_movies = recommender.get_similar_movies(
                movie_title=args.movie,
                movie_year=args.year,
                w_plot=args.plot_weight,
                w_flow=args.flow_weight,
                w_genre=args.genre_weight,
                top_k=args.top_k
            )
            recommender.display_similar_movies(similar_movies, args.movie)
        else:
            # Interactive mode
            interactive_mode(recommender)
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
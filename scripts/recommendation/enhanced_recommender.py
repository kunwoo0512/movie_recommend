"""
하이브리드 추천 시스템 개선안
- 장르별 부스팅
- 쿼리 의도 분석
- 다단계 검색
"""
import json
import numpy as np
import faiss
import openai
from dotenv import load_dotenv
import os
import re

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class EnhancedMovieRecommender:
    def __init__(self):
        print("🚀 고도화된 하이브리드 영화 추천 시스템")
        print("=" * 50)
        
        # 벡터 DB 로드
        self.index = faiss.read_index('faiss_movie_index_plot_only.bin')
        
        with open('faiss_movie_meta_plot_only.json', 'r', encoding='utf-8') as f:
            self.movies = json.load(f)
        
        # 장르별 영화 인덱스 생성
        self.genre_index = self._build_genre_index()
        
        print(f"✅ 벡터 DB 로드 완료: {self.index.ntotal}개 영화")
        print(f"✅ 장르 인덱스 구축 완료: {len(self.genre_index)}개 장르")
    
    def _build_genre_index(self):
        """장르별 영화 인덱스 구축"""
        genre_index = {}
        
        for i, movie in enumerate(self.movies):
            genres = movie.get('genres', {})
            for genre, score in genres.items():
                if score >= 7:  # 높은 점수의 장르만
                    if genre not in genre_index:
                        genre_index[genre] = []
                    genre_index[genre].append(i)
        
        return genre_index
    
    def _analyze_query_intent(self, query):
        """쿼리 의도 분석"""
        intent = {
            'genre_keywords': [],
            'theme_keywords': [],
            'specific_movies': [],
            'intensity': 'medium'
        }
        
        # 장르 키워드 매핑
        genre_keywords = {
            'zombie': ['좀비', 'zombie', '감염', 'infected', '바이러스', 'virus'],
            'horror': ['공포', 'horror', '무서운', 'scary'],
            'romance': ['로맨스', 'romance', '사랑', 'love', '연애'],
            'action': ['액션', 'action', '전투', 'fight'],
            'sci_fi': ['SF', 'sci-fi', '우주', 'space', '미래', 'future'],
            'thriller': ['스릴러', 'thriller', '긴장', 'suspense']
        }
        
        # 테마 키워드
        theme_keywords = {
            'apocalypse': ['아포칼립스', 'apocalypse', '종말', '멸망'],
            'survival': ['생존', 'survival', '서바이벌'],
            'revenge': ['복수', 'revenge', '보복'],
            'family': ['가족', 'family', '부모', 'parent']
        }
        
        query_lower = query.lower()
        
        # 장르 분석
        for genre, keywords in genre_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent['genre_keywords'].append(genre)
        
        # 테마 분석
        for theme, keywords in theme_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent['theme_keywords'].append(theme)
        
        # 특정 영화 언급 체크
        for movie in self.movies:
            title = movie.get('title', '').lower()
            if title in query_lower and len(title) > 3:
                intent['specific_movies'].append(movie.get('title', ''))
        
        return intent
    
    def enhanced_search(self, query, top_k=30, max_results=5):
        """고도화된 검색"""
        print(f"\n🔍 고도화된 검색: '{query}'")
        print("=" * 60)
        
        # 1단계: 쿼리 의도 분석
        intent = self._analyze_query_intent(query)
        print(f"🧠 쿼리 의도 분석:")
        print(f"   장르: {intent['genre_keywords']}")
        print(f"   테마: {intent['theme_keywords']}")
        print(f"   언급된 영화: {intent['specific_movies']}")
        
        # 2단계: 기본 벡터 검색
        candidates = self._vector_search(query, top_k)
        
        # 3단계: 장르/테마 부스팅
        if intent['genre_keywords'] or intent['theme_keywords']:
            candidates = self._apply_boosting(candidates, intent)
        
        # 4단계: GPT 검증 (상위 후보만)
        final_results = self._gpt_verification(candidates[:top_k//2], query, max_results)
        
        return final_results
    
    def _vector_search(self, query, top_k):
        """벡터 검색"""
        try:
            response = openai.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            
            distances, indices = self.index.search(query_embedding, top_k)
            
            candidates = []
            for distance, idx in zip(distances[0], indices[0]):
                movie = self.movies[idx]
                candidates.append({
                    'movie': movie,
                    'vector_score': 1 / (1 + distance),
                    'boost_score': 0,
                    'final_score': 1 / (1 + distance)
                })
            
            return candidates
            
        except Exception as e:
            print(f"❌ 벡터 검색 실패: {e}")
            return []
    
    def _apply_boosting(self, candidates, intent):
        """장르/테마 기반 부스팅"""
        print("🚀 장르/테마 부스팅 적용")
        
        for candidate in candidates:
            movie = candidate['movie']
            boost = 0
            
            # 장르 부스팅
            genres = movie.get('genres', {})
            for genre in intent['genre_keywords']:
                if genre in genres:
                    boost += genres[genre] * 0.1  # 장르 점수 * 0.1
            
            # 테마 키워드 부스팅 (줄거리 기반)
            plot = movie.get('plot', '').lower()
            for theme in intent['theme_keywords']:
                theme_keywords = {
                    'apocalypse': ['apocalypse', '종말', '멸망', 'end of world'],
                    'survival': ['survival', '생존', 'survive'],
                    'revenge': ['revenge', '복수', 'vengeance'],
                    'family': ['family', '가족', 'father', 'mother']
                }
                
                if theme in theme_keywords:
                    keyword_matches = sum(1 for kw in theme_keywords[theme] if kw in plot)
                    boost += keyword_matches * 0.05
            
            candidate['boost_score'] = boost
            candidate['final_score'] = candidate['vector_score'] + boost
        
        # 부스팅 후 재정렬
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates
    
    def _gpt_verification(self, candidates, query, max_results):
        """GPT 검증"""
        print(f"🤖 GPT 검증 ({len(candidates)}개 후보)")
        
        verified = []
        for i, candidate in enumerate(candidates):
            if len(verified) >= max_results:
                break
                
            movie = candidate['movie']
            title = movie.get('title', '')
            plot = movie.get('plot', '')
            
            prompt = f"""다음 영화가 사용자 검색 의도와 얼마나 일치하는지 평가해주세요.

검색어: "{query}"
영화 제목: "{title}"
줄거리: "{plot[:500]}..."

다음 형식으로만 답변해주세요:
점수: [1-10]
이유: [한 줄 설명]"""

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                
                result = response.choices[0].message.content.strip()
                
                # 점수 추출
                score_match = re.search(r'점수:\s*(\d+)', result)
                reason_match = re.search(r'이유:\s*(.+)', result)
                
                if score_match and int(score_match.group(1)) >= 7:
                    verified.append({
                        'title': title,
                        'year': movie.get('year', ''),
                        'director': movie.get('director', ''),
                        'plot': plot,
                        'vector_score': candidate['vector_score'],
                        'boost_score': candidate['boost_score'],
                        'gpt_score': int(score_match.group(1)),
                        'gpt_reason': reason_match.group(1) if reason_match else "",
                        'final_score': candidate['final_score']
                    })
                    print(f"   ✅ {title} - GPT: {score_match.group(1)}/10")
                else:
                    print(f"   ❌ {title} - 검증 실패")
                    
            except Exception as e:
                print(f"   ❌ {title} - GPT 오류: {e}")
        
        return verified

def test_enhanced_system():
    """고도화된 시스템 테스트"""
    recommender = EnhancedMovieRecommender()
    
    test_queries = [
        "좀비 바이러스 감염 영화",
        "28 Days Later 같은 포스트 아포칼립스 생존 영화",
        "무서운 좀비 공포 영화"
    ]
    
    for query in test_queries:
        results = recommender.enhanced_search(query, top_k=25, max_results=3)
        
        print(f"\n🎯 최종 결과 ({len(results)}개):")
        for i, movie in enumerate(results, 1):
            print(f"{i}. {movie['title']} ({movie['year']})")
            print(f"   벡터: {movie['vector_score']:.3f} + 부스트: {movie['boost_score']:.3f} = {movie['final_score']:.3f}")
            print(f"   GPT: {movie['gpt_score']}/10 - {movie['gpt_reason']}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    test_enhanced_system()

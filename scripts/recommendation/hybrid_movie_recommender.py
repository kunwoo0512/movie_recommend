"""
벡터 DB + LLM 하이브리드 영화 추천 시스템
"""
import json
import numpy as np
import faiss
import openai
import os
from dotenv import load_dotenv
import time

# .env 파일 로드
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class HybridMovieRecommender:
    def __init__(self):
        """하이브리드 영화 추천 시스템 초기화"""
        print("🎬 하이브리드 영화 추천 시스템 초기화")
        print("=" * 50)
        
        # 벡터 DB 로드
        self.index = faiss.read_index('faiss_movie_index_plot_only.bin')
        
        with open('faiss_movie_meta_plot_only.json', 'r', encoding='utf-8') as f:
            self.titles = json.load(f)
        
        with open('movies_dataset.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        
        # 영화 제목 -> 상세 정보 매핑 (백업용)
        self.movie_details = {movie['title']: movie for movie in movies_data}
        
        print(f"✅ 벡터 DB 로드 완료: {self.index.ntotal}개 영화")
        print(f"✅ 영화 메타데이터 로드 완료: {len(self.movie_details)}개")
    
    def get_query_embedding(self, query):
        """사용자 쿼리를 임베딩으로 변환"""
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            embedding = np.array(response.data[0].embedding).astype('float32')
            # L2 정규화
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"❌ 쿼리 임베딩 생성 실패: {e}")
            return None
    
    def vector_search(self, query, top_k=20):
        """1단계: 벡터 DB로 후보 영화 검색"""
        print(f"🔍 1단계: 벡터 DB 검색 (상위 {top_k}개 후보)")
        
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # FAISS 검색
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k=top_k)
        
        candidates = []
        for distance, idx in zip(distances[0], indices[0]):
            # titles는 이제 전체 영화 정보 딕셔너리들의 리스트입니다
            movie_info = self.titles[idx]
            movie_title = movie_info.get('title', 'Unknown')
            
            candidates.append({
                'title': movie_title,
                'plot': movie_info.get('plot', ''),
                'year': movie_info.get('year', 'N/A'),
                'director': movie_info.get('director', 'N/A'),
                'vector_similarity': 1 / (1 + distance),
                'genres': movie_info.get('genres', {})
            })
        
        print(f"   ✅ {len(candidates)}개 후보 영화 추출 완료")
        return candidates
    
    def gpt_verify(self, query, candidates, max_results=5):
        """2단계: GPT로 후보 영화들 검증"""
        print(f"🤖 2단계: GPT 검증 (최대 {max_results}개 선별)")
        
        verified_movies = []
        verification_costs = 0
        
        for i, candidate in enumerate(candidates):
            print(f"   [{i+1}/{len(candidates)}] '{candidate['title']}' 검증 중...")
            
            # GPT 검증 프롬프트 (개선된 버전)
            prompt = f"""사용자가 "{query}"라는 검색어로 영화를 찾고 있습니다.

영화 정보:
제목: {candidate['title']} ({candidate['year']})
감독: {candidate['director']}
줄거리: {candidate['plot']}

관련성 판단 기준:
- 주요 테마나 장르가 검색어와 일치하면 "예"
- 줄거리 내용이 검색어와 부분적으로라도 관련이 있으면 "예"  
- 완전히 무관하고 전혀 다른 내용인 경우만 "아니오"
- 애매한 경우에는 "예"로 판단 (관대하게)

답변 형식:
일치여부: 예/아니오
신뢰도: 1-10점 (1=전혀무관, 10=완벽일치)
이유: (한 줄 설명)"""

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",  # 더 정확한 모델로 변경
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,  # 더 자세한 응답을 위해 증가
                    temperature=0.1
                )
                
                gpt_response = response.choices[0].message.content
                verification_costs += 0.0006  # gpt-4o-mini 비용 (더 저렴함)
                
                # 응답 파싱
                if "일치여부: 예" in gpt_response or "일치여부:예" in gpt_response:
                    # 신뢰도 추출
                    confidence = 7  # 기본값
                    try:
                        if "신뢰도:" in gpt_response:
                            confidence_text = gpt_response.split("신뢰도:")[1].split("점")[0].strip()
                            confidence = int(confidence_text)
                    except:
                        pass
                    
                    # 이유 추출
                    reason = "GPT 검증 통과"
                    try:
                        if "이유:" in gpt_response:
                            reason = gpt_response.split("이유:")[1].strip()
                    except:
                        pass
                    
                    verified_movies.append({
                        **candidate,
                        'gpt_confidence': confidence,
                        'gpt_reason': reason,
                        'final_score': candidate['vector_similarity'] * 0.7 + confidence/10 * 0.3
                    })
                    
                    print(f"      ✅ 검증 통과 (신뢰도: {confidence}/10)")
                else:
                    print(f"      ❌ 검증 실패")
                
                # API 제한 고려
                time.sleep(0.1)
                
                # 충분한 결과가 나오면 조기 종료
                if len(verified_movies) >= max_results:
                    print(f"   🎯 목표 개수 달성, 검증 완료")
                    break
                    
            except Exception as e:
                print(f"      ❌ GPT 검증 오류: {e}")
                continue
        
        # 최종 점수순 정렬
        verified_movies.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"   ✅ 총 {len(verified_movies)}개 영화 검증 통과")
        print(f"   💰 예상 비용: ${verification_costs:.4f}")
        
        return verified_movies[:max_results]
    
    def search(self, query, top_k_candidates=20, max_results=5):
        """통합 검색 메서드"""
        print(f"\n🎬 하이브리드 검색: '{query}'")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1단계: 벡터 검색
        candidates = self.vector_search(query, top_k_candidates)
        if not candidates:
            print("❌ 벡터 검색 실패")
            return []
        
        # 2단계: GPT 검증
        verified_results = self.gpt_verify(query, candidates, max_results)
        
        # 검색 완료
        search_time = time.time() - start_time
        print(f"\n⏱️ 총 검색 시간: {search_time:.2f}초")
        
        return verified_results
    
    def display_results(self, results):
        """검색 결과 출력"""
        if not results:
            print("😞 검색 결과가 없습니다.")
            return
        
        print(f"\n🎯 최종 추천 결과:")
        print("=" * 50)
        
        for i, movie in enumerate(results, 1):
            print(f"\n{i}. {movie['title']} ({movie['year']})")
            print(f"   감독: {movie['director']}")
            print(f"   최종 점수: {movie['final_score']:.3f}")
            print(f"   벡터 유사도: {movie['vector_similarity']:.3f}")
            print(f"   GPT 신뢰도: {movie['gpt_confidence']}/10")
            print(f"   GPT 이유: {movie['gpt_reason']}")
            
            # 장르 정보
            genres = movie.get('genres', {})
            if genres:
                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                genre_str = ', '.join([f"{g}:{s}" for g, s in top_genres])
                print(f"   주요 장르: {genre_str}")

def main():
    """테스트 실행"""
    # 하이브리드 추천 시스템 초기화
    recommender = HybridMovieRecommender()
    
    # 테스트 쿼리들 (개선된 전처리 테스트)
    test_queries = [
        "좀비 아포칼립스 생존",  # 전처리로 개선될 쿼리
        "로맨틱 코미디 영화"     # 정상 쿼리
    ]
    
    for query in test_queries:
        # 검색 실행 (좀비 영화 테스트를 위해 후보를 더 많이)
        results = recommender.search(query, top_k_candidates=20, max_results=5)
        
        # 결과 출력
        recommender.display_results(results)
        
        print("\n" + "="*80)
        
        # 자동으로 다음 검색 진행
        print("자동으로 다음 검색 진행...")
    
    print("🎬 하이브리드 영화 추천 시스템 테스트 완료!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SentenceBERT 기반 영화 추천 시스템 테스터
bert_movie_recommender_new.py와 동일한 구조 (벡터 검색 + GPT 검증)
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv
import time
import re

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class SentenceBertMovieRecommender:
    def __init__(self):
        """SentenceBERT 기반 영화 추천 시스템 초기화"""
        print("[시작] SentenceBERT 기반 영화 추천 시스템 초기화")
        print("=" * 60)
        
        # SentenceBERT 모델 로드
        self.load_sentencebert_model()
        
        # 벡터 DB 로드
        self.load_vector_db()
        
        print("[완료] 시스템 초기화 완료")
        print("=" * 60)
    
    def load_sentencebert_model(self):
        """SentenceBERT 모델 로드"""
        print("[로드] SentenceBERT 모델 로딩...")
        
        try:
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("   [성공] SentenceBERT 모델 로드 완료")
        except Exception as e:
            print(f"   [오류] SentenceBERT 모델 로드 실패: {e}")
            self.model = None
    
    def load_vector_db(self):
        """SentenceBERT 기반 벡터 DB 로드"""
        print("[로드] SentenceBERT 벡터 DB 로딩...")
        
        try:
            # FAISS 인덱스 로드
            self.index = faiss.read_index('faiss_movie_index_bert.bin')
            
            # 메타데이터 로드
            with open('faiss_movie_meta_bert.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"   [성공] SentenceBERT 벡터 DB: {self.index.ntotal}개 영화")
            
        except Exception as e:
            print(f"   [오류] 벡터 DB 로드 실패: {e}")
            print(f"   [해결책] build_bert_vector_db.py를 먼저 실행하세요.")
            self.index = None
            self.metadata = None
    
    def search_movies(self, query: str, top_k: int = 20):
        """SentenceBERT 임베딩으로 영화 검색"""
        if not self.index or not self.metadata:
            return []
        
        print(f"[검색] SentenceBERT 기반 벡터 검색 (상위 {top_k}개)")
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # FAISS 검색
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 결과 정리
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                movie_info = self.metadata[idx]
                results.append({
                    'title': movie_info['title'],
                    'plot': movie_info['plot'],
                    'similarity': float(score),
                    'year': movie_info['year'],
                    'director': movie_info['director'],
                    'genres': movie_info['genres'],
                    'movie_info': movie_info
                })
        
        print(f"   [결과] {len(results)}개 후보 발견")
        return results
    
    def gpt_verification(self, query: str, candidates, max_results: int = 5):
        """GPT를 통한 최종 검증"""
        verified_movies = []
        
        print(f"[검증] GPT 최종 검증 (최대 {max_results}개 선별)")
        
        for i, candidate in enumerate(candidates[:max_results * 2]):
            if len(verified_movies) >= max_results:
                break
            
            title = candidate['title']
            plot = candidate['plot']
            
            print(f"   [{i+1:2d}] {title} 검증 중...")
            
            try:
                prompt = f"""영화 추천 정확도 평가를 해주세요.

사용자 요청: "{query}"

영화 정보:
제목: {title}
줄거리: {plot}

이 영화가 사용자 요청에 얼마나 적합한지 1-10점으로 평가해주세요.

평가 기준:
- 요청의 핵심 키워드와 정확히 일치하는가?
- 영화의 주요 테마가 요청과 부합하는가?
- 부분적 유사성이 아닌 명확한 일치인가?

**최종 점수: X/10** (X는 1-10 사이 숫자)

6점 이상만 추천 가능합니다."""

                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "당신은 영화 추천 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.3
                )
                
                gpt_response = response.choices[0].message.content
                
                # 점수 추출
                def extract_score(text):
                    patterns = [
                        r'최종\s*점수[:\s]*(\d+(?:\.\d+)?)',
                        r'점수[:\s]*(\d+(?:\.\d+)?)',
                        r'(\d+(?:\.\d+)?)/10',
                        r'(\d+(?:\.\d+)?)점'
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            for match in matches:
                                try:
                                    score = float(match)
                                    if 1 <= score <= 10:
                                        return score
                                except ValueError:
                                    continue
                    return None
                
                gpt_score = extract_score(gpt_response)
                
                if gpt_score is not None and gpt_score >= 6.0:
                    candidate['gpt_score'] = gpt_score
                    candidate['gpt_reason'] = gpt_response
                    verified_movies.append(candidate)
                    print(f"      [통과] GPT: {gpt_score}/10")
                else:
                    score_text = f"{gpt_score}/10" if gpt_score else "추출실패"
                    print(f"      [제외] GPT: {score_text}")
                
                time.sleep(1)  # API 제한 고려
                
            except Exception as e:
                print(f"      [오류] GPT 검증 실패: {e}")
        
        print(f"   [완료] {len(verified_movies)}개 영화 최종 선별")
        
        # GPT 점수순 정렬
        verified_movies.sort(key=lambda x: x.get('gpt_score', 0), reverse=True)
        
        return verified_movies[:max_results]
    
    def search(self, query: str, max_results: int = 5):
        """통합 검색 (SentenceBERT + GPT)"""
        print(f"\n[검색] '{query}'")
        print("=" * 60)
        
        # 1단계: SentenceBERT 기반 벡터 검색
        candidates = self.search_movies(query, top_k=20)
        
        if not candidates:
            print("[결과] 검색 결과가 없습니다.")
            return []
        
        # OpenAI API 키 확인
        if not openai.api_key:
            print("[알림] GPT 검증을 위해서는 .env 파일에 OPENAI_API_KEY 설정이 필요합니다.")
            print("[대안] SentenceBERT 검색 결과만 출력합니다.")
            return candidates[:max_results]
        
        # 2단계: GPT 검증
        final_results = self.gpt_verification(query, candidates, max_results)
        
        return final_results
    
    def display_results(self, results, query: str):
        """검색 결과 출력"""
        print(f"\n[최종] '{query}' 검색 결과")
        print("=" * 60)
        
        if not results:
            print(" 조건에 맞는 영화를 찾을 수 없습니다.")
            print(" 다른 키워드로 다시 검색해보세요.")
            return
        
        for i, movie in enumerate(results, 1):
            title = movie['title']
            gpt_score = movie.get('gpt_score', 0)
            bert_similarity = movie.get('similarity', 0)
            
            print(f"\n {i}. {title}")
            
            # GPT 점수가 있으면 표시, 없으면 SentenceBERT 유사도만 표시
            if gpt_score > 0:
                print(f"    GPT 점수: {gpt_score}/10")
                print(f"    SentenceBERT 유사도: {bert_similarity:.3f}")
            else:
                print(f"    SentenceBERT 유사도: {bert_similarity:.3f}")
            
            # 영화 기본 정보
            year = movie.get('year', 'N/A')
            director = movie.get('director', 'N/A')
            print(f"    연도: {year} | 🎭 감독: {director}")
            
            # 장르 정보
            genres = movie.get('genres', {})
            if genres:
                genre_list = [f"{k}({v})" for k, v in genres.items()]
                print(f"    장르: {', '.join(genre_list)}")
            
            # 줄거리 미리보기 추가
            plot = movie.get('plot', '')
            if plot:
                plot_preview = plot[:200] + "..." if len(plot) > 200 else plot
                print(f"    줄거리: {plot_preview}")
    
    def run_test_queries(self):
        """미리 정의된 테스트 쿼리 실행"""
        print("\n[테스트] SentenceBERT + GPT 검증 테스트")
        print("=" * 60)
        
        test_queries = [
            "매트릭스",
            "matrix", 
            "가상현실",
            "꿈과 현실",
            "inception",
            "꿈 속의 꿈",
            "타이타닉",
            "사랑",
            "공포 영화",
            "horror",
            "comedy",
            "christopher nolan",
            "leonardo dicaprio"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} 테스트 {i}/{len(test_queries)} {'='*20}")
            
            try:
                # 검색 실행
                results = self.search(query, max_results=3)
                
                # 결과 출력
                self.display_results(results, query)
                
                print("\n" + "-"*50)
                
            except Exception as e:
                print(f"테스트 실패: {e}")
    
    def interactive_search(self):
        """대화형 검색 모드"""
        print("\n SentenceBERT + GPT 영화 추천 시스템")
        print("• SentenceBERT 임베딩으로 의미적 검색")
        print("• GPT 검증으로 정확도 향상")
        print("• 한국어 쿼리 + 영어 줄거리 지원")
        print("• 'quit' 입력시 종료")
        
        while True:
            try:
                query = input("\n 어떤 영화를 찾고 계신가요? > ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print(" 시스템을 종료합니다.")
                    break
                
                if not query:
                    print(" 검색어를 입력해주세요.")
                    continue
                
                # 검색 실행
                results = self.search(query, max_results=5)
                
                # 결과 출력
                self.display_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\n 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f" 검색 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    try:
        # 시스템 초기화
        recommender = SentenceBertMovieRecommender()
        
        if not recommender.index or not recommender.model:
            print(" 시스템 초기화에 실패했습니다.")
            return
        
        print("\n SentenceBERT 기반 영화 추천 시스템")
        print("• SentenceBERT 임베딩으로 의미적 검색")
        print("• GPT 검증으로 정확도 향상")
        print("• 한국어 쿼리 + 영어 줄거리 지원")
        print("• 'quit' 입력시 종료")
        
        while True:
            try:
                query = input("\n 어떤 영화를 찾고 계신가요? > ").strip()
                
                if query.lower() in ['quit', 'exit', '종료']:
                    print(" 시스템을 종료합니다.")
                    break
                
                if not query:
                    print(" 검색어를 입력해주세요.")
                    continue
                
                # 검색 실행
                results = recommender.search(query, max_results=5)
                
                # 결과 출력
                recommender.display_results(results, query)
                
            except KeyboardInterrupt:
                print("\n\n 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f" 검색 중 오류 발생: {e}")
                
    except Exception as e:
        print(f" 시스템 초기화 실패: {e}")
        print("\n 필요한 파일들:")
        print("1. movie_embeddings_bert.json")
        print("2. faiss_movie_index_bert.bin")
        print("3. faiss_movie_meta_bert.json")
        print("4. .env 파일 (OPENAI_API_KEY)")
        print("\n 실행 순서:")
        print("1. bert_embedding_generator_new.py")
        print("2. build_bert_vector_db.py")
        print("3. sentencebert_recommender_test.py (현재 파일)")

if __name__ == "__main__":
    main()
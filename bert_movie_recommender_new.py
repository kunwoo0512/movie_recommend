#!/usr/bin/env python3
"""
SentenceBERT 기반 영화 추천 시스템
단일 SentenceBERT 임베딩 + GPT 검증으로 정확한 추천
"""

import json
import numpy as np
import faiss
import torch
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
        self.load_sentence_bert_model()
        
        # 벡터 DB 로드
        self.load_bert_vector_db()
        
        print("[완료] 시스템 초기화 완료")
        print("=" * 60)
    
    def load_sentence_bert_model(self):
        """SentenceBERT 모델 로드"""
        print("[로드] SentenceBERT 모델 로딩...")
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
            print(f"   [성공] SentenceBERT 모델 로드 완료 (장치: {self.device})")
        except Exception as e:
            print(f"   [오류] SentenceBERT 모델 로드 실패: {e}")
            raise
    
    def load_bert_vector_db(self):
        """BERT 벡터 DB 로드"""
        print("[로드] BERT 벡터 DB 로딩...")
        try:
            # FAISS 인덱스 로드
            self.index = faiss.read_index('faiss_movie_index_bert.bin')
            
            # 메타데이터 로드
            with open('faiss_movie_meta_bert.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print(f"   [성공] BERT 벡터 DB: {len(self.metadata)}개 영화")
        except Exception as e:
            print(f"   [오류] BERT 벡터 DB 로드 실패: {e}")
            raise
    
    def get_query_embedding(self, query):
        """쿼리의 SentenceBERT 임베딩 생성"""
        try:
            # SentenceBERT로 임베딩 생성
            embedding = self.model.encode(query, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"[오류] 쿼리 임베딩 생성 실패: {e}")
            return None
    
    def search_similar_movies(self, query, top_k=20):
        """유사한 영화 검색"""
        print(f"\n[검색] 쿼리: '{query}'")
        print("-" * 50)
        
        # 쿼리 임베딩 생성
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # FAISS 검색
        query_embedding = query_embedding.reshape(1, -1)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # 결과 정리
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.metadata):
                movie = self.metadata[idx].copy()
                movie['similarity'] = float(similarity)
                movie['rank'] = i + 1
                results.append(movie)
        
        print(f"[결과] {len(results)}개 영화 검색 완료")
        return results
    
    def verify_with_gpt(self, query, movies, max_verify=5):
        """GPT로 추천 결과 검증"""
        if not openai.api_key:
            print("[경고] OpenAI API 키가 없어 GPT 검증을 건너뜁니다.")
            return movies
        
        print(f"[검증] GPT로 상위 {min(max_verify, len(movies))}개 영화 검증 중...")
        
        # GPT 검증용 프롬프트
        movies_text = ""
        for i, movie in enumerate(movies[:max_verify]):
            movies_text += f"{i+1}. {movie['title']} ({movie.get('year', 'N/A')})\n"
            movies_text += f"   줄거리: {movie['plot'][:200]}...\n\n"
        
        prompt = f"""
다음은 사용자 질의 "{query}"에 대한 AI 검색 결과입니다.
각 영화가 사용자의 질의와 얼마나 관련이 있는지 1-10점으로 평가해주세요.

영화 목록:
{movies_text}

평가 기준:
- 10점: 질의와 완벽히 일치
- 8-9점: 매우 관련이 높음
- 6-7점: 어느 정도 관련이 있음
- 4-5점: 약간 관련이 있음
- 1-3점: 거의 관련이 없음
- 0점: 전혀 관련이 없음

각 영화에 대해 "영화번호: 점수 (간단한 이유)" 형태로 답변해주세요.
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            gpt_analysis = response.choices[0].message.content
            print(f"[GPT 분석]\n{gpt_analysis}")
            
            # GPT 점수 파싱 및 적용
            lines = gpt_analysis.split('\n')
            for line in lines:
                if ':' in line and any(char.isdigit() for char in line):
                    try:
                        # "1: 8점" 또는 "1: 8 (이유)" 형태 파싱
                        parts = line.split(':')
                        if len(parts) >= 2:
                            movie_num = int(parts[0].strip()) - 1
                            score_text = parts[1].strip()
                            score = float(re.findall(r'\d+', score_text)[0])
                            
                            if 0 <= movie_num < len(movies):
                                movies[movie_num]['gpt_score'] = score
                    except:
                        continue
            
            # GPT 점수가 있는 영화들을 GPT 점수로 재정렬
            movies_with_gpt = [m for m in movies if 'gpt_score' in m]
            movies_without_gpt = [m for m in movies if 'gpt_score' not in m]
            
            movies_with_gpt.sort(key=lambda x: x['gpt_score'], reverse=True)
            
            return movies_with_gpt + movies_without_gpt
            
        except Exception as e:
            print(f"[오류] GPT 검증 실패: {e}")
            return movies
    
    def recommend_movies(self, query, top_k=10, use_gpt_verification=True):
        """영화 추천 (검색 + GPT 검증)"""
        # 1단계: 벡터 검색
        search_results = self.search_similar_movies(query, top_k * 2)  # 더 많이 검색해서 GPT가 필터링
        
        if not search_results:
            print("[결과] 검색 결과가 없습니다.")
            return []
        
        # 2단계: GPT 검증 (선택적)
        if use_gpt_verification:
            final_results = self.verify_with_gpt(query, search_results)
        else:
            final_results = search_results
        
        # 상위 결과만 반환
        return final_results[:top_k]
    
    def display_recommendations(self, movies, query):
        """추천 결과 출력"""
        print(f"\n{'='*60}")
        print(f"🎬 '{query}'에 대한 영화 추천 결과")
        print(f"{'='*60}")
        
        if not movies:
            print("추천할 영화가 없습니다.")
            return
        
        for i, movie in enumerate(movies):
            print(f"\n{i+1}. 🎥 {movie['title']}")
            print(f"   📅 연도: {movie.get('year', 'N/A')}")
            print(f"   🎭 감독: {movie.get('director', 'N/A')}")
            print(f"   ⭐ 유사도: {movie.get('similarity', 0):.4f}")
            
            if 'gpt_score' in movie:
                print(f"   🤖 GPT 점수: {movie['gpt_score']}/10")
            
            # 줄거리 미리보기
            plot_preview = movie['plot'][:150] + "..." if len(movie['plot']) > 150 else movie['plot']
            print(f"   📝 줄거리: {plot_preview}")
    
    def analyze_search_quality(self, test_queries=None):
        """검색 품질 분석"""
        if test_queries is None:
            test_queries = [
                "꿈과 현실을 오가는 영화",
                "matrix",
                "inception", 
                "dream and reality movie",
                "switching between dream and reality",
                "로봇과 인간의 사랑",
                "time travel movie"
            ]
        
        print(f"\n{'='*60}")
        print("🔍 검색 품질 분석")
        print(f"{'='*60}")
        
        for query in test_queries:
            print(f"\n쿼리: '{query}'")
            print("-" * 40)
            
            results = self.search_similar_movies(query, 5)
            
            for i, movie in enumerate(results[:3]):
                print(f"  {i+1}. {movie['title']} (유사도: {movie['similarity']:.4f})")
    
    def find_movie_by_title(self, title):
        """제목으로 영화 찾기"""
        for i, movie in enumerate(self.metadata):
            if title.lower() in movie['title'].lower():
                return i, movie
        return None, None
    
    def get_movie_similarity(self, query, movie_title):
        """특정 영화와 쿼리의 유사도 계산"""
        idx, movie = self.find_movie_by_title(movie_title)
        if movie is None:
            print(f"영화 '{movie_title}'을 찾을 수 없습니다.")
            return None
        
        # 쿼리 임베딩
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return None
        
        # 영화 임베딩 (인덱스에서 가져오기)
        movie_embedding = self.index.reconstruct(idx)
        
        # 코사인 유사도 계산 (정규화된 벡터이므로 내적이 코사인 유사도)
        similarity = np.dot(query_embedding, movie_embedding)
        
        return float(similarity)

def main():
    """메인 실행 함수"""
    try:
        # 추천 시스템 초기화
        recommender = SentenceBertMovieRecommender()
        
        # 테스트 쿼리들
        test_queries = [
            "꿈과 현실을 오가는 영화",
            "dream and reality movie", 
            "switching between dream and reality",
            "matrix",
            "inception",
            "로봇과 인간의 사랑 이야기"
        ]
        
        # 각 쿼리에 대해 추천 실행
        for query in test_queries:
            recommendations = recommender.recommend_movies(query, top_k=5, use_gpt_verification=False)
            recommender.display_recommendations(recommendations, query)
            
            # Matrix와 Inception의 유사도 확인
            if "dream" in query.lower() or "reality" in query.lower():
                print(f"\n🎯 특정 영화 유사도 분석:")
                for target_movie in ["The Matrix", "Inception"]:
                    similarity = recommender.get_movie_similarity(query, target_movie)
                    if similarity is not None:
                        print(f"   {target_movie}: {similarity:.4f}")
            
            print("\n" + "="*80 + "\n")
    
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

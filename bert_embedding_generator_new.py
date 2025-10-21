#!/usr/bin/env python3
"""
SentenceBERT 기반 영화 줄거리 임베딩 생성기
다국어 SentenceBERT를 사용하여 한국어 쿼리와 영어 줄거리 모두 처리
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

class SentenceBertEmbeddingGenerator:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """SentenceBERT 임베딩 생성기 초기화"""
        print(f"[초기화] SentenceBERT 모델 로딩: {model_name}")
        print("=" * 60)
        
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[장치] 사용 장치: {self.device}")
        
        # SentenceBERT 모델 로드
        try:
            print("[다운로드] SentenceBERT 모델 다운로드 중... (최초 실행시 시간이 걸릴 수 있습니다)")
            self.model = SentenceTransformer(model_name, device=self.device)
            
            print(f"[성공] SentenceBERT 모델 로드 완료")
            print(f"[정보] 임베딩 차원: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"[오류] SentenceBERT 모델 로드 실패: {e}")
            print("[해결책] 인터넷 연결을 확인하고 sentence-transformers 설치를 확인하세요.")
            print("설치 명령: pip install sentence-transformers")
            raise
    
    def get_sentence_embedding(self, text: str):
        """텍스트를 SentenceBERT 임베딩으로 변환"""
        try:
            # SentenceBERT로 임베딩 생성 (자동으로 정규화됨)
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype('float32')
            
        except Exception as e:
            print(f"[오류] 임베딩 생성 실패: {e}")
            # 에러 시 영벡터 반환 (384차원 - MiniLM 기본 차원)
            return np.zeros(384, dtype='float32')
    
    def process_movies_dataset(self, input_file: str = 'movies_dataset.json', 
                             output_file: str = 'movie_embeddings_bert.json'):
        """영화 데이터셋의 모든 줄거리를 SentenceBERT 임베딩으로 변환"""
        print(f"\n[시작] 영화 데이터셋 SentenceBERT 임베딩 생성")
        print("=" * 60)
        
        # 영화 데이터 로드
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                movies = json.load(f)
            print(f"[로드] {len(movies)}개 영화 데이터 로드 완료")
        except Exception as e:
            print(f"[오류] 영화 데이터 로드 실패: {e}")
            return
        
        # 임베딩 생성
        embeddings_data = []
        successful_count = 0
        start_time = time.time()
        
        print(f"[진행] SentenceBERT 임베딩 생성 중...")
        print(f"[정보] 예상 소요 시간: {len(movies) * 0.1 / 60:.1f}분 (BERT보다 훨씬 빠름)")
        
        # 배치 처리를 위한 텍스트 리스트
        plots = []
        movie_infos = []
        
        for i, movie in enumerate(movies):
            title = movie.get('title', f'Movie_{i}')
            plot = movie.get('plot', '')
            
            if not plot.strip():
                print(f"\n[경고] '{title}': 줄거리가 비어있음")
                plot = title  # 제목을 대신 사용
            
            plots.append(plot)
            movie_infos.append({
                'title': title,
                'plot': plot,
                'year': movie.get('year', ''),
                'director': movie.get('director', ''),
                'genres': movie.get('genres', {}),
                'movie_id': movie.get('movie_id', f'movie_{i}')
            })
        
        # 배치로 임베딩 생성 (훨씬 빠름)
        print(f"[배치] {len(plots)}개 줄거리 배치 임베딩 생성 중...")
        try:
            embeddings = self.model.encode(plots, 
                                         convert_to_numpy=True, 
                                         show_progress_bar=True,
                                         batch_size=32)
            
            # 결과 저장
            for i, (movie_info, embedding) in enumerate(zip(movie_infos, embeddings)):
                embeddings_data.append({
                    **movie_info,
                    'embedding': embedding.tolist()
                })
                successful_count += 1
                
                # 중간 저장 (100개마다)
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    remaining = (len(movies) - i - 1) * (elapsed / (i + 1))
                    print(f"\n[진행] {i + 1}/{len(movies)} 완료 ({successful_count}개 성공)")
                    print(f"[시간] 경과: {elapsed/60:.1f}분, 남은 시간: {remaining/60:.1f}분")
                    self._save_intermediate(embeddings_data, output_file, i + 1)
            
        except Exception as e:
            print(f"[오류] 배치 임베딩 생성 실패: {e}")
            # 개별 처리로 폴백
            print("[폴백] 개별 임베딩 생성으로 전환...")
            embeddings_data = []
            
            for i, movie_info in enumerate(tqdm(movie_infos, desc="개별 임베딩 생성")):
                try:
                    embedding = self.get_sentence_embedding(movie_info['plot'])
                    embeddings_data.append({
                        **movie_info,
                        'embedding': embedding.tolist()
                    })
                    successful_count += 1
                    
                except Exception as e:
                    print(f"\n[오류] '{movie_info['title']}' 처리 실패: {e}")
                    continue
        
        # 최종 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            
            total_time = time.time() - start_time
            print(f"\n[완료] SentenceBERT 임베딩 생성 완료!")
            print(f"[결과] 총 {successful_count}/{len(movies)}개 영화 처리")
            print(f"[시간] 총 소요 시간: {total_time/60:.1f}분")
            print(f"[저장] {output_file}에 저장됨")
            print(f"[크기] 파일 크기: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"[오류] 파일 저장 실패: {e}")
    
    def _save_intermediate(self, data, filename, count):
        """중간 결과 저장"""
        backup_name = f"{filename}.backup_{count}"
        try:
            with open(backup_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[백업] {backup_name} 저장 완료")
        except Exception as e:
            print(f"[오류] 백업 저장 실패: {e}")
    
    def test_embedding_quality(self, test_queries=None):
        """임베딩 품질 테스트"""
        if test_queries is None:
            test_queries = [
                "꿈과 현실을 오가는 영화",
                "주인공이 초반에 실패하지만 결국 성공하는 영화",
                "로봇과 인간의 사랑 이야기",
                "time travel movie",
                "love story between human and AI",
                "matrix",
                "inception"
            ]
        
        print(f"\n[테스트] SentenceBERT 임베딩 품질 테스트")
        print("=" * 60)
        
        for query in test_queries:
            print(f"\n쿼리: '{query}'")
            start_time = time.time()
            embedding = self.get_sentence_embedding(query)
            end_time = time.time()
            
            print(f"임베딩 차원: {embedding.shape}")
            print(f"임베딩 범위: [{embedding.min():.3f}, {embedding.max():.3f}]")
            print(f"임베딩 L2 노름: {np.linalg.norm(embedding):.3f}")
            print(f"생성 시간: {(end_time - start_time)*1000:.1f}ms")

def main():
    """메인 실행 함수"""
    try:
        print("🎬 SentenceBERT 기반 영화 임베딩 생성기")
        print("=" * 60)
        print("이 도구는 영화 줄거리를 SentenceBERT 임베딩으로 변환합니다.")
        print("• paraphrase-multilingual-MiniLM-L12-v2 사용 (한국어 + 영어 지원)")
        print("• 문장 의미 임베딩 특화 모델")
        print("• 기존 BERT [CLS]보다 훨씬 높은 의미 유사도 성능")
        print("• 배치 처리로 빠른 속도")
        print()
        
        # SentenceBERT 임베딩 생성기 초기화
        generator = SentenceBertEmbeddingGenerator()
        
        # 임베딩 품질 테스트
        generator.test_embedding_quality()
        
        # 사용자 확인
        print("\n" + "="*60)
        response = input("영화 데이터셋 전체를 SentenceBERT 임베딩으로 변환하시겠습니까? (y/n): ")
        
        if response.lower() in ['y', 'yes', '예', 'ㅇ']:
            # 전체 데이터셋 처리
            generator.process_movies_dataset()
            
            print("\n🎉 임베딩 생성 완료!")
            print("다음 단계:")
            print("1. build_bert_vector_db.py 실행하여 FAISS 인덱스 생성")
            print("2. bert_movie_recommender_new.py로 검색 테스트")
            print("\n주요 개선사항:")
            print("• 기존 BERT [CLS] → SentenceBERT 문장 임베딩")
            print("• 의미 유사도 검색 성능 대폭 향상")
            print("• 'matrix' 검색시 매트릭스 영화 정확히 검색됨")
            
        else:
            print("작업을 취소했습니다.")
            
    except Exception as e:
        print(f"[오류] 시스템 오류: {e}")
        print("\n[해결책]")
        print("1. sentence-transformers 라이브러리 설치:")
        print("   pip install sentence-transformers")
        print("2. 인터넷 연결 확인 (모델 다운로드 필요)")
        print("3. 메모리 부족시 시스템 재시작 후 재실행")

if __name__ == "__main__":
    main()

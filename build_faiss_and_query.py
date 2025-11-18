#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

import numpy as np
import faiss

def load_metadata(jsonl_path: Path) -> List[Dict[str, Any]]:
    metas = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def ensure_unit_norm(x: np.ndarray) -> np.ndarray:
    # 코사인=내적을 위해 L2 정규화
    x = x.astype('float32', copy=False)
    faiss.normalize_L2(x)
    return x

def build_faiss_index(embeddings: np.ndarray, out_index: Path, out_idmap: Path):
    dim = embeddings.shape[1]
    # 안전망: 혹시 정규화 안 된 임베딩이 들어오면 여기서 정규화
    if not np.allclose((embeddings**2).sum(axis=1).mean(), 1.0, atol=1e-2):
        embeddings = ensure_unit_norm(embeddings)

    index = faiss.IndexFlatIP(dim)  # 코사인=IP (정규화 전제)
    index.add(embeddings)
    faiss.write_index(index, str(out_index))
    id_map = np.arange(embeddings.shape[0], dtype=np.int64)
    np.save(out_idmap, id_map)
    return index

def query_index(index: faiss.Index, q_vecs: np.ndarray, top_k: int = 5):
    return index.search(q_vecs, top_k)

def aggregate_by_movie(sims: np.ndarray, ids: np.ndarray, metas: List[Dict[str, Any]], 
                      top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """윈도우 결과를 영화별로 집계"""
    # 영화별 점수 집계 (딕셔너리 사용)
    movie_scores = {}
    
    for qi in range(sims.shape[0]):  # 쿼리별 처리
        for sid, score in zip(ids[qi], sims[qi]):
            try:
                meta = metas[int(sid)]
                movie_index = meta.get('movie_index')
                title = meta.get('title', f'Unknown_{movie_index}')
                
                if movie_index not in movie_scores:
                    movie_scores[movie_index] = {
                        'scores': [],
                        'meta': {
                            'title': title,
                            'year': meta.get('year', ''),
                            'director': meta.get('director', ''),
                            'movie_index': movie_index
                        }
                    }
                movie_scores[movie_index]['scores'].append(float(score))
            except (IndexError, KeyError) as e:
                print(f"[경고] 메타데이터 처리 오류: {e}")
                continue
    
    # 영화별 최종 점수 계산 (상위 점수들의 가중 평균)
    final_movies = []
    for movie_index, data in movie_scores.items():
        scores = sorted(data['scores'], reverse=True)
        # 상위 3개 윈도우의 가중 평균 (1.0, 0.7, 0.5)
        weights = [1.0, 0.7, 0.5]
        final_score = sum(s * w for s, w in zip(scores[:3], weights[:len(scores)])) / sum(weights[:len(scores)])
        
        final_movies.append({
            'score': final_score,
            'meta': data['meta']
        })
    
    # 점수순 정렬 후 top_k 선택
    final_movies.sort(key=lambda x: x['score'], reverse=True)
    final_movies = final_movies[:top_k]
    
    # numpy 배열 형태로 변환
    final_sims = np.array([[movie['score'] for movie in final_movies]], dtype='float32')
    final_ids = np.array([[i for i in range(len(final_movies))]], dtype='int64')
    final_metas = [movie['meta'] for movie in final_movies]
    
    return final_sims, final_ids, final_metas
    """윈도우 점수를 영화별로 집계"""
    from collections import defaultdict
    
    # 영화별 점수 수집
    movie_scores = defaultdict(list)
    movie_info = {}
    
    for sid, sim in zip(ids[0], sims[0]):
        meta = metas[int(sid)]
        movie_idx = meta.get('movie_index', meta.get('title'))  # movie_index 또는 title로 그룹핑
        
        movie_scores[movie_idx].append(float(sim))
        
        # 영화 정보 저장 (첫 번째 윈도우 정보 사용)
        if movie_idx not in movie_info:
            movie_info[movie_idx] = {
                'title': meta.get('title', 'Unknown'),
                'year': meta.get('year', ''),
                'director': meta.get('director', ''),
                'movie_index': movie_idx
            }
    
    # 영화별 최종 점수 계산 (상위 3개 윈도우 평균)
    movie_final_scores = []
    for movie_idx, scores in movie_scores.items():
        # 상위 3개 점수의 가중 평균 (첫 번째가 가장 높은 가중치)
        scores = sorted(scores, reverse=True)
        if len(scores) >= 3:
            final_score = (scores[0] * 0.5 + scores[1] * 0.3 + scores[2] * 0.2)
        elif len(scores) == 2:
            final_score = (scores[0] * 0.7 + scores[1] * 0.3)
        else:
            final_score = scores[0]
        
        movie_final_scores.append((final_score, movie_idx))
    
    # 점수순 정렬
    movie_final_scores.sort(reverse=True)
    
    # 결과 재구성
    final_sims = np.array([[score for score, _ in movie_final_scores]], dtype='float32')
    final_ids = np.array([[movie_idx for _, movie_idx in movie_final_scores]], dtype='int64')
    final_metas = [movie_info[movie_idx] for _, movie_idx in movie_final_scores]
    
    return final_sims, final_ids, final_metas


# -----------------------------
# LLM 필터링 클래스  
# -----------------------------

class LLMMovieFilter:
    def __init__(self, api_key: str):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.available = True
        except ImportError:
            print("[경고] openai 패키지가 설치되지 않음. LLM 필터링 비활성화")
            self.available = False
        except Exception as e:
            print(f"[경고] OpenAI 클라이언트 초기화 실패: {e}")
            self.available = False

    def filter_search_results(self, query: str, movies: List[Dict], threshold: int = 7) -> List[Dict]:
        """검색 결과를 LLM으로 필터링"""
        if not self.available:
            print("[스킵] LLM 필터링을 사용할 수 없음")
            return movies

        print(f"[LLM 필터링] {len(movies)}개 영화 분석 중...")
        filtered_movies = []
        
        for i, movie in enumerate(movies, 1):
            try:
                print(f"  [{i}/{len(movies)}] {movie.get('title', 'Unknown')} 분석 중...")
                
                judgment = self._analyze_movie_relevance(query, movie)
                
                if judgment['pass'] and judgment['score'] >= threshold:
                    filtered_movies.append({
                        **movie,
                        'llm_score': judgment['score'],
                        'llm_reason': judgment['reason']
                    })
                    print(f"    ✅ PASS (점수: {judgment['score']}/10)")
                else:
                    print(f"    ❌ FAIL (점수: {judgment['score']}/10) - {judgment['reason']}")
                
                time.sleep(0.5)  # API 요청 간격
                
            except Exception as e:
                print(f"    ⚠️ 오류: {e}")
                filtered_movies.append(movie)
        
        print(f"[LLM 필터링 완료] {len(movies)} → {len(filtered_movies)}개")
        return filtered_movies

    def _analyze_movie_relevance(self, query: str, movie: Dict) -> Dict:
        """개별 영화의 검색어 적합성 분석"""
        prompt = self._create_filter_prompt(query, movie)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 영화 추천 시스템의 정확도를 평가하는 전문가입니다. 사용자가 찾는 영화의 특성을 정확히 파악하고, 줄거리의 구체적인 내용을 바탕으로 상세한 분석을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # 더 긴 응답을 위해 증가
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
            
        except Exception as e:
            print(f"[API 오류] {e}")
            return {"pass": True, "score": 5, "reason": "API 오류로 기본 통과"}

    def _create_filter_prompt(self, query: str, movie: Dict) -> str:
        """필터링용 프롬프트 생성"""
        title = movie.get('title', 'Unknown')
        year = movie.get('year', 'Unknown')
        plot = movie.get('plot', '')[:800]  # 더 긴 줄거리 포함
        
        return f"""
검색어: "{query}"

영화 정보:
제목: {title} ({year})
줄거리: {plot}

이 영화가 검색어의 의도에 부합하는지 상세히 평가해주세요.

평가 기준:
1. 검색어의 핵심 키워드/테마가 줄거리에 구체적으로 나타나는가?
2. 검색어의 의도(장르, 감정, 상황, 캐릭터 등)와 영화 내용이 일치하는가?
3. 단순한 키워드 매칭이 아닌 실제 스토리/테마 연관성이 있는가?

반드시 다음 형식으로만 답변하세요:
판정: [PASS 또는 FAIL]
점수: [1-10]
이유: [줄거리의 구체적인 부분을 인용하며 왜 검색어와 연관되는지 2-3문장으로 상세 설명]

예시:
판정: PASS
점수: 8
이유: 줄거리에서 "좀비 바이러스가 전 세계로 퍼져나가며 인류가 감염된 좀비들과 싸우는" 내용이 나타나므로 '좀비' 검색어와 직접적으로 일치합니다. 특히 "좀비 무리들이 도시를 습격하고 생존자들이 안전한 곳을 찾아 도망치는" 장면들이 전형적인 좀비 장르의 핵심 요소를 포함하고 있어 사용자가 원하는 영화와 정확히 부합합니다.
"""

    def _parse_llm_response(self, response: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            pass_match = re.search(r'판정:\s*(PASS|FAIL)', response, re.IGNORECASE)
            score_match = re.search(r'점수:\s*(\d+)', response)
            reason_match = re.search(r'이유:\s*(.+)', response, re.DOTALL)  # 여러 줄 이유 지원
            
            pass_result = pass_match.group(1).upper() == 'PASS' if pass_match else True
            score = int(score_match.group(1)) if score_match else 5
            reason = reason_match.group(1).strip() if reason_match else "분석 완료"
            
            # 이유가 너무 길면 첫 번째 문장들만 사용 (최대 200자)
            if len(reason) > 200:
                sentences = reason.split('. ')
                truncated = sentences[0]
                for sent in sentences[1:]:
                    if len(truncated + '. ' + sent) <= 200:
                        truncated += '. ' + sent
                    else:
                        break
                reason = truncated + '.' if not truncated.endswith('.') else truncated
            
            return {
                "pass": pass_result,
                "score": max(1, min(10, score)),
                "reason": reason
            }
            
        except Exception as e:
            return {"pass": True, "score": 5, "reason": "파싱 오류"}


# -----------------------------
# 웹 연동을 위한 결과 포맷터
# -----------------------------

def format_results_for_web(movies: List[Dict], query: str, llm_filtered: bool = False) -> Dict:
    """웹페이지 표시용 결과 포맷"""
    formatted_results = {
        "query": query,
        "total_results": len(movies),
        "llm_filtered": llm_filtered,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "movies": []
    }
    
    for i, movie in enumerate(movies, 1):
        formatted_movie = {
            "rank": i,
            "title": movie.get('title', 'Unknown'),
            "year": movie.get('year', 'Unknown'),
            "director": movie.get('director', 'Unknown'),
            "plot": movie.get('plot', ''),
            "score": float(movie.get('score', 0.0)),
            "poster_url": f"/static/posters/{movie.get('title', 'default').replace(' ', '_')}.jpg",
            "llm_analysis": {
                "score": movie.get('llm_score'),
                "reason": movie.get('llm_reason')
            } if llm_filtered and 'llm_score' in movie else None
        }
        formatted_results["movies"].append(formatted_movie)
    
    return formatted_results


def save_results_for_web(results: Dict, output_file: str = "web_results.json"):
    """웹 연동용 결과 저장"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[웹 연동] 결과 저장: {output_file}")
    except Exception as e:
        print(f"[오류] 웹 결과 저장 실패: {e}")


def print_results(sims, ids, metas: List[Dict[str, Any]]):
    for qi in range(ids.shape[0]):
        print(f"\n[Query {qi}] Top-{ids.shape[1]} results")
        for rank, (sid, sim) in enumerate(zip(ids[qi], sims[qi]), start=1):
            # sid는 이제 집계된 영화의 인덱스
            m = metas[int(sid)]
            print(f"  {rank:>2}. score={sim:.4f} | {m.get('title')} ({m.get('year')}) | dir={m.get('director')}")

def interactive_search():
    """대화형 검색 모드"""
    import os
    from dotenv import load_dotenv
    from weighted_search_utils import get_weighted_helper
    
    # 환경변수 로드
    load_dotenv()
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print("=" * 60)
    print("🎬 영화 추천 시스템")
    print("=" * 60)
    print("1. 기존 검색 (청킹 기반 + LLM 필터링)")
    print("2. 가중치 조절 검색 (분리 임베딩)")
    print("=" * 60)
    
    search_mode = input("검색 모드를 선택하세요 (1 또는 2): ").strip()
    
    if search_mode == "2":
        weighted_interactive_search()
        return
    
    print("=" * 60)
    print("🎬 기존 영화 추천 시스템 (SentenceBERT + LLM 필터링)")
    print("=" * 60)
    print("• 20개 후보 검색 → LLM 검증 → 최종 5개 추천")
    print("• 'quit' 또는 'exit' 입력시 종료")
    print("=" * 60)
    
    # 필요한 파일들 체크
    data_dir = Path('data')
    required_files = {
        'embeddings.npy': data_dir / 'embeddings.npy',
        'metadata.jsonl': data_dir / 'metadata.jsonl', 
        'index.faiss': data_dir / 'index.faiss'
    }
    
    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        print(f"❌ 필요한 파일들이 없습니다: {missing_files}")
        print("먼저 다음 명령어들을 실행하세요:")
        print("1. python create_chunk_embeddings.py")
        print("2. python build_faiss_and_query.py --build")
        return
    
    # SentenceTransformer 모델 로드
    print("📁 모델 및 데이터 로딩 중...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
        
        # 데이터 로드
        embeddings = np.load(required_files['embeddings.npy'])
        metas = load_metadata(required_files['metadata.jsonl'])
        index = faiss.read_index(str(required_files['index.faiss']))
        
        print(f"✅ 로딩 완료: {embeddings.shape[0]}개 윈도우, {len(set(m.get('movie_index') for m in metas))}개 영화")
        
        # LLM 필터 초기화
        llm_filter = None
        if openai_key:
            llm_filter = LLMMovieFilter(openai_key)
            if llm_filter.available:
                print("✅ LLM 필터링 활성화")
            else:
                print("⚠️ LLM 필터링 비활성화 (API 키 문제)")
        else:
            print("⚠️ LLM 필터링 비활성화 (OPENAI_API_KEY 없음)")
            
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 대화형 검색 루프
    while True:
        try:
            print("\n" + "─" * 40)
            query = input("🔍 어떤 영화를 찾고 계신가요? > ").strip()
            
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 시스템을 종료합니다.")
                break
                
            if not query:
                print("❌ 검색어를 입력해주세요.")
                continue
            
            print(f"\n🔍 '{query}' 검색 중...")
            
            # 쿼리 임베딩 생성
            with torch.inference_mode():
                q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
            q = ensure_unit_norm(q)
            
            # FAISS 검색 (20개)
            print("📊 벡터 검색 중...")
            sims, ids = query_index(index, q, top_k=60)  # 넉넉히 검색
            
            # 영화별 집계 (20개)
            print("🎯 영화별 점수 집계 중...")
            final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, metas, top_k=20)
            
            # 메타데이터를 리스트 형태로 변환
            candidate_movies = []
            for i, meta in enumerate(final_metas):
                candidate_movies.append({
                    **meta,
                    'score': float(final_sims[0][i])
                })
            
            # LLM 필터링 (5개)
            final_movies = candidate_movies
            if llm_filter and llm_filter.available:
                print("🤖 LLM 검증 중...")
                final_movies = llm_filter.filter_search_results(
                    query=query,
                    movies=candidate_movies[:20],  # 상위 20개 LLM 검증
                    threshold=6  # 임계값을 6으로 낮춤
                )
                
                if not final_movies:
                    print("⚠️ LLM 검증을 통과한 영화가 없어서 원본 결과를 표시합니다.")
                    final_movies = candidate_movies  # 모든 후보 표시
            else:
                final_movies = candidate_movies  # LLM 없으면 모든 후보
            
            # 결과 출력
            print(f"\n🎬 '{query}' 검색 결과")
            print("=" * 60)
            
            if not final_movies:
                print("❌ 검색 결과가 없습니다.")
                continue
            
            for i, movie in enumerate(final_movies, 1):  # 5개 제한 해제
                title = movie.get('title', 'Unknown')
                year = movie.get('year', 'Unknown') 
                director = movie.get('director', 'Unknown')
                score = movie.get('score', 0)
                
                print(f"\n{i}. {title} ({year})")
                print(f"   감독: {director}")
                print(f"   유사도: {score:.3f}")
                
                # LLM 분석 결과
                if 'llm_score' in movie:
                    llm_score = movie.get('llm_score', 0)
                    llm_reason = movie.get('llm_reason', '')
                    print(f"   LLM 점수: {llm_score}/10")
                    print(f"   추천 이유: {llm_reason}")
            
            # 웹 연동용 결과 저장
            web_results = format_results_for_web(
                movies=final_movies,
                query=query,
                llm_filtered=(llm_filter and llm_filter.available)
            )
            save_results_for_web(web_results)
            
        except KeyboardInterrupt:
            print("\n👋 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 검색 중 오류: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--build', action='store_true', help='FAISS 인덱스 생성')
    ap.add_argument('--demo_query', type=str, default='')
    ap.add_argument('--top_k', type=int, default=5)
    ap.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # 🆕 LLM 필터링 옵션들
    ap.add_argument('--llm_filter', action='store_true', help='LLM 기반 결과 필터링 활성화')
    ap.add_argument('--openai_key', type=str, help='OpenAI API 키')
    ap.add_argument('--llm_threshold', type=int, default=7, help='LLM 필터링 임계값 (1-10)')
    ap.add_argument('--save_web', action='store_true', help='웹 연동용 결과 저장')
    args = ap.parse_args()
    
    # 🆕 인수가 없으면 대화형 모드 실행
    import sys
    if len(sys.argv) == 1:
        interactive_search()
        return

    data_dir = Path(args.data_dir)
    emb_path = data_dir / 'embeddings.npy'
    meta_path = data_dir / 'metadata.jsonl'
    index_path = data_dir / 'index.faiss'
    idmap_path = data_dir / 'id_map.npy'

    embeddings = np.load(emb_path)          # (N, D)
    metas = load_metadata(meta_path)

    if args.build:
        print('[1/3] Building FAISS index')
        build_faiss_index(embeddings, index_path, idmap_path)
        print(f'✅ Saved index to {index_path} and id_map to {idmap_path}')

    if args.demo_query:
        print('[2/3] Encoding demo query')
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)

        with torch.inference_mode():
            q = model.encode([args.demo_query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
        q = ensure_unit_norm(q)  # 쿼리도 정규화

        # 차원 검증(모델 통일 확인)
        if q.shape[1] != embeddings.shape[1]:
            raise ValueError(f"Dim mismatch: query({q.shape[1]}) vs corpus({embeddings.shape[1]}). "
                             f"--model을 임베딩 생성 시 사용한 것과 동일하게 맞추세요.")

        print('[3/3] Searching')
        index = faiss.read_index(str(index_path))
        sims, ids = query_index(index, q, top_k=args.top_k * 3)  # 더 많이 검색해서 집계

        # 윈도우 점수를 영화별로 집계
        print('[4/4] Aggregating by movie')
        final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, metas, top_k=args.top_k)
        
        # 🆕 LLM 필터링 (선택적)
        if args.llm_filter:
            if not args.openai_key:
                print("[오류] LLM 필터링을 위해서는 --openai_key가 필요합니다")
            else:
                print('[5/5] LLM Filtering')
                # 메타데이터를 리스트 형태로 변환
                movies_for_filtering = []
                for i, meta in enumerate(final_metas):
                    movies_for_filtering.append({
                        **meta,
                        'score': float(final_sims[0][i])
                    })
                
                # LLM 필터링 실행
                llm_filter = LLMMovieFilter(args.openai_key)
                filtered_movies = llm_filter.filter_search_results(
                    query=args.demo_query,
                    movies=movies_for_filtering,
                    threshold=args.llm_threshold
                )
                
                # 필터링된 결과를 다시 배열 형태로 변환
                if filtered_movies:
                    final_sims = np.array([[movie['score'] for movie in filtered_movies]], dtype='float32')
                    final_ids = np.array([[i for i in range(len(filtered_movies))]], dtype='int64')
                    final_metas = filtered_movies
                else:
                    print("[경고] LLM 필터링 후 결과가 없습니다")
        
        # 결과 출력
        print_results(final_sims, final_ids, final_metas)
        
        # 🆕 웹 연동용 결과 저장 (선택적)
        if args.save_web:
            # 웹 표시용 포맷으로 변환
            movies_for_web = []
            for i, meta in enumerate(final_metas):
                movies_for_web.append({
                    **meta,
                    'score': float(final_sims[0][i])
                })
            
            web_results = format_results_for_web(
                movies=movies_for_web,
                query=args.demo_query,
                llm_filtered=args.llm_filter
            )
            save_results_for_web(web_results)
            print(f"[웹 연동] 포스터 경로: /static/posters/")
            print(f"[웹 연동] 흐름곡선 데이터도 추가 가능")

    data_dir = Path(args.data_dir)
    emb_path = data_dir / 'embeddings.npy'
    meta_path = data_dir / 'metadata.jsonl'
    index_path = data_dir / 'index.faiss'
    idmap_path = data_dir / 'id_map.npy'

    embeddings = np.load(emb_path)          # (N, D)
    metas = load_metadata(meta_path)

    if args.build:
        print('[1/3] Building FAISS index')
        build_faiss_index(embeddings, index_path, idmap_path)
        print(f'✅ Saved index to {index_path} and id_map to {idmap_path}')

    if args.demo_query:
        print('[2/3] Encoding demo query')
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)

        with torch.inference_mode():
            q = model.encode([args.demo_query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
        q = ensure_unit_norm(q)  # 쿼리도 정규화

        # 차원 검증(모델 통일 확인)
        if q.shape[1] != embeddings.shape[1]:
            raise ValueError(f"Dim mismatch: query({q.shape[1]}) vs corpus({embeddings.shape[1]}). "
                             f"--model을 임베딩 생성 시 사용한 것과 동일하게 맞추세요.")

        print('[3/3] Searching')
        index = faiss.read_index(str(index_path))
        sims, ids = query_index(index, q, top_k=args.top_k * 3)  # 더 많이 검색해서 집계

        # 윈도우 점수를 영화별로 집계
        print('[4/4] Aggregating by movie')
        final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, metas, top_k=args.top_k)
        
        # 🆕 LLM 필터링 (선택적)
        if args.llm_filter:
            if not args.openai_key:
                print("[오류] LLM 필터링을 위해서는 --openai_key가 필요합니다")
            else:
                print('[5/5] LLM Filtering')
                # 메타데이터를 리스트 형태로 변환
                movies_for_filtering = []
                for i, meta in enumerate(final_metas):
                    movies_for_filtering.append({
                        **meta,
                        'score': float(final_sims[0][i])
                    })
                
                # LLM 필터링 실행
                llm_filter = LLMMovieFilter(args.openai_key)
                filtered_movies = llm_filter.filter_search_results(
                    query=args.demo_query,
                    movies=movies_for_filtering,
                    threshold=args.llm_threshold
                )
                
                # 필터링된 결과를 다시 배열 형태로 변환
                if filtered_movies:
                    final_sims = np.array([[movie['score'] for movie in filtered_movies]], dtype='float32')
                    final_ids = np.array([[i for i in range(len(filtered_movies))]], dtype='int64')
                    final_metas = filtered_movies
                else:
                    print("[경고] LLM 필터링 후 결과가 없습니다")
        
        # 결과 출력
        print_results(final_sims, final_ids, final_metas)
        
        # 🆕 웹 연동용 결과 저장 (선택적)
        if args.save_web:
            # 웹 표시용 포맷으로 변환
            movies_for_web = []
            for i, meta in enumerate(final_metas):
                movies_for_web.append({
                    **meta,
                    'score': float(final_sims[0][i])
                })
            
            web_results = format_results_for_web(
                movies=movies_for_web,
                query=args.demo_query,
                llm_filtered=args.llm_filter
            )
            save_results_for_web(web_results)
            print(f"[웹 연동] 포스터 경로: /static/posters/")
            print(f"[웹 연동] 흐름곡선 데이터도 추가 가능")

def weighted_interactive_search():
    """가중치 조절 대화형 검색"""
    print("=" * 60)
    print("🎭 가중치 조절 영화 검색 시스템")
    print("=" * 60)
    print("• 이 기능은 movie_similarity_finder.py에서 이용하세요")
    print("• 현재는 기본 청킹 검색만 지원합니다")
    print("=" * 60)
    
    print("💡 movie_similarity_finder.py를 실행하여 가중치 조절 검색을 이용하세요!")
    return

if __name__ == '__main__':
    main()
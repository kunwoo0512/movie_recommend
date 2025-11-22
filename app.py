#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Application for Movie Recommendation System
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import json
import subprocess
import sys
import signal
import atexit
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
import time
import threading
from queue import Queue

# build_faiss_and_query.py의 기존 함수들 import
from build_faiss_and_query import (
    load_metadata, 
    ensure_unit_norm,
    query_index, 
    aggregate_by_movie,
    LLMMovieFilter,
    format_results_for_web
)

# 환경변수 로드
load_dotenv()

# 글로벌 모델 변수들 (서버 시작시 로딩)
global_index = None
global_metadata = None
global_models_loaded = False
global_sentence_model = None  # 임베딩 모델 사전 로딩

def load_models_on_startup():
    """서버 시작시 모델과 데이터를 사전 로딩"""
    global global_index, global_metadata, global_models_loaded, global_sentence_model
    
    print("🔄 [시작] 모델 및 데이터 사전 로딩 중...")
    start_time = time.time()
    
    try:
        # FAISS 인덱스 로드 (build_faiss_and_query.py 함수 사용)
        index_path = Path('data/index.faiss')
        if index_path.exists():
            import faiss
            global_index = faiss.read_index(str(index_path))
            print(f"✅ FAISS 인덱스 로드 완료: {global_index.ntotal}개 벡터")
        
        # 메타데이터 로드 (build_faiss_and_query.py 함수 사용)
        metadata_path = Path('data/metadata.jsonl')
        if metadata_path.exists():
            global_metadata = load_metadata(metadata_path)  # build_faiss_and_query.py 함수 사용
            print(f"✅ 메타데이터 로드 완료: {len(global_metadata)}개 영화")
        
        # 임베딩 모델도 서버 시작 시 미리 로딩
        from sentence_transformers import SentenceTransformer
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🤖 임베딩 모델 로딩 중... (device: {device})")
        global_sentence_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device=device
        )
        print("✅ 임베딩 모델 로드 완료")
        
        global_models_loaded = True
        load_time = time.time() - start_time
        
        # 메모리 사용량 체크 (선택적)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"💾 현재 메모리 사용량: {memory_mb:.1f}MB")
        except ImportError:
            print("💾 메모리 모니터링을 위해 'pip install psutil' 실행 가능")
        except Exception as e:
            print(f"💾 메모리 체크 오류: {e}")
            
        print(f"🎉 [완료] 모델 사전 로딩 완료! ({load_time:.2f}초)")
        
    except Exception as e:
        print(f"❌ [오류] 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        global_models_loaded = False

# 활성 프로세스들을 추적하기 위한 리스트 (간단한 버전)
active_processes = []

def cleanup_on_exit():
    """프로그램 종료 시 남은 프로세스들 정리"""
    global active_processes
    for process in active_processes[:]:  # 복사본으로 순회
        try:
            if process.poll() is None:  # 아직 실행 중인 프로세스
                print(f"🧹 정리 중: PID {process.pid}")
                process.terminate()
                try:
                    process.wait(timeout=3)  # 3초 대기
                except subprocess.TimeoutExpired:
                    process.kill()  # 강제 종료
            active_processes.remove(process)
        except Exception as e:
            print(f"⚠️ 프로세스 정리 오류: {e}")

# 프로그램 종료 시 정리 함수 등록
atexit.register(cleanup_on_exit)

# Ctrl+C 처리
def signal_handler(signum, frame):
    print(f"\n🛑 종료 시그널 수신 - 정리 중...")
    cleanup_on_exit()
    sys.exit(0)

if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, signal_handler)

app = Flask(__name__)

# 정적 파일 설정
app.static_folder = 'static'
app.template_folder = 'templates'

@app.route('/')
def index():
    """메인 페이지"""
    print("🏠 [DEBUG] 메인 페이지 접속!")
    
    # 모델이 로딩되지 않았다면 지금 로딩
    if not global_models_loaded:
        print("📚 [DEBUG] 모델이 아직 로딩되지 않음 - 지금 로딩합니다")
        load_models_on_startup()
    else:
        print("✅ [DEBUG] 모델이 이미 로딩되어 있습니다")
    
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_movies():
    """영화 검색 API"""
    print("🔥 [DEBUG] /search 엔드포인트 진입!")
    try:
        print("🔥 [DEBUG] 요청 데이터 파싱 시작")
        data = request.get_json()
        query = data.get('query', '').strip()
        print(f"🔥 [DEBUG] 받은 검색어: '{query}'")
        
        if not query:
            print("🔥 [DEBUG] 검색어가 비어있음")
            return jsonify({'error': '검색어를 입력해주세요.'}), 400
        
        # build_faiss_and_query.py 실행
        print(f"[검색] '{query}' 검색 시작...")
        
        # 환경변수에서 OpenAI API 키 가져오기
        openai_key = os.getenv('OPENAI_API_KEY')
        print(f"🔥 [DEBUG] OPENAI_API_KEY 존재 여부: {bool(openai_key)}")
        if openai_key:
            print(f"🔥 [DEBUG] OPENAI_API_KEY 앞 8글자: {openai_key[:8]}...")
        if not openai_key:
            print("🔥 [DEBUG] API 키 없음 - 500 에러 반환")
            print("[경고] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다.'}), 500
        
        cmd_args = [
            sys.executable, 'build_faiss_and_query.py',
            '--demo_query', query,
            '--llm_filter',
            '--save_web',
            '--openai_key', openai_key,
            '--top_k', '20'  # 적절한 후보군 수, LLM이 필터링함
        ]
        
        print(f"🔥 [DEBUG] subprocess 실행: {' '.join(cmd_args[:6])}...")
        
        # 환경변수에 UTF-8 설정 추가
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'  # Python 3.7+ UTF-8 모드
        
        try:
            # 하이브리드 방식: Popen으로 프로세스 추적하되 안전하게 처리
            process = subprocess.Popen(
                cmd_args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env,
                encoding='utf-8',  # 명시적 인코딩 설정
                errors='ignore'    # 인코딩 에러 무시
            )
            
            # 프로세스를 리스트에 추가하여 추적
            global active_processes
            active_processes.append(process)
            
            # 프로세스 완료까지 대기 (타임아웃 포함)
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5분 타임아웃
            except subprocess.TimeoutExpired:
                print(f"🔥 [DEBUG] subprocess 타임아웃 - 강제 종료")
                process.kill()
                stdout, stderr = process.communicate()
                return jsonify({'error': '검색 시간이 너무 오래 걸립니다. 다시 시도해주세요.'}), 500
            
            # 완료된 프로세스는 리스트에서 제거
            if process in active_processes:
                active_processes.remove(process)
                
            print(f"🔥 [DEBUG] subprocess 종료 코드: {process.returncode}")
            if stderr:
                print(f"🔥 [DEBUG] subprocess stderr: {stderr[:200]}...")
            if stdout:
                print(f"🔥 [DEBUG] subprocess stdout: {stdout[:200]}...")
            
            if process.returncode != 0:
                print(f"🔥 [DEBUG] subprocess 실패 - 500 에러 반환")
                print(f"[오류] 검색 실패: {stderr}")
                return jsonify({'error': '검색 중 오류가 발생했습니다.'}), 500
                
        except subprocess.TimeoutExpired:
            print(f"🔥 [DEBUG] subprocess 타임아웃 (5분 초과)")
            return jsonify({'error': '검색 시간이 너무 오래 걸립니다. 다시 시도해주세요.'}), 500
        except Exception as e:
            print(f"🔥 [DEBUG] subprocess 실행 중 예외: {e}")
            return jsonify({'error': f'검색 프로세스 실행 오류: {str(e)}'}), 500
        
        # 결과 로드
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'error': '검색 결과가 생성되지 않았습니다.'}), 500
        
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # LLM 분석 결과 확인을 위한 디버그 출력
        movies = results.get('movies', [])
        llm_count = 0
        for movie in movies:
            if movie and isinstance(movie, dict):
                title = movie.get('title')
                llm_analysis = movie.get('llm_analysis')
                if llm_analysis and isinstance(llm_analysis, dict):
                    llm_reason = llm_analysis.get('reason', '')
                    if title and llm_reason:
                        llm_count += 1
                        print(f"[디버그] LLM 분석 완료: {title}")
        
        print(f"[성공] {len(results['movies'])}개 영화 검색 완료, {llm_count}개 LLM 분석 완료")
        return jsonify(results)
        
    except Exception as e:
        print(f"[예외] 검색 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stream-search', methods=['GET'])
def stream_search():
    """실시간 스트리밍 검색 - LLM 분석 완료되는 대로 결과 전송"""
    # request context가 활성화된 상태에서 파라미터 추출
    query = request.args.get('query', '').strip()
    
    def generate_stream():
        try:
            if not query:
                yield f"data: {json.dumps({'error': '검색어를 입력해주세요.'})}\n\n"
                return
            
            # 초기 상태 전송
            yield f"data: {json.dumps({'status': 'searching', 'message': f'검색 중: {query}'})}\n\n"
            
            # subprocess 실행 (기존 로직과 동일하지만 실시간 처리)
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                yield f"data: {json.dumps({'error': 'OpenAI API 키가 설정되지 않았습니다.'})}\n\n"
                return
            
            cmd_args = [
                sys.executable, 'build_faiss_and_query.py',
                '--demo_query', query,
                '--llm_filter',
                '--save_web',
                '--openai_key', openai_key,
                '--top_k', '20'
            ]
            
            # 환경변수 설정
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # 프로세스 실행
            process = subprocess.Popen(
                cmd_args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env,
                encoding='utf-8',
                errors='ignore'
            )
            
            global active_processes
            active_processes.append(process)
            
            # 프로세스 완료 대기
            try:
                stdout, stderr = process.communicate(timeout=300)
                
                if process in active_processes:
                    active_processes.remove(process)
                
                if process.returncode == 0:
                    # 완료 후 결과 파일 읽기
                    web_results_path = Path('web_results.json')
                    if web_results_path.exists():
                        with open(web_results_path, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        # 각 영화를 개별적으로 전송 (스트리밍 효과)
                        movies = results.get('movies', [])
                        for i, movie in enumerate(movies):
                            yield f"data: {json.dumps({'type': 'movie', 'data': movie, 'index': i, 'total': len(movies)})}\n\n"
                            time.sleep(0.1)  # 스트리밍 효과를 위한 약간의 지연
                        
                        yield f"data: {json.dumps({'status': 'completed', 'message': '검색 완료', 'total_results': len(movies)})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': '검색 결과를 찾을 수 없습니다.'})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': '검색 중 오류가 발생했습니다.'})}\n\n"
                    
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                yield f"data: {json.dumps({'error': '검색 시간이 너무 오래 걸립니다.'})}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/streaming-search', methods=['POST'])
def streaming_search():
    """실시간 LLM 분석 스트리밍 - 분석 완료된 영화를 하나씩 전송"""
    try:
        print("🎬 [스트리밍] 실시간 LLM 분석 시작!")
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': '검색어를 입력해주세요.'}), 400
        
        # OpenAI API 키 확인
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다.'}), 500

        def generate_streaming_results():
            global global_sentence_model  # ⭐ 전역 변수 선언 추가
            try:
                # 1. FAISS 검색 (빠른 후보 선별)
                yield f"data: {json.dumps({'status': 'searching', 'message': 'FAISS 검색 중...'})}\n\n"
                
                # 임베딩 모델 사용 (이미 사전 로딩됨)
                if global_sentence_model is None:
                    yield f"data: {json.dumps({'status': 'loading', 'message': '임베딩 모델 로딩 중...'})}\n\n"
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    global_sentence_model = SentenceTransformer(
                        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        device=device
                    )

                model = global_sentence_model
                
                # 임베딩 생성 및 검색
                import torch
                with torch.inference_mode():
                    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
                q = ensure_unit_norm(q)
                
                # FAISS 검색
                top_k = 20
                top_k_search = top_k * 2  # 여유 있게 검색
                sims, ids = query_index(global_index, q, top_k=top_k_search)
                
                # 영화별 집계
                final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, global_metadata, top_k=top_k)
                
                yield f"data: {json.dumps({'status': 'llm_start', 'message': f'LLM 분석 시작 ({len(final_metas)}개 영화)', 'total_movies': len(final_metas)})}\n\n"
                
                # 2. 🌟 실시간 LLM 분석 및 스트리밍
                llm_filter = LLMMovieFilter(openai_key)
                approved_count = 0
                
                for i, meta in enumerate(final_metas):
                    try:
                        movie_data = {
                            **meta,
                            'score': float(final_sims[0][i])
                        }
                        
                        # 분석 진행 상황 전송
                        title = movie_data.get('title', 'Unknown')
                        message = f'[{i+1}/{len(final_metas)}] {title} 분석 중...'
                        yield f"data: {json.dumps({'status': 'analyzing', 'message': message, 'current': i+1, 'total': len(final_metas)})}\n\n"
                        
                        # LLM 분석
                        judgment = llm_filter._analyze_movie_relevance(query, movie_data)
                        
                        if judgment['pass'] and judgment['score'] >= 7:  # 통과한 영화만
                            approved_count += 1
                            
                            # 포맷팅
                            formatted_movie = {
                                "title": movie_data.get('title', ''),
                                "year": movie_data.get('year', ''),
                                "director": movie_data.get('director', ''),
                                "score": movie_data.get('score', 0),
                                "poster_url": f"/static/posters/{movie_data.get('title', '').replace(':', '').replace('/', '_')}.jpg",
                                "llm_analysis": {
                                    "score": judgment['score'],
                                    "reason": judgment['reason']
                                }
                            }
                            
                            # ✨ 통과한 영화를 즉시 전송!
                            yield f"data: {json.dumps({'type': 'approved_movie', 'data': formatted_movie, 'approved_count': approved_count, 'progress': i+1, 'total': len(final_metas)})}\n\n"
                            print(f"    ✅ [PASS] {movie_data.get('title')} (점수: {judgment['score']}/10) - 즉시 전송됨")
                        else:
                            print(f"    ❌ [FAIL] {movie_data.get('title')} (점수: {judgment['score']}/10)")
                        
                        # API 제한 고려
                        time.sleep(0.4)
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'status': 'error', 'message': f'영화 {i+1} 분석 오류: {str(e)}'})}\n\n"
                
                # 완료 메시지
                yield f"data: {json.dumps({'status': 'completed', 'message': f'검색 완료: {approved_count}개 영화 추천', 'approved_count': approved_count})}\n\n"
                print(f"🎉 [스트리밍 완료] 총 {approved_count}개 영화 승인됨")
                
            except Exception as e:
                yield f"data: {json.dumps({'error': f'스트리밍 검색 오류: {str(e)}'})}\n\n"
        
        return Response(stream_with_context(generate_streaming_results()), 
                       mimetype='text/event-stream',
                       headers={
                           'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'
                       })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fast-search', methods=['POST'])
def fast_search():
    """사전 로딩된 모델을 사용한 빠른 검색"""
    try:
        print("\n" + "="*50)
        print("🔥 [DEBUG] /fast-search 엔드포인트 진입!")
        print("🔥 [DEBUG] 요청 메서드:", request.method)
        print("🔥 [DEBUG] 요청 헤더:", dict(request.headers))
        
        data = request.get_json()
        print("🔥 [DEBUG] 받은 JSON 데이터:", data)
        
        query = data.get('query', '').strip() if data else ''
        print(f"🔥 [DEBUG] 추출된 검색어: '{query}'")
        
        if not query:
            print("🔥 [DEBUG] 검색어가 비어있음")
            return jsonify({'error': '검색어를 입력해주세요.'}), 400
        
        # 모델 로딩 확인
        print(f"🔥 [DEBUG] global_models_loaded: {global_models_loaded}")
        print(f"🔥 [DEBUG] global_index is None: {global_index is None}")
        print(f"🔥 [DEBUG] global_metadata is None: {global_metadata is None}")
        
        if not global_models_loaded or global_index is None or global_metadata is None:
            print("🔥 [DEBUG] 모델이 로딩되지 않음 - 폴백 검색")
            return handle_regular_search_fallback(query)
        
        print(f"🔥 [DEBUG] 사전 로딩된 모델 사용 - 빠른 검색 시작")
        print("="*50 + "\n")
        
        # 직접 검색 실행 (subprocess 없이)
        return perform_direct_search(query)
        
    except Exception as e:
        print(f"[예외] 빠른 검색 오류: {e}")
        return jsonify({'error': str(e)}), 500

def perform_direct_search(query):
    """build_faiss_and_query.py의 함수들을 사용한 직접 검색"""
    global global_sentence_model  # ⭐ 전역 변수 선언 추가
    
    try:
        start_time = time.time()
        print(f"🚀 [직접 검색] '{query}' 검색 시작...")
        
        # OpenAI API 키 확인
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다.'}), 500
        
        # 1. 임베딩 생성 (사전 로딩된 모델 사용)
        embed_start = time.time()
        
        if global_sentence_model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🤖 임베딩 모델 로딩 중... (device: {device})")
            global_sentence_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            print("✅ 임베딩 모델 로드 완료")

        model = global_sentence_model
        
        # 임베딩 생성 및 정규화 (build_faiss_and_query.py와 동일한 방식)
        import torch
        with torch.inference_mode():
            q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False).astype('float32')
        q = ensure_unit_norm(q)  # build_faiss_and_query.py의 정규화 함수 사용
        
        embed_time = time.time() - embed_start
        print(f"⚡ 임베딩 생성 완료: {embed_time:.2f}초")
        
        # 2. FAISS 검색 (build_faiss_and_query.py와 동일한 방식)
        search_start = time.time()
        top_k = 20  # 최종 결과 개수
        top_k_search = top_k * 3  # build_faiss_and_query.py와 동일: 더 많이 검색해서 집계
        
        sims, ids = query_index(global_index, q, top_k=top_k_search)  # build_faiss_and_query.py 함수 사용
        search_time = time.time() - search_start
        print(f"⚡ FAISS 검색 완료: {search_time:.2f}초 (초기 {top_k_search}개)")
        
        # 3. 영화별 집계 (build_faiss_and_query.py의 정확한 함수 사용)
        aggregate_start = time.time()
        final_sims, final_ids, final_metas = aggregate_by_movie(sims, ids, global_metadata, top_k=top_k)
        aggregate_time = time.time() - aggregate_start
        print(f"⚡ 영화별 집계 완료: {aggregate_time:.2f}초 (최종 {len(final_metas)}개)")
        
        # 4. LLM 필터링 (build_faiss_and_query.py와 동일한 방식)
        llm_start = time.time()
        print(f"🤖 LLM 필터링 시작... ({len(final_metas)}개 영화)")
        
        # 메타데이터를 LLM 필터링용 형식으로 변환
        movies_for_filtering = []
        for i, meta in enumerate(final_metas):
            movies_for_filtering.append({
                **meta,
                'score': float(final_sims[0][i])
            })
        
        # LLM 필터링 실행 (build_faiss_and_query.py 함수 사용)
        llm_filter = LLMMovieFilter(openai_key)
        filtered_movies = llm_filter.filter_search_results(
            query=query,
            movies=movies_for_filtering,
            threshold=7
        )
        
        llm_time = time.time() - llm_start
        print(f"⚡ LLM 필터링 완료: {llm_time:.2f}초 ({len(final_metas)} → {len(filtered_movies)}개)")
        
        # 5. 웹 결과 포맷팅 (build_faiss_and_query.py 함수 사용)
        format_start = time.time()
        web_results = format_results_for_web(
            movies=filtered_movies,
            query=query,
            llm_filtered=True
        )
        format_time = time.time() - format_start
        print(f"⚡ 결과 포맷팅 완료: {format_time:.2f}초")
        
        # 총 시간 출력
        total_time = time.time() - start_time
        print(f"🎉 [직접 검색 완료] 총 시간: {total_time:.2f}초")
        print(f"   - 임베딩: {embed_time:.2f}초")
        print(f"   - FAISS 검색: {search_time:.2f}초") 
        print(f"   - 영화 집계: {aggregate_time:.2f}초")
        print(f"   - LLM 필터링: {llm_time:.2f}초")
        print(f"   - 결과 포맷팅: {format_time:.2f}초")
        
        return web_results
        
    except Exception as e:
        print(f"❌ [오류] 직접 검색 실패: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def handle_regular_search_fallback(query):
    """기존 검색 로직 (폴백용)"""
    try:
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다.'}), 500
        
        cmd_args = [
            sys.executable, 'build_faiss_and_query.py',
            '--demo_query', query,
            '--llm_filter',
            '--save_web',
            '--openai_key', openai_key,
            '--top_k', '20'
        ]
        
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        
        result = subprocess.run(
            cmd_args, 
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            env=env,
            timeout=300,
            encoding='utf-8',
            errors='ignore'
        )
            
        if result.returncode != 0:
            return jsonify({'error': '검색 중 오류가 발생했습니다.'}), 500
        
        # 결과 로드
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'error': '검색 결과가 생성되지 않았습니다.'}), 500
        
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def explain_movie(movie_title):
    """영화 추천 이유 설명 (제목 기반) - JSON 파일에서 직접 읽기"""
    try:
        # web_results.json에서 영화 설명 읽기
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'error': '검색 결과가 없습니다.'}), 404
            
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # URL 디코딩
        movie_title = movie_title.strip()
        
        print(f"[디버그] 요청된 영화: '{movie_title}'")
        
        # 검색 결과에서 해당 제목의 영화 찾기
        movies = results.get('movies', [])
        for movie in movies:
            if movie and isinstance(movie, dict):
                title = movie.get('title', '')
                if title == movie_title:
                    llm_analysis = movie.get('llm_analysis')
                    if llm_analysis and isinstance(llm_analysis, dict):
                        explanation = llm_analysis.get('reason', '')
                        if explanation:
                            return jsonify({
                                'title': title,
                                'explanation': explanation
                            })
        
        # 찾지 못한 경우
        return jsonify({'error': f'"{movie_title}" 영화의 설명을 찾을 수 없습니다.'}), 404
        
    except Exception as e:
        print(f"[예외] 설명 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/movie/<int:movie_rank>')
def movie_detail(movie_rank):
    """영화 상세 정보 페이지"""
    try:
        # 검색 결과에서 영화 정보 로드
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return render_template('error.html', message='검색 결과를 찾을 수 없습니다.')
        
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        movies = results.get('movies', [])
        if movie_rank < 1 or movie_rank > len(movies):
            return render_template('error.html', message='유효하지 않은 영화입니다.')
        
        movie = movies[movie_rank - 1]
        return render_template('movie_detail.html', movie=movie, query=results.get('query'))
        
    except Exception as e:
        print(f"[예외] 영화 상세 조회 오류: {e}")
        return render_template('error.html', message='오류가 발생했습니다.')

@app.route('/explanation/<movie_title>')
def get_explanation_by_title(movie_title):
    """영화 제목으로 추천 이유 조회 - JSON 파일에서 직접 읽기"""
    return explain_movie(movie_title)  # 위의 함수를 재사용

@app.route('/explanations')
def get_all_explanations():
    """모든 영화 설명 데이터 조회 - JSON 파일에서 직접 읽기"""
    try:
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'explanations': []})
            
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        explanations = []
        movies = results.get('movies', [])
        for movie in movies:
            if movie and isinstance(movie, dict):
                title = movie.get('title', '')
                llm_analysis = movie.get('llm_analysis')
                if llm_analysis and isinstance(llm_analysis, dict):
                    explanation = llm_analysis.get('reason', '')
                    if title and explanation:
                        explanations.append([title, explanation])
        
        return jsonify({
            'explanations': explanations,
            'count': len(explanations)
        })
    except Exception as e:
        print(f"[예외] 전체 설명 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# 정적 파일 제공을 위한 라우트
@app.route('/static/posters/<filename>')
def serve_poster(filename):
    """포스터 이미지 제공 (기본 이미지로 대체)"""
    # 실제 포스터가 없으면 기본 이미지 반환
    return app.send_static_file(f'images/default_poster.jpg')

if __name__ == '__main__':
    # 필요한 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    print("🎬 영화 추천 웹 서버 시작")
    print("📍 http://localhost:5000 에서 접속하세요")
    print("⚠️  서버 종료 시 Ctrl+C를 눌러주세요")
    
    # 🚀 모델 사전 로딩은 첫 요청에서 실행
    # with app.app_context():
    #     load_models_on_startup()
    
    try:
        # 프로세스 중복 방지를 위해 use_reloader=False 설정
        # debug=True는 유지하되 auto-reload 기능 비활성화
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        # 강제 종료로 확실하게 정리
        os._exit(0)
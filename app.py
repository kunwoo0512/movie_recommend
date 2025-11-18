#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web Application for Movie Recommendation System
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import subprocess
import sys
import signal
import atexit
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 실행 중인 자식 프로세스들을 추적하기 위한 리스트
active_processes = []

def cleanup_processes():
    """모든 자식 프로세스를 정리하는 함수"""
    global active_processes
    for process in active_processes:
        try:
            if process.poll() is None:  # 프로세스가 아직 실행 중인 경우
                print(f"🧹 정리 중: 프로세스 PID {process.pid}")
                process.terminate()
                process.wait(timeout=5)  # 5초 대기
        except Exception as e:
            print(f"⚠️ 프로세스 정리 중 오류: {e}")
            try:
                process.kill()  # 강제 종료
            except:
                pass
    active_processes.clear()

def signal_handler(signum, frame):
    """시그널 핸들러"""
    print(f"🛑 시그널 {signum} 수신 - 정리 중...")
    cleanup_processes()
    sys.exit(0)

# 시그널 핸들러 등록 (Windows에서는 SIGTERM, SIGINT 지원)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGINT'):
    signal.signal(signal.SIGINT, signal_handler)

# 프로그램 종료 시 자동으로 정리
atexit.register(cleanup_processes)

app = Flask(__name__)

# 정적 파일 설정
app.static_folder = 'static'
app.template_folder = 'templates'

@app.route('/')
def index():
    """메인 페이지"""
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
        
        # Windows에서 새로운 프로세스 그룹으로 실행하여 부모와 함께 종료되도록 설정
        creation_flags = 0
        if os.name == 'nt':  # Windows
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        try:
            # Popen으로 프로세스 추적 가능하도록 변경
            process = subprocess.Popen(
                cmd_args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env,
                creationflags=creation_flags
            )
            
            # 활성 프로세스 리스트에 추가
            global active_processes
            active_processes.append(process)
            
            # 프로세스 완료까지 대기
            stdout, stderr = process.communicate()
            
            # 완료된 프로세스는 리스트에서 제거
            if process in active_processes:
                active_processes.remove(process)
                
            print(f"🔥 [DEBUG] subprocess 종료 코드: {process.returncode}")
            if stderr:
                print(f"🔥 [DEBUG] subprocess stderr: {stderr}")
            if stdout:
                print(f"🔥 [DEBUG] subprocess stdout: {stdout[:200]}...")
            
            if process.returncode != 0:
                print(f"🔥 [DEBUG] subprocess 실패 - 500 에러 반환")
                print(f"[오류] 검색 실패: {stderr}")
                return jsonify({'error': '검색 중 오류가 발생했습니다.'}), 500
                
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

@app.route('/explain/<movie_title>')
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
    app.run(debug=True, host='0.0.0.0', port=5000)
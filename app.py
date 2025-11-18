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
from pathlib import Path

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
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': '검색어를 입력해주세요.'}), 400
        
        # build_faiss_and_query.py 실행
        print(f"[검색] '{query}' 검색 시작...")
        
        result = subprocess.run([
            sys.executable, 'build_faiss_and_query.py',
            '--demo_query', query,
            '--llm_filter',
            '--save_web',
            '--top_k', '10'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"[오류] 검색 실패: {result.stderr}")
            return jsonify({'error': '검색 중 오류가 발생했습니다.'}), 500
        
        # 결과 로드
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'error': '검색 결과가 생성되지 않았습니다.'}), 500
        
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"[성공] {len(results['movies'])}개 영화 검색 완료")
        return jsonify(results)
        
    except Exception as e:
        print(f"[예외] 검색 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/explain/<int:movie_rank>')
def explain_movie(movie_rank):
    """영화 추천 이유 설명"""
    try:
        # 기존 검색 결과에서 해당 영화의 LLM 설명 반환
        web_results_path = Path('web_results.json')
        if not web_results_path.exists():
            return jsonify({'error': '검색 결과를 찾을 수 없습니다.'}), 404
        
        with open(web_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        movies = results.get('movies', [])
        if movie_rank < 1 or movie_rank > len(movies):
            return jsonify({'error': '유효하지 않은 영화 순위입니다.'}), 400
        
        movie = movies[movie_rank - 1]  # 0-based index
        llm_analysis = movie.get('llm_analysis')
        
        if not llm_analysis or not llm_analysis.get('reason'):
            return jsonify({'error': 'LLM 설명을 찾을 수 없습니다.'}), 404
        
        return jsonify({
            'title': movie.get('title'),
            'year': movie.get('year'),
            'explanation': llm_analysis.get('reason'),
            'score': llm_analysis.get('score')
        })
        
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
#!/usr/bin/env python3

import json

def find_common_movies():
    # Get unique movies from plot metadata (chunked)
    plot_movies = set()
    with open('data/metadata.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            plot_movies.add((data['title'], data['year']))
    
    # Get movies from movie metadata (single record per movie)
    movie_movies = set()
    with open('data/separated_embeddings/movie_metadata.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            movie_movies.add((data['title'], data['year']))
    
    # Find common movies
    common_movies = plot_movies & movie_movies
    
    print(f"📊 데이터셋 분석:")
    print(f"  Plot 데이터셋 (청킹된): {len(plot_movies)}개 고유 영화")
    print(f"  Movie 데이터셋 (흐름/장르): {len(movie_movies)}개 영화")
    print(f"  공통 영화: {len(common_movies)}개")
    
    if len(common_movies) > 0:
        print(f"\n🎬 공통 영화 목록 (처음 20개):")
        for i, (title, year) in enumerate(sorted(common_movies)[:20]):
            print(f"  {i+1:2d}. {title} ({year})")
        
        if len(common_movies) > 20:
            print(f"  ... 그리고 {len(common_movies) - 20}개 더")
    else:
        print(f"\n❌ 공통 영화가 없습니다.")
        print(f"\n📝 Plot 데이터셋 영화 예시:")
        for i, (title, year) in enumerate(sorted(plot_movies)[:10]):
            print(f"  {i+1:2d}. {title} ({year})")
        
        print(f"\n📝 Movie 데이터셋 영화 예시:")
        for i, (title, year) in enumerate(sorted(movie_movies)[:10]):
            print(f"  {i+1:2d}. {title} ({year})")
    
    return common_movies

if __name__ == "__main__":
    find_common_movies()
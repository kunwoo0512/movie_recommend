#!/usr/bin/env python3
import json

# 영화 데이터 로드
with open('movies_dataset.json', 'r', encoding='utf-8') as f:
    movies = json.load(f)

# "자신을 배신한 조직에게 복수하는" 검색 결과 상위 10개 영화
search_results = [
    "Fantastic Beasts and Where to Find Them",
    "Guardians of the Galaxy Vol. 2", 
    "Popeye's Revenge",
    "Kantara",
    "KPop Demon Hunters",
    "Ghost Rider",
    "The Lord of the Rings: The War of the Rohirrim",
    "Omniscient Reader: The Prophecy",
    "G.I. Joe: Retaliation",
    "Heads of State"
]

print("'자신을 배신한 조직에게 복수하는' 검색 결과 분석")
print("=" * 80)

# 각 영화의 줄거리에서 복수/배신 관련 키워드 찾기
betrayal_keywords = [
    'betray', 'betrayal', 'revenge', 'vengeance', 'retaliation', 
    'backstab', 'double-cross', 'turncoat', 'traitor', 'deceive',
    'organization', 'gang', 'group', 'team', 'company', 'agency',
    '배신', '복수', '배반', '조직', '단체', '회사'
]

for i, movie_title in enumerate(search_results, 1):
    print(f"\n{i}. {movie_title}")
    print("-" * 40)
    
    # 영화 찾기
    found_movie = None
    for movie in movies:
        if movie_title.lower() in movie.get('title', '').lower():
            found_movie = movie
            break
    
    if found_movie:
        title = found_movie.get('title')
        year = found_movie.get('year')
        plot = found_movie.get('plot', '')
        
        print(f"제목: {title} ({year})")
        print(f"줄거리: {plot[:300]}...")
        
        # 키워드 분석
        plot_lower = plot.lower()
        found_keywords = [kw for kw in betrayal_keywords if kw in plot_lower]
        
        if found_keywords:
            print(f"\n🎯 복수/배신 관련 키워드: {found_keywords}")
        else:
            print(f"\n❌ 복수/배신 키워드 없음")
            
        # 관련 문장 추출
        sentences = plot.split('.')
        relevant_sentences = []
        for sent in sentences:
            if any(kw in sent.lower() for kw in betrayal_keywords):
                relevant_sentences.append(sent.strip())
        
        if relevant_sentences:
            print(f"📝 관련 문장들:")
            for sent in relevant_sentences[:2]:
                print(f"   - {sent[:100]}...")
        
        # 간단한 관련성 점수 (키워드 개수 기반)
        relevance_score = len(found_keywords)
        if relevance_score >= 3:
            print(f"✅ 관련성: 높음 ({relevance_score}개 키워드)")
        elif relevance_score >= 1:
            print(f"⚠️ 관련성: 보통 ({relevance_score}개 키워드)")
        else:
            print(f"❌ 관련성: 낮음 (키워드 없음)")
            
    else:
        print("❓ 영화를 찾을 수 없음")

print("\n" + "=" * 80)
print("분석 요약:")
print("- ✅ 높은 관련성: 3개 이상 키워드")
print("- ⚠️ 보통 관련성: 1-2개 키워드") 
print("- ❌ 낮은 관련성: 키워드 없음")
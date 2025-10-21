"""
    GPT    movies_dataset.json 
"""
import json
import os
from dotenv import load_dotenv
from real_gpt_analyzer import GPTMovieAnalyzer

# .env  
load_dotenv()

def main():
    #  
    with open('movies_dataset.json', 'r', encoding='utf-8') as f:
        movies = json.load(f)

    analyzer = GPTMovieAnalyzer()
    if not analyzer.client:
        print("[] GPT API  . .env  API  .")
        return

    results = []
    for idx, movie in enumerate(movies, 1):
        print(f"\n[{idx}/{len(movies)}] [] '{movie['title']}'  ...")
        try:
            analysis = analyzer.analyze_movie(
                title=movie['title'],
                plot=movie['plot'],
                year=movie.get('year')
            )
            #   
            movie.update({
                'flow_curve': analysis.get('flow_curve'),
                'genres': analysis.get('genres'),
                'detailed_analysis': analysis.get('detailed_analysis'),
                'gpt_success': analysis.get('success', False)
            })
            print(f"[] '{movie['title']}'  !")
        except Exception as e:
            print(f"[] '{movie['title']}'  : {str(e)}")
            movie['gpt_success'] = False
        results.append(movie)

    #  
    with open('movies_dataset_gpt.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n      movies_dataset_gpt.json !")

if __name__ == "__main__":
    main()

import json
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

with open("tmdb_movieList_dedup.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

results = []
for i, movie in enumerate(movies):
    title = movie["title"]
    year = movie["year"]
    # TMDb  API   
    query_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}&language=en-US"
    resp = requests.get(query_url)
    if resp.status_code != 200:
        print(f"Error: {title} ({year})  ")
        continue
    data = resp.json()
    if not data["results"]:
        print(f"No result: {title} ({year})")
        continue
    info = data["results"][0]
    overview = info.get("overview", "")
    poster_path = info.get("poster_path", "")
    if overview and poster_path:
        movie["overview"] = overview
        movie["poster_path"] = poster_path
        results.append(movie)
    if i % 50 == 0:
        print(f" : {i+1}/{len(movies)}")
    time.sleep(0.2)  # API rate limit 

print(f"   : {len(results)}")
with open("movies_dataset_final.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("   : movies_dataset_final.json")

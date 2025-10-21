import json
import requests
import os
from dotenv import load_dotenv
import time

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

with open("tmdb_movieList_dedup.json", "r", encoding="utf-8") as f:
    movies = json.load(f)

final_dataset = []
for idx, movie in enumerate(movies):
    title = movie["title"]
    year = movie["year"]
    # TMDb  API   
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}&year={year}&language=en-US"
    resp = requests.get(search_url)
    if resp.status_code != 200:
        continue
    results = resp.json().get("results", [])
    if not results:
        continue
    info = results[0]
    overview = info.get("overview", "")
    poster_path = info.get("poster_path", "")
    if not overview or not poster_path:
        continue
    #  URL 
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
    final_dataset.append({
        "title": title,
        "year": year,
        "genre": movie["genre"],
        "overview": overview,
        "poster_url": poster_url
    })
    # TMDb API rate limit 
    time.sleep(0.2)
    if (idx+1) % 100 == 0:
        print(f"{idx+1}  ...")

with open("movies_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=2)
print(f"   : movies_dataset.json ( {len(final_dataset)})")

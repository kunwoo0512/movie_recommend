import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# TMDb   (id: name)
GENRES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    18: "Drama",
    14: "Fantasy",
    27: "Horror",
    10749: "Romance",
    878: "Sci-Fi",
    53: "Thriller"
}

TOP_N = 100  #    (  10 x 100 = 1000)
all_movies = {}

for genre_id, genre_name in GENRES.items():
    print(f"Fetching {genre_name} (id={genre_id}) top {TOP_N}")
    page = 1
    count = 0
    while count < TOP_N:
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=popularity.desc&language=en-US&page={page}"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Error: {genre_name} page {page}")
            break
        data = resp.json()
        for movie in data.get("results", []):
            key = (movie.get("title"), movie.get("release_date", "")[:4])
            if key not in all_movies:
                all_movies[key] = {
                    "title": movie.get("title"),
                    "year": movie.get("release_date", "")[:4],
                    "genre": [genre_name],
                    "genre_ids": [genre_id]
                }
                count += 1
            if count >= TOP_N:
                break
        page += 1

#     
movie_list = list(all_movies.values())
print(f" {len(movie_list)}   .")

with open("tmdb_movieList.json", "w", encoding="utf-8") as f:
    json.dump(movie_list, f, ensure_ascii=False, indent=2)
print("     : tmdb_movieList.json")

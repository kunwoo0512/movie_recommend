import requests
import json
import os
import re
from dotenv import load_dotenv
import wikipedia
from wiki_extractor import WikiplotExtractor

# 환경 변수에서 OMDb API 키 로드
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
POSTER_DIR = "posters"
os.makedirs(POSTER_DIR, exist_ok=True)

# tmdb_movieList.json의 전체 영화 대상으로 포스터와 줄거리가 모두 있는 영화만 데이터셋에 포함
def get_omdb_metadata(title, year):
    url = f"http://www.omdbapi.com/?t={title.replace(' ', '+')}&y={year}&apikey={OMDB_API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("Response") != "True":
        return None
    return {
        "director": data.get("Director"),
        "poster_url": data.get("Poster")
    }

def download_image(url, filename):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(filename, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False

extractor = WikiplotExtractor()
dataset = []
with open("tmdb_movieList.json", "r", encoding="utf-8") as f:
    movies = json.load(f)
for idx, movie in enumerate(movies):
    title = movie["title"]
    year = movie["year"]
    omdb = get_omdb_metadata(title, year)
    director = omdb["director"] if omdb else ""
    poster_path = ""
    has_poster = False
    if omdb and omdb["poster_url"] and omdb["poster_url"] != "N/A":
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
        poster_filename = os.path.join(POSTER_DIR, f"{safe_title}.jpg")
        if download_image(omdb["poster_url"], poster_filename):
            poster_path = poster_filename
            has_poster = True
    plot = extractor.extract_plot(title, int(year) if year else None)
    has_plot = bool(plot and plot.strip())
    if has_poster and has_plot:
        dataset.append({
            "title": title,
            "year": year,
            "director": director,
            "plot": plot,
            "poster": poster_path
        })
    if (idx+1) % 50 == 0:
        print(f"{idx+1}개 처리 완료...")

with open("movies_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f"포스터와 줄거리가 모두 있는 영화 {len(dataset)}개 데이터셋 생성 완료: movies_dataset.json")

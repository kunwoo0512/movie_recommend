"""
FAISS   DB   
"""
import json
import numpy as np
import faiss

EMBEDDING_FILE = 'movie_embeddings.json'
DB_FILE = 'faiss_movie_index.bin'
META_FILE = 'faiss_movie_meta.json'

#    
def load_embeddings():
    with open(EMBEDDING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    embeddings = np.array([item['embedding'] for item in data]).astype('float32')
    titles = [item['title'] for item in data]
    return embeddings, titles

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, db_file):
    faiss.write_index(index, db_file)

def save_meta(titles, meta_file):
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)

def main():
    embeddings, titles = load_embeddings()
    index = build_faiss_index(embeddings)
    save_index(index, DB_FILE)
    save_meta(titles, META_FILE)
    print(f"[] FAISS DB  : {DB_FILE}, : {META_FILE}")

if __name__ == "__main__":
    main()

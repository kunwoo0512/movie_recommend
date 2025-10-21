"""
 DB + LLM    
"""
import json
import numpy as np
import faiss
import openai
import os
from dotenv import load_dotenv
import time

# .env  
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class HybridMovieRecommender:
    def __init__(self):
        """    """
        print("[]     ")
        print("=" * 50)
        
        #  DB  (   )
        self.index = faiss.read_index('faiss_movie_index_plot_only.bin')
        
        with open('faiss_movie_meta_plot_only.json', 'r', encoding='utf-8') as f:
            self.titles = json.load(f)
        
        with open('movies_dataset.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        
        #   ->    ()
        self.movie_details = {movie['title']: movie for movie in movies_data}
        
        print(f"[]  DB  : {self.index.ntotal} ")
        print(f"[]    : {len(self.movie_details)}")
    
    def get_query_embedding(self, query):
        """   """
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            embedding = np.array(response.data[0].embedding).astype('float32')
            # L2 
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"[]    : {e}")
            return None
    
    def vector_search(self, query, top_k=20):
        """1:  DB   """
        print(f"[1]  DB  ( {top_k} )")
        
        query_embedding = self.get_query_embedding(query)
        if query_embedding is None:
            return []
        
        # FAISS 
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k=top_k)
        
        candidates = []
        for distance, idx in zip(distances[0], indices[0]):
            # titles      
            movie_info = self.titles[idx]
            movie_title = movie_info.get('title', 'Unknown')
            
            candidates.append({
                'title': movie_title,
                'plot': movie_info.get('plot', ''),
                'year': movie_info.get('year', 'N/A'),
                'director': movie_info.get('director', 'N/A'),
                'vector_similarity': 1 / (1 + distance),
                'genres': movie_info.get('genres', {})
            })
        
        print(f"   [] {len(candidates)}    ")
        return candidates
    
    def gpt_verify(self, query, candidates, max_results=5):
        """2: GPT   """
        print(f"[2] GPT  ( {max_results} )")
        
        verified_movies = []
        verification_costs = 0
        
        for i, candidate in enumerate(candidates):
            print(f"   [{i+1}/{len(candidates)}] '{candidate['title']}'  ...")
            
            # GPT   ( )
            prompt = f""" "{query}"    .

 :
: {candidate['title']} ({candidate['year']})
: {candidate['director']}
: {candidate['plot']}

  :
-      ""
-       ""  
-       ""
-   ""  ()

 :
: /
: 1-10 (1=, 10=)
: (  )"""

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",  #    
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,  #     
                    temperature=0.1
                )
                
                gpt_response = response.choices[0].message.content
                verification_costs += 0.0006  # gpt-4o-mini  ( )
                
                #  
                if ": " in gpt_response or ":" in gpt_response:
                    #  
                    confidence = 7  # 
                    try:
                        if ":" in gpt_response:
                            confidence_text = gpt_response.split(":")[1].split("")[0].strip()
                            confidence = int(confidence_text)
                    except:
                        pass
                    
                    #  
                    reason = "GPT  "
                    try:
                        if ":" in gpt_response:
                            reason = gpt_response.split(":")[1].strip()
                    except:
                        pass
                    
                    verified_movies.append({
                        **candidate,
                        'gpt_confidence': confidence,
                        'gpt_reason': reason,
                        'final_score': candidate['vector_similarity'] * 0.7 + confidence/10 * 0.3
                    })
                    
                    print(f"      []   (: {confidence}/10)")
                else:
                    print(f"      []  ")
                
                # API  
                time.sleep(0.1)
                
                #     
                if len(verified_movies) >= max_results:
                    print(f"   []   ,  ")
                    break
                    
            except Exception as e:
                print(f"      [] GPT  : {e}")
                continue
        
        #   
        verified_movies.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"   []  {len(verified_movies)}   ")
        print(f"     : ${verification_costs:.4f}")
        
        return verified_movies[:max_results]
    
    def search(self, query, top_k_candidates=20, max_results=5):
        """  """
        print(f"\n[]  : '{query}'")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1:  
        candidates = self.vector_search(query, top_k_candidates)
        if not candidates:
            print("[]   ")
            return []
        
        # 2: GPT 
        verified_results = self.gpt_verify(query, candidates, max_results)
        
        #  
        search_time = time.time() - start_time
        print(f"\n⏱   : {search_time:.2f}")
        
        return verified_results
    
    def display_results(self, results):
        """  """
        if not results:
            print("   .")
            return
        
        print(f"\n[]   :")
        print("=" * 50)
        
        for i, movie in enumerate(results, 1):
            print(f"\n{i}. {movie['title']} ({movie['year']})")
            print(f"   : {movie['director']}")
            print(f"    : {movie['final_score']:.3f}")
            print(f"    : {movie['vector_similarity']:.3f}")
            print(f"   GPT : {movie['gpt_confidence']}/10")
            print(f"   GPT : {movie['gpt_reason']}")
            
            #  
            genres = movie.get('genres', {})
            if genres:
                top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
                genre_str = ', '.join([f"{g}:{s}" for g, s in top_genres])
                print(f"    : {genre_str}")

def main():
    """ """
    #    
    recommender = HybridMovieRecommender()
    
    #  
    test_queries = [
        "   "
    ]
    
    for query in test_queries:
        #   (      )
        results = recommender.search(query, top_k_candidates=20, max_results=5)
        
        #  
        recommender.display_results(results)
        
        print("\n" + "="*80)
        
        #    
        print("   ...")
    
    print("[]      !")

if __name__ == "__main__":
    main()

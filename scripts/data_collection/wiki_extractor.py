"""
    
"""

import wikipedia
import re
import json
from typing import Dict, Optional

class WikiplotExtractor:
    def __init__(self):
        #   
        wikipedia.set_lang("en")  #   (  )
    
    def extract_plot(self, movie_title: str, year: int) -> Optional[str]:
        """
             
        
        Args:
            movie_title:  
            year:  
            
        Returns:
               None
        """
        try:
            #    
            search_patterns = [
                f"{movie_title} ({year} film)",
                f"{movie_title} (film)",
                f"{movie_title} {year}",
                movie_title
            ]
            
            for pattern in search_patterns:
                try:
                    print(f" : {pattern}")
                    page = wikipedia.page(pattern, auto_suggest=False)
                    content = page.content
                    
                    # Plot  
                    plot = self._extract_plot_section(content)
                    if plot:
                        print(f" {movie_title}   !")
                        return plot
                        
                except wikipedia.exceptions.DisambiguationError as e:
                    #       
                    try:
                        page = wikipedia.page(e.options[0])
                        content = page.content
                        plot = self._extract_plot_section(content)
                        if plot:
                            print(f" {movie_title}   ! (disambiguation)")
                            return plot
                    except:
                        continue
                        
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    print(f": {e}")
                    continue
            
            print(f" {movie_title}    .")
            return None
            
        except Exception as e:
            print(f" {movie_title}   : {e}")
            return None
    
    def _extract_plot_section(self, content: str) -> Optional[str]:
        """
          Plot  
        """
        # Plot, Synopsis, Story   
        plot_patterns = [
            r"==\s*Plot\s*==\s*\n(.*?)(?=\n==|\n\n==|\Z)",
            r"==\s*Synopsis\s*==\s*\n(.*?)(?=\n==|\n\n==|\Z)", 
            r"==\s*Story\s*==\s*\n(.*?)(?=\n==|\n\n==|\Z)",
            r"==\s*Storyline\s*==\s*\n(.*?)(?=\n==|\n\n==|\Z)"
        ]
        
        for pattern in plot_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                plot = match.group(1).strip()
                #  
                plot = self._clean_plot_text(plot)
                if len(plot) > 100:  #   
                    return plot
        
        return None
    
    def _clean_plot_text(self, text: str) -> str:
        """
          
        """
        #    
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[]] → 
        text = re.sub(r'\[([^\]]+)\]', '', text)  # [] 
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # {{}} 
        
        #   
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

def main():
    """
      
    """
    extractor = WikiplotExtractor()
    
    #  
    test_movies = [
        {"title": "The Matrix", "year": 1999},
        {"title": "Toy Story", "year": 1995},
        {"title": "Inception", "year": 2010}
    ]
    
    results = {}
    
    for movie in test_movies:
        print(f"\n{'='*50}")
        print(f"[] {movie['title']} ({movie['year']})  ...")
        print(f"{'='*50}")
        
        plot = extractor.extract_plot(movie["title"], movie["year"])
        
        results[movie["title"]] = {
            "year": movie["year"],
            "plot": plot,
            "success": plot is not None
        }
        
        if plot:
            print(f"\n  :")
            print(f"{plot[:200]}...")
        else:
            print("   ")
    
    #  
    with open("movie_plots.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print("  movie_plots.json !")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

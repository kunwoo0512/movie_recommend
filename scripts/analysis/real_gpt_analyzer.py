"""
GPT API     
"""

import json
import os
from typing import Dict, List
import openai
from openai import OpenAI

class GPTMovieAnalyzer:
    def __init__(self, api_key: str = None):
        """
        GPT API 
        
        Args:
            api_key: OpenAI API  (  )
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            print(" OpenAI API   !")
            print(" 1:   - OPENAI_API_KEY=your_api_key")
            print(" 2:    - GPTMovieAnalyzer(api_key='your_key')")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(" GPT API  !")
    
    def analyze_movie(self, title: str, plot: str, year: int = None) -> Dict:
        """
        GPT   
        
        Args:
            title:  
            plot:  
            year:  
            
        Returns:
              
        """
        if not self.client:
            print(" GPT API  . API  .")
            return self._fallback_analysis(title)
        
        try:
            print(f" GPT '{title}'  ...")
            
            # GPT  
            prompt = self._create_analysis_prompt(title, plot, year)
            
            # GPT API 
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  #  "gpt-4"
                messages=[
                    {
                        "role": "system", 
                        "content": "   .         ."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  #     
                max_tokens=1500
            )
            
            #  
            gpt_response = response.choices[0].message.content
            analysis_result = self._parse_gpt_response(gpt_response, title)
            
            print(f" '{title}' GPT  !")
            return analysis_result
            
        except Exception as e:
            print(f" GPT    : {str(e)}")
            return self._fallback_analysis(title)
    
    def _create_analysis_prompt(self, title: str, plot: str, year: int = None) -> str:
        """
        GPT    
        """
        year_info = f"({year} )" if year else ""
        
        prompt = f"""
 : {title} {year_info}

:
{plot}

    :

1.   :  10     / 1-10 
   - 1:    ( ,  )
   - 5:   ( ,  )
   - 10:   (, ,  )

2.  :    1-10 
   - action ()
   - thriller ()
   - romance ()
   - drama ()
   - comedy ()
   - sci_fi ()
   - horror ()

3.    

  JSON   :

{{
    "flow_curve": [1, 2, ..., 10],
    "genres": {{
        "action": ,
        "thriller": ,
        "romance": ,
        "drama": ,
        "comedy": ,
        "sci_fi": ,
        "horror": 
    }},
    "detailed_analysis": [
        " 1: ...",
        " 2: ...",
        ...
        " 10: ..."
    ]
}}
"""
        return prompt
    
    def _parse_gpt_response(self, response: str, title: str) -> Dict:
        """
        GPT     
        """
        try:
            # JSON  
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON    ")
            
            json_str = response[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            
            #  
            if "flow_curve" not in parsed_data or "genres" not in parsed_data:
                raise ValueError("  ")
            
            if len(parsed_data["flow_curve"]) != 10:
                raise ValueError("flow_curve 10  ")
            
            #  
            parsed_data["analysis_method"] = "gpt_api"
            parsed_data["model_used"] = "gpt-3.5-turbo"
            parsed_data["success"] = True
            
            return parsed_data
            
        except Exception as e:
            print(f" GPT   : {str(e)}")
            print(f" : {response[:200]}...")
            return self._fallback_analysis(title)
    
    def _fallback_analysis(self, title: str) -> Dict:
        """
        API     ( )
        """
        print(f" '{title}'   ")
        
        #   
        if "Matrix" in title:
            return {
                "flow_curve": [2, 3, 5, 7, 8, 9, 6, 8, 9, 7],
                "genres": {
                    "action": 9, "thriller": 8, "romance": 2,
                    "drama": 6, "comedy": 1, "sci_fi": 10, "horror": 3
                },
                "detailed_analysis": ["  "] * 10,
                "analysis_method": "fallback",
                "success": False
            }
        elif "Toy Story" in title:
            return {
                "flow_curve": [4, 3, 5, 6, 7, 8, 6, 9, 8, 7],
                "genres": {
                    "action": 4, "thriller": 2, "romance": 1,
                    "drama": 6, "comedy": 9, "sci_fi": 1, "horror": 0
                },
                "detailed_analysis": ["  "] * 10,
                "analysis_method": "fallback",
                "success": False
            }
        else:
            return {
                "flow_curve": [3, 4, 5, 6, 7, 8, 6, 7, 6, 5],
                "genres": {
                    "action": 5, "thriller": 5, "romance": 3,
                    "drama": 7, "comedy": 4, "sci_fi": 2, "horror": 2
                },
                "detailed_analysis": ["  "] * 10,
                "analysis_method": "fallback",
                "success": False
            }

def main():
    """
     
    """
    # 1. API   
    print(" GPT API  :")
    print("1. : export OPENAI_API_KEY='your_api_key'")
    print("2.  : analyzer = GPTMovieAnalyzer(api_key='your_key')")
    print()
    
    # 2.  
    analyzer = GPTMovieAnalyzer()
    
    # 3.   
    try:
        with open('movie_plots.json', 'r', encoding='utf-8') as f:
            movies = json.load(f)
        
        # 4.     
        first_movie = list(movies.keys())[0]
        movie_data = movies[first_movie]
        
        result = analyzer.analyze_movie(
            title=first_movie,
            plot=movie_data['plot'],
            year=movie_data.get('year')
        )
        
        print(f"\n '{first_movie}'  :")
        print(f" : {result['flow_curve']}")
        print(f" : {dict(list(result['genres'].items())[:3])}")
        print(f" : {result['analysis_method']}")
        
    except FileNotFoundError:
        print(" movie_plots.json    .")
    except Exception as e:
        print(f"  : {str(e)}")

if __name__ == "__main__":
    main()

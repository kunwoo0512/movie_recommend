"""
      
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import os

class FlowVisualizer:
    def __init__(self):
        #    (  )
        plt.rcParams['axes.unicode_minus'] = False
        # Windows    
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'  #  
        except:
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans'  # 
            except:
                pass
        
        #  
        self.colors = {
            'flow': '#2E86AB',
            'action': '#A23B72',
            'thriller': '#F18F01', 
            'romance': '#C73E1D',
            'drama': '#8E44AD',
            'comedy': '#27AE60',
            'sci_fi': '#3498DB'
        }
    
    def create_flow_visualization(self, movie_data: Dict, save_path: str = None):
        """
              
        """
        title = list(movie_data.keys())[0]
        data = movie_data[title]
        
        analysis = data['analysis']
        flow_curve = analysis['flow_curve']
        genres = analysis['genres']
        
        #   (2x1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{title} ({data["year"]}) - Movie Flow Analysis', 
                    fontsize=16, fontweight='bold')
        
        # 1.   
        x = range(1, len(flow_curve) + 1)
        ax1.plot(x, flow_curve, marker='o', linewidth=3, markersize=8, 
                color=self.colors['flow'], markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=self.colors['flow'])
        
        ax1.fill_between(x, flow_curve, alpha=0.3, color=self.colors['flow'])
        ax1.set_title('Emotion/Tension Flow Curve', fontsize=14, pad=20)
        ax1.set_xlabel('Story Progression', fontsize=12)
        ax1.set_ylabel('Intensity (1-10)', fontsize=12)
        ax1.set_ylim(0, 10)
        ax1.grid(True, alpha=0.3)
        
        #   
        sections = ['Intro', 'Setup', 'Dev1', 'Dev2', 'Crisis1', 'Climax', 'Crisis2', 'Resolve1', 'Peak', 'End']
        for i, (pos, intensity) in enumerate(zip(x, flow_curve)):
            if i < len(sections):
                ax1.annotate(sections[i], (pos, intensity), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9, alpha=0.7)
        
        # 2.    
        genre_names = list(genres.keys())
        genre_values = list(genres.values())
        genre_colors = [self.colors.get(genre, '#95A5A6') for genre in genre_names]
        
        bars = ax2.bar(genre_names, genre_values, color=genre_colors, alpha=0.8)
        ax2.set_title('Genre Weight Analysis', fontsize=14, pad=20)
        ax2.set_ylabel('Weight (1-10)', fontsize=12)
        ax2.set_ylim(0, 10)
        
        #    
        for bar, value in zip(bars, genre_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        #   ()
        if 'description' in analysis:
            fig.text(0.5, 0.02, f"Flow Summary: {analysis['description']}", 
                    ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph for {title} saved to {save_path}!")
        
        plt.show()
        return fig
    
    def create_comparison_chart(self, movies_data: Dict, save_path: str = None):
        """
             
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
        
        for i, (title, data) in enumerate(movies_data.items()):
            flow = data['analysis']['flow_curve']
            x = range(1, len(flow) + 1)
            
            color = colors[i % len(colors)]
            ax.plot(x, flow, marker='o', linewidth=2.5, label=f"{title} ({data['year']})",
                   color=color, markersize=6)
        
        ax.set_title('Movie Flow Curves Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Story Progression', fontsize=12)
        ax.set_ylabel('Emotion/Tension Intensity (1-10)', fontsize=12)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to {save_path}!")
        
        plt.show()
        return fig

def visualize_all_movies(json_file="movie_analysis_results.json"):
    """
       
    """
    #   
    with open(json_file, "r", encoding="utf-8") as f:
        movies_data = json.load(f)
    
    visualizer = FlowVisualizer()
    
    #    
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[]   ...")
    print("="*50)
    
    # 1.   
    for title, data in movies_data.items():
        print(f"\n[] {title}  ...")
        
        #   
        single_movie = {title: data}
        save_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_analysis.png")
        visualizer.create_flow_visualization(single_movie, save_path)
    
    # 2.   
    print(f"\n[]     ...")
    comparison_path = os.path.join(output_dir, "movies_comparison.png")
    visualizer.create_comparison_chart(movies_data, comparison_path)
    
    print(f"\n[]   !")
    print(f"[]  {output_dir}  .")

def main():
    """
      
    """
    print("[]   ")
    print("="*50)
    
    try:
        visualize_all_movies()
    except FileNotFoundError:
        print("[] movie_analysis_results.json    .")
        print(" gpt_analyzer.py .")

if __name__ == "__main__":
    main()

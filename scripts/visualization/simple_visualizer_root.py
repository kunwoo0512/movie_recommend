"""
    
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def setup_plot_style():
    """  """
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'DejaVu Sans'

def visualize_flow_curve(title, flow_curve, analysis_type=""):
    """  """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(1, len(flow_curve) + 1)
    
    #  
    ax.plot(x, flow_curve, marker='o', linewidth=3, markersize=8, 
            color='#2E86AB', label=f'{title}')
    
    #  
    ax.fill_between(x, flow_curve, alpha=0.3, color='#2E86AB')
    
    #   
    plot_title = f'{title} - Narrative Flow'
    if analysis_type:
        plot_title += f' ({analysis_type})'
    
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Story Progression (1-10 segments)', fontsize=12)
    ax.set_ylabel('Emotional/Tension Intensity', fontsize=12)
    
    #   
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, len(flow_curve) + 0.5)
    ax.set_ylim(0, 11)
    
    # Y 
    ax.set_yticks(range(0, 11, 2))
    
    # 
    ax.legend()
    
    # 
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{title.replace(' ', '_')}_flow_{analysis_type.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[] {title}   : {filepath}")

def visualize_genres(title, genres, analysis_type=""):
    """  """
    setup_plot_style()
    
    #  5  
    sorted_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5]
    
    genre_names = [item[0].replace('_', ' ').title() for item in sorted_genres]
    genre_scores = [item[1] for item in sorted_genres]
    
    #  
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(genre_names, genre_scores, color=colors[:len(genre_names)], alpha=0.8)
    
    #  
    plot_title = f'{title} - Genre Composition'
    if analysis_type:
        plot_title += f' ({analysis_type})'
    
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Genre Strength (1-10)', fontsize=12)
    
    #    
    for bar, score in zip(bars, genre_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    #  
    ax.set_ylim(0, 11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # x  
    plt.xticks(rotation=45, ha='right')
    
    # 
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{title.replace(' ', '_')}_genres_{analysis_type.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[] {title}   : {filepath}")

def compare_flow_curves(movies_data, analysis_type=""):
    """    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60']
    
    for i, (title, data) in enumerate(movies_data.items()):
        flow_curve = data.get('flow_curve', [])
        if flow_curve:
            x = range(1, len(flow_curve) + 1)
            color = colors[i % len(colors)]
            
            ax.plot(x, flow_curve, marker='o', linewidth=2.5, 
                   label=title, color=color, markersize=6)
    
    #  
    plot_title = 'Movie Flow Curves Comparison'
    if analysis_type:
        plot_title += f' ({analysis_type})'
    
    ax.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Story Progression', fontsize=12)
    ax.set_ylabel('Emotional/Tension Intensity (1-10)', fontsize=12)
    
    #   
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 11)
    ax.legend()
    
    # 
    output_dir = 'visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"comparison_{analysis_type.lower().replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[]   : {filepath}")

def main():
    """ """
    #    
    result_files = []
    
    if os.path.exists('movie_analysis_results.json'):
        result_files.append(('movie_analysis_results.json', 'GPT Analysis'))
    
    if os.path.exists('ollama_analysis_results.json'):
        result_files.append(('ollama_analysis_results.json', 'Ollama Analysis'))
    
    if not result_files:
        print("[]      .")
        return
    
    for filename, analysis_type in result_files:
        print(f"\n[] {analysis_type}  ...")
        
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        #    
        for title, movie_data in results.items():
            print(f"[] {title}   ...")
            
            flow_curve = movie_data.get('flow_curve', [])
            genres = movie_data.get('genres', {})
            
            if flow_curve:
                visualize_flow_curve(title, flow_curve, analysis_type)
            
            if genres:
                visualize_genres(title, genres, analysis_type)
        
        #   (2    )
        if len(results) >= 2:
            print(f"[] {analysis_type} -     ...")
            compare_flow_curves(results, analysis_type)
        
        print(f"[] {analysis_type}  !")

if __name__ == "__main__":
    main()

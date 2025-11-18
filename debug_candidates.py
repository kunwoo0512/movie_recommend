#!/usr/bin/env python3

from weighted_similarity_calculator import get_weighted_calculator

def debug_candidates():
    calc = get_weighted_calculator()
    calc.load_all_data()
    
    print("=== Inception Debug ===")
    
    # Get candidates from each modality
    plot_candidates = calc.search_plot_similarity('Inception', 5)
    flow_candidates = calc.search_flow_similarity('Inception', 5) 
    genre_candidates = calc.search_genre_similarity('Inception', 5)
    
    print(f"\nPlot candidates ({len(plot_candidates)}):")
    for i, movie in enumerate(plot_candidates):
        print(f"  {i+1}. {movie['title']} ({movie['year']}) - score: {movie['similarity']:.4f}")
    
    print(f"\nFlow candidates ({len(flow_candidates)}):")
    for i, movie in enumerate(flow_candidates):
        print(f"  {i+1}. {movie['title']} ({movie['year']}) - score: {movie['similarity']:.4f}")
    
    print(f"\nGenre candidates ({len(genre_candidates)}):")
    for i, movie in enumerate(genre_candidates):
        print(f"  {i+1}. {movie['title']} ({movie['year']}) - score: {movie['similarity']:.4f}")
    
    # Check for overlaps
    plot_keys = {(m['title'], m['year']) for m in plot_candidates}
    flow_keys = {(m['title'], m['year']) for m in flow_candidates}
    genre_keys = {(m['title'], m['year']) for m in genre_candidates}
    
    print(f"\nOverlaps:")
    print(f"  Plot ∩ Flow: {len(plot_keys & flow_keys)}")
    print(f"  Plot ∩ Genre: {len(plot_keys & genre_keys)}")
    print(f"  Flow ∩ Genre: {len(flow_keys & genre_keys)}")
    print(f"  All three: {len(plot_keys & flow_keys & genre_keys)}")

if __name__ == "__main__":
    debug_candidates()
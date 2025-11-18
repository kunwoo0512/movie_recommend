#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted Movie Similarity Calculator

This module provides weighted similarity calculation using:
- Chunked plot embeddings (data/index.faiss)
- Flow curve embeddings (data/separated_embeddings/flow_index.faiss)  
- Genre embeddings (data/separated_embeddings/genre_index.faiss)
"""

import os
import json
import numpy as np
import faiss
from collections import defaultdict

class WeightedSimilarityCalculator:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.plot_index = None
        self.flow_index = None 
        self.genre_index = None
        self.plot_metadata = None
        self.movie_metadata = None
        self.genre_info = None
        self.model = None
        
    def load_all_data(self):
        """Load all necessary data files"""
        try:
            # Load SBERT model
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Load plot index and metadata (chunked)
            plot_index_path = os.path.join(self.data_dir, "index.faiss")
            plot_metadata_path = os.path.join(self.data_dir, "metadata.jsonl")
            
            if not os.path.exists(plot_index_path):
                print(f"❌ Plot index not found: {plot_index_path}")
                return False
                
            self.plot_index = faiss.read_index(plot_index_path)
            
            # Load chunked metadata
            self.plot_metadata = []
            if os.path.exists(plot_metadata_path):
                with open(plot_metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.plot_metadata.append(json.loads(line.strip()))
            else:
                print(f"❌ Plot metadata not found: {plot_metadata_path}")
                return False
            
            # Load flow index
            flow_index_path = os.path.join(self.data_dir, "separated_embeddings", "flow_index.faiss")
            if not os.path.exists(flow_index_path):
                print(f"❌ Flow index not found: {flow_index_path}")
                return False
            self.flow_index = faiss.read_index(flow_index_path)
            
            # Load genre index
            genre_index_path = os.path.join(self.data_dir, "separated_embeddings", "genre_index.faiss")
            if not os.path.exists(genre_index_path):
                print(f"❌ Genre index not found: {genre_index_path}")
                return False
            self.genre_index = faiss.read_index(genre_index_path)
            
            # Load movie metadata (for flow and genre)
            movie_metadata_path = os.path.join(self.data_dir, "separated_embeddings", "movie_metadata.jsonl")
            self.movie_metadata = []
            if os.path.exists(movie_metadata_path):
                with open(movie_metadata_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.movie_metadata.append(json.loads(line.strip()))
            else:
                print(f"❌ Movie metadata not found: {movie_metadata_path}")
                return False
                
            # Load genre info
            genre_info_path = os.path.join(self.data_dir, "separated_embeddings", "genre_info.json")
            if os.path.exists(genre_info_path):
                with open(genre_info_path, 'r', encoding='utf-8') as f:
                    self.genre_info = json.load(f)
            
            print(f"✅ Loaded: Plot({len(self.plot_metadata)} chunks), "
                  f"Flow({len(self.movie_metadata)} movies), "
                  f"Genre({len(self.movie_metadata)} movies)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def find_movie_by_title(self, title: str):
        """Find movie in metadata by title"""
        title_lower = title.lower().strip()
        
        # Check movie metadata first
        for i, movie in enumerate(self.movie_metadata):
            movie_title = movie.get('title', '').lower().strip()
            if title_lower == movie_title:
                return movie, i
                
            # Try without year
            if '(' in movie_title and ')' in movie_title:
                title_without_year = movie_title.split('(')[0].strip()
                if title_lower == title_without_year:
                    return movie, i
        
        return None, -1
    
    def search_plot_similarity(self, movie_title: str, k: int = 10):
        """Search similar movies based on plot chunks"""
        # Find target movie in chunked metadata
        target_chunks = []
        target_chunk_idx = -1
        
        for i, chunk in enumerate(self.plot_metadata):
            if chunk['title'].lower().strip() == movie_title.lower().strip():
                target_chunks.append(chunk)
                target_chunk_idx = i
                break
        
        if not target_chunks:
            # Try without year
            for i, chunk in enumerate(self.plot_metadata):
                title_without_year = chunk['title'].split('(')[0].strip().lower()
                if title_without_year == movie_title.lower().strip():
                    target_chunks.append(chunk)
                    target_chunk_idx = i
                    break
        
        if not target_chunks or target_chunk_idx == -1:
            return []
        
        # Search similar chunks using the chunk index
        scores, indices = self.plot_index.search(
            np.array([self.plot_index.reconstruct(target_chunk_idx)]), 
            k * 5  # Get more to aggregate by movie
        )
        
        # Aggregate by movie
        movie_scores = defaultdict(list)
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.plot_metadata):
                chunk_meta = self.plot_metadata[idx]
                movie_key = (chunk_meta['title'], chunk_meta['year'])
                movie_scores[movie_key].append(float(score))
        
        # Calculate average score per movie
        movie_similarities = []
        for (title, year), score_list in movie_scores.items():
            if title.lower().strip() != movie_title.lower().strip():
                avg_score = np.mean(score_list)
                movie_similarities.append({
                    'title': title,
                    'year': year,
                    'similarity': avg_score
                })
        
        # Sort and return top k
        movie_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return movie_similarities[:k]
    
    def search_flow_similarity(self, movie_title: str, k: int = 10):
        """Search similar movies based on flow curves"""
        target_movie, target_idx = self.find_movie_by_title(movie_title)
        if target_movie is None:
            return []
        
        # Search similar movies
        scores, indices = self.flow_index.search(
            np.array([self.flow_index.reconstruct(target_idx)]), k + 1
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.movie_metadata):
                movie = self.movie_metadata[idx]
                if movie['title'].lower().strip() != movie_title.lower().strip():
                    results.append({
                        'title': movie['title'],
                        'year': movie['year'],
                        'similarity': float(score)
                    })
        
        return results[:k]
    
    def search_genre_similarity(self, movie_title: str, k: int = 10):
        """Search similar movies based on genres"""
        target_movie, target_idx = self.find_movie_by_title(movie_title)
        if target_movie is None:
            return []
        
        # Search similar movies
        scores, indices = self.genre_index.search(
            np.array([self.genre_index.reconstruct(target_idx)]), k + 1
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.movie_metadata):
                movie = self.movie_metadata[idx]
                if movie['title'].lower().strip() != movie_title.lower().strip():
                    results.append({
                        'title': movie['title'],
                        'year': movie['year'],
                        'similarity': float(score)
                    })
        
        return results[:k]
    
    def calculate_weighted_similarity(self, movie_title: str, w_plot: float = 0.65, 
                                    w_flow: float = 0.25, w_genre: float = 0.10, 
                                    top_k: int = 10):
        """Calculate weighted similarity combining all modalities"""
        print(f"🔍 Searching for movies similar to: {movie_title}")
        
        # Get candidates from each modality (get more to increase chance of overlap)
        plot_candidates = self.search_plot_similarity(movie_title, 50) if w_plot > 0 else []
        flow_candidates = self.search_flow_similarity(movie_title, 50) if w_flow > 0 else []
        genre_candidates = self.search_genre_similarity(movie_title, 50) if w_genre > 0 else []
        
        print(f"Found candidates: Plot({len(plot_candidates)}), Flow({len(flow_candidates)}), Genre({len(genre_candidates)})")
        
        # Create union of all candidates
        all_candidates = {}
        overlap_count = 0
        
        # Add plot candidates
        for movie in plot_candidates:
            key = (movie['title'], movie['year'])
            all_candidates[key] = {
                'title': movie['title'],
                'year': movie['year'],
                'plot_similarity': movie['similarity'],
                'flow_similarity': 0.0,
                'genre_similarity': 0.0
            }
        
        # Add flow candidates
        for movie in flow_candidates:
            key = (movie['title'], movie['year'])
            if key in all_candidates:
                all_candidates[key]['flow_similarity'] = movie['similarity']
                overlap_count += 1
            else:
                all_candidates[key] = {
                    'title': movie['title'],
                    'year': movie['year'],
                    'plot_similarity': 0.0,
                    'flow_similarity': movie['similarity'],
                    'genre_similarity': 0.0
                }
        
        # Add genre candidates  
        for movie in genre_candidates:
            key = (movie['title'], movie['year'])
            if key in all_candidates:
                all_candidates[key]['genre_similarity'] = movie['similarity']
                overlap_count += 1
            else:
                all_candidates[key] = {
                    'title': movie['title'],
                    'year': movie['year'],
                    'plot_similarity': 0.0,
                    'flow_similarity': 0.0,
                    'genre_similarity': movie['similarity']
                }
        
        print(f"Total candidates: {len(all_candidates)}, Overlapping scores: {overlap_count}")
        
        # Calculate weighted scores
        final_candidates = []
        total_weight = w_plot + w_flow + w_genre
        
        for movie_data in all_candidates.values():
            weighted_score = (
                (w_plot * movie_data['plot_similarity'] +
                 w_flow * movie_data['flow_similarity'] +
                 w_genre * movie_data['genre_similarity']) / total_weight
            )
            
            # Find additional metadata
            movie_meta = None
            for meta in self.movie_metadata:
                if (meta['title'] == movie_data['title'] and 
                    str(meta['year']) == str(movie_data['year'])):
                    movie_meta = meta
                    break
            
            result = {
                'rank': 0,
                'title': movie_data['title'],
                'year': movie_data['year'],
                'director': movie_meta.get('director', 'Unknown') if movie_meta else 'Unknown',
                'similarity_score': weighted_score,
                'component_scores': {
                    'plot': movie_data['plot_similarity'],
                    'flow': movie_data['flow_similarity'],
                    'genre': movie_data['genre_similarity']
                },
                'weights_used': {
                    'plot': w_plot / total_weight,
                    'flow': w_flow / total_weight,
                    'genre': w_genre / total_weight
                }
            }
            
            if movie_meta:
                result['genres'] = movie_meta.get('genres', {})
                result['plot'] = movie_meta.get('plot', '')
            
            final_candidates.append(result)
        
        # Sort by weighted score and add ranks
        final_candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        for i, movie in enumerate(final_candidates[:top_k]):
            movie['rank'] = i + 1
        
        return final_candidates[:top_k]

# Factory function for easy import
def get_weighted_calculator(data_dir: str = "data"):
    """Get a WeightedSimilarityCalculator instance"""
    return WeightedSimilarityCalculator(data_dir)

# Test functionality
if __name__ == "__main__":
    calculator = get_weighted_calculator()
    if calculator.load_all_data():
        print("\n🧪 Testing with Inception...")
        results = calculator.calculate_weighted_similarity("Inception", top_k=5)
        
        for movie in results:
            print(f"\n{movie['rank']}. {movie['title']} ({movie['year']})")
            print(f"   Score: {movie['similarity_score']:.4f}")
            scores = movie['component_scores']
            print(f"   Components: plot={scores['plot']:.3f}, flow={scores['flow']:.3f}, genre={scores['genre']:.3f}")
    else:
        print("❌ Failed to load data")
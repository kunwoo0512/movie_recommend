#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create SBERT embeddings from movies_dataset.json plots.
Outputs:
- data/embeddings.npy (N, D) float32, L2-normalized
- data/metadata.jsonl (one JSON per line)
- data/manifest.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('Input JSON must be a list of records')
    return data


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='movies_dataset.json')
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f'[1/4] Loading records from {input_path}')
    records = load_records(input_path)

    texts = []
    for r in records:
        texts.append(normalize_whitespace(r.get('plot', '') or ''))

    meta_path = out_dir / 'metadata.jsonl'
    print(f'[2/4] Writing metadata to {meta_path}')
    with open(meta_path, 'w', encoding='utf-8') as f:
        for r, t in zip(records, texts):
            item = {
                'title': r.get('title'),
                'year': r.get('year'),
                'director': r.get('director'),
                'plot': t,
                'genres': r.get('genres'),
                'movie_id': r.get('movie_id')
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'[3/4] Loading SBERT model: {args.model}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(args.model, device=device)

    all_vecs = []
    print(f'[4/4] Encoding texts (batch_size={args.batch_size}, device={device})')
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i+args.batch_size]
        with torch.inference_mode():
            emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
        emb = emb.astype('float32')
        # L2 normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        all_vecs.append(emb)

    embeddings = np.vstack(all_vecs).astype('float32')
    emb_path = out_dir / 'embeddings.npy'
    np.save(emb_path, embeddings)
    print(f'✅ Saved embeddings to {emb_path} (shape={embeddings.shape})')

    manifest = {
        'model': args.model,
        'dim': int(embeddings.shape[1]),
        'count': int(embeddings.shape[0]),
        'normalized': True,
        'text_key': 'plot',
        'source': str(input_path)
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'✅ Saved manifest to {out_dir / "manifest.json"}')


if __name__ == '__main__':
    main()

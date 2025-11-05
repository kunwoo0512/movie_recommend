#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentenceBERT 기반 영화 줄거리 임베딩 생성기 (윈도우 인덱싱 버전)
- 한국어 쿼리 ↔ 영어(또는 혼합) 줄거리 매칭
- 문장 분할 후 2~3문장 슬라이딩 윈도우 임베딩
- L2 정규화 및 제로벡터/NaN 가드
- 결과: embeddings.npy + metadata.jsonl (FAISS 호환)
"""

"""python bert_embedding_generator_new.py --input movies_dataset.json --output_dir data --window_size 2 --stride 1 --batch_size 16
1. build_faiss_and_query.py --build 로 FAISS 인덱스 생성
2. build_faiss_and_query.py --demo_query '검색어' 로 테스트

"""

import os
import re
import json
import time
import argparse
import warnings
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

warnings.filterwarnings("ignore")


# -----------------------------
# 유틸: 문장 분할 & 윈도우 구성
# -----------------------------

def split_sentences(text: str) -> List[str]:
    """외부 라이브러리 없이 가벼운 문장 분할"""
    if not text:
        return []
    
    # 공백 정돈
    t = re.sub(r"\s+", " ", text.strip())
    
    # 문장 경계 휴리스틱 (간단/경량)
    sent_boundaries = re.compile(
        r"""
        (?<=[\.!\?])\s+       |   # 영문 .!? 뒤 공백
        (?<=[다요죠음임니니까까]\.)\s+ |   # 한글 흔한 종결 + 마침표
        (?<=[다요죠음임니니까까])\s+(?=[A-Z가-힣0-9])   # 한글 종결 뒤 공백
        """,
        re.VERBOSE
    )
    
    # 너무 긴 줄 대비: 구두점 없으면 대강 120~200자마다 끊기
    if not re.search(r"[\.!\?]", t) and len(t) > 200:
        chunks = [t[i:i+180] for i in range(0, len(t), 180)]
        return [c.strip() for c in chunks if c.strip()]
    
    parts = re.split(sent_boundaries, t)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents

def build_windows(sents: List[str], window_size: int = 2, stride: int = 1, max_chars: int = 600) -> List[Tuple[int, int, str]]:
    """
    2~3문장 슬라이딩 윈도우.
    반환: (start_sent_idx, end_sent_idx_exclusive, window_text)
    """
    windows = []
    n = len(sents)
    if n == 0:
        return windows
    
    for start in range(0, max(1, n - window_size + 1), stride):
        end = min(n, start + window_size)
        chunk = " ".join(sents[start:end]).strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars]
        if chunk:
            windows.append((start, end, chunk))
    
    # 만약 문장이 1개뿐이면 최소 1창 확보
    if not windows and n > 0:
        chunk = sents[0][:max_chars]
        windows.append((0, 1, chunk))
    
    return windows


# -----------------------------
# 임베딩 래퍼
# -----------------------------

class SBertEmbedder:
    def __init__(self, model_name: str, device: str = None, batch_size: int = 32):
        self.model_name = model_name
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[모델] {model_name} 로딩 중... (장치: {device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        print(f"[완료] 모델 로드 완료 (차원: {self.get_dim()})")

    def encode_texts(self, texts: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        텍스트 리스트 → L2 정규화된 임베딩 (float32)
        - normalize_embeddings=True 로 내적=코사인
        - 반환: (valid_embeddings, valid_indices)
        """
        if not texts:
            return np.zeros((0, self.get_dim()), dtype="float32"), []
        
        print(f"[임베딩] {len(texts)}개 텍스트 처리 중...")
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # ★ 중요: 정규화
            batch_size=self.batch_size,
            show_progress_bar=True
        ).astype("float32")

        # NaN/Inf/제로 가드
        valid_mask = (
            np.isfinite(vecs).all(axis=1) &
            (np.linalg.norm(vecs, axis=1) >= 1e-6)
        )
        
        if (~valid_mask).any():
            invalid_count = (~valid_mask).sum()
            print(f"[경고] 비정상 임베딩 {invalid_count}개 제거 (남은 {valid_mask.sum()}개)")
            vecs = vecs[valid_mask]
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
        else:
            valid_indices = list(range(len(texts)))
            
        return vecs, valid_indices

    def get_dim(self) -> int:
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            return 384


# -----------------------------
# 메인 파이프라인
# -----------------------------

def process_dataset(
    input_json: str,
    output_dir: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    window_size: int = 2,
    stride: int = 1,
    batch_size: int = 32,
    use_windows: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[설정] model={model_name}")
    print(f"[설정] window_size={window_size}, stride={stride}, batch_size={batch_size}")
    print(f"[입력] {input_json}")
    print(f"[출력] {output_dir}")

    # 1) 데이터 로드
    with open(input_json, "r", encoding="utf-8") as f:
        movies = json.load(f)
    assert isinstance(movies, list), "movies_dataset.json은 리스트여야 합니다."
    print(f"[로드] {len(movies)}편 영화 데이터 로드")

    # 2) 임베딩 생성기 초기화
    embedder = SBertEmbedder(model_name=model_name, batch_size=batch_size)
    
    if use_windows:
        return _process_with_windows(movies, embedder, output_dir, window_size, stride)
    else:
        return _process_full_plots(movies, embedder, output_dir)

def _process_with_windows(movies, embedder, output_dir, window_size, stride):
    """윈도우 기반 처리"""
    print(f"[모드] 윈도우 기반 처리 (window_size={window_size}, stride={stride})")
    
    # 줄거리 → 문장 → 윈도우
    all_texts = []
    all_metadata = []
    
    for mi, movie in enumerate(tqdm(movies, desc="윈도우 생성")):
        title = movie.get("title", f"Movie_{mi}")
        plot = (movie.get("plot") or "").strip()
        if not plot:
            plot = title
            
        sents = split_sentences(plot)
        windows = build_windows(sents, window_size=window_size, stride=stride)
        
        for wi, (start, end, text) in enumerate(windows):
            all_texts.append(text)
            all_metadata.append({
                "movie_index": mi,
                "title": title,
                "window_index": wi,
                "start_sent": start,
                "end_sent": end,
                "text": text,
                "year": movie.get("year", ""),
                "director": movie.get("director", ""),
                "genres": movie.get("genres", {}),
                "movie_id": movie.get("movie_id", f"movie_{mi}")
            })
    
    print(f"[통계] 총 {len(all_texts)}개 윈도우 생성")
    
    # 임베딩 생성
    embeddings, valid_indices = embedder.encode_texts(all_texts)
    
    # 유효한 메타데이터만 유지
    valid_metadata = [all_metadata[i] for i in valid_indices]
    
    # 저장 (FAISS 호환 형식)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    
    with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for meta in valid_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    
    print(f"[저장] embeddings.npy: {embeddings.shape}")
    print(f"[저장] metadata.jsonl: {len(valid_metadata)}개 항목")

def _process_full_plots(movies, embedder, output_dir):
    """전체 줄거리 기반 처리 (기존 방식)"""
    print(f"[모드] 전체 줄거리 기반 처리")
    
    all_texts = []
    all_metadata = []
    
    for mi, movie in enumerate(movies):
        title = movie.get("title", f"Movie_{mi}")
        plot = (movie.get("plot") or "").strip()
        if not plot:
            plot = title
            
        all_texts.append(plot)
        all_metadata.append({
            "movie_index": mi,
            "title": title,
            "plot": plot,
            "year": movie.get("year", ""),
            "director": movie.get("director", ""),
            "genres": movie.get("genres", {}),
            "movie_id": movie.get("movie_id", f"movie_{mi}")
        })
    
    # 임베딩 생성
    embeddings, valid_indices = embedder.encode_texts(all_texts)
    
    # 유효한 메타데이터만 유지
    valid_metadata = [all_metadata[i] for i in valid_indices]
    
    # 저장 (FAISS 호환 형식)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    
    with open(os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for meta in valid_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    
    print(f"[저장] embeddings.npy: {embeddings.shape}")
    print(f"[저장] metadata.jsonl: {len(valid_metadata)}개 항목")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="SentenceBERT 영화 줄거리 임베딩 생성기")
    parser.add_argument("--input", type=str, default="movies_dataset.json", help="입력 JSON 경로")
    parser.add_argument("--output_dir", type=str, default="data", help="출력 디렉토리")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="SentenceBERT 모델명")
    parser.add_argument("--window_size", type=int, default=2, help="윈도우 문장 수 (2~3 권장)")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 스트라이드")
    parser.add_argument("--batch_size", type=int, default=32, help="임베딩 배치 크기")
    parser.add_argument("--no_windows", action="store_true", help="윈도우 모드 비활성화 (전체 줄거리 사용)")
    
    args = parser.parse_args()
    
    print("🎬 SentenceBERT 기반 영화 임베딩 생성기")
    print("=" * 60)
    
    try:
        process_dataset(
            input_json=args.input,
            output_dir=args.output_dir,
            model_name=args.model,
            window_size=args.window_size,
            stride=args.stride,
            batch_size=args.batch_size,
            use_windows=not args.no_windows
        )
        
        print("\n✅ 임베딩 생성 완료!")
        print("\n다음 단계:")
        print("1. build_faiss_and_query.py --build 로 FAISS 인덱스 생성")
        print("2. build_faiss_and_query.py --demo_query '검색어' 로 테스트")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n해결 방법:")
        print("1. sentence-transformers 설치: pip install sentence-transformers")
        print("2. 메모리 부족 시 --batch_size 줄이기")
        print("3. GPU 메모리 부족 시 CPU 사용")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Late-interaction RAG pipeline using PyLate (ColBERT) with manual MaxSim scoring.

Demonstrates: model loading, token embedding extraction, MaxSim search,
and BM25 + ColBERT hybrid reranking.

Install: pip install pylate bm25s stemmer
Run: python scripts/ragatouille_pipeline.py
"""

import time
import numpy as np
from typing import List, Dict


# Sample corpus — ML optimization concepts
CORPUS = [
    "Gradient checkpointing trades compute for memory by recomputing activations during backpropagation. It reduces memory from O(L) to O(sqrt(L)).",
    "Flash Attention computes exact attention in O(N^2) time but O(N) memory by tiling the computation and avoiding materialization of the full attention matrix.",
    "Speculative decoding uses a smaller draft model to propose multiple candidate tokens, verified in parallel by the large model. Achieves 2-3x inference speedup.",
    "KV cache stores key-value tensors from previous autoregressive steps to avoid recomputation. Requires O(N*H*L) memory per sequence.",
    "Post-training quantization reduces precision from FP16 to INT4, shrinking memory by 4x. Negligible accuracy loss on most benchmarks.",
    "PagedAttention manages the KV cache in fixed-size blocks (pages), eliminating fragmentation. Core technique behind vLLM serving system.",
    "Continuous batching processes requests as they arrive rather than waiting for fixed-size batches. Improves throughput by 10-20x.",
    "LoRA freezes pretrained weights and injects trainable rank decomposition matrices into attention layers. Reduces parameters from billions to millions.",
    "Prefix caching stores KV cache entries for common prefixes across requests, enabling 5-10x faster prefill for shared prompt prefixes.",
    "Tensor parallelism shards model parameters across GPUs, with each GPU computing a portion of each layer. Essential for models larger than single GPU memory.",
]

QUERIES = [
    "how to reduce GPU memory during inference",
    "techniques for faster transformer decoding speed",
    "memory efficient attention mechanisms for long sequences",
    "methods to reduce model size for deployment",
    "serving large language models efficiently",
]


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """ColBERT MaxSim: sum of max cos similarities per query token."""
    # Normalize for cosine similarity
    q_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    d_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-10)
    # Similarity matrix (n_query_tokens, n_doc_tokens)
    sim = np.dot(q_norm, d_norm.T)
    # Max per query token, then sum
    return float(sim.max(axis=1).sum())


def load_model():
    """Load a ColBERT model."""
    from pylate import models
    print("Loading ColBERT model (answerai-colbert-small-v1)...")
    start = time.time()
    model = models.ColBERT(model_name_or_path="answerdotai/answerai-colbert-small-v1")
    print(f"  Loaded in {time.time() - start:.2f}s")
    return model


def token_embeddings(model, texts: List[str]) -> List[np.ndarray]:
    """Get per-token embeddings for a list of texts."""
    return model.encode(texts)  # List of (n_tokens, dim) numpy arrays


def build_index(corpus: List[str], model) -> List[np.ndarray]:
    """Pre-compute token embeddings for all documents."""
    print(f"\nIndexing {len(corpus)} documents...")
    start = time.time()
    doc_embeddings = token_embeddings(model, corpus)
    elapsed = time.time() - start
    print(f"  Indexed in {elapsed:.2f}s")
    total_tokens = sum(emb.shape[0] for emb in doc_embeddings)
    print(f"  Total token vectors: {total_tokens}")
    print(f"  Avg tokens per doc: {total_tokens / len(corpus):.0f}")
    return doc_embeddings


def search(query: str, model, doc_embeddings: List[np.ndarray],
           corpus: List[str], k: int = 5) -> List[Dict]:
    """Search using MaxSim scoring."""
    start = time.time()
    q_emb = token_embeddings(model, [query])[0]

    scores = [maxsim_score(q_emb, d_emb) for d_emb in doc_embeddings]
    elapsed = time.time() - start

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "rank": len(results) + 1,
            "score": scores[idx],
            "content": corpus[idx],
            "index": int(idx),
        })

    print(f"  Retrieved in {elapsed*1000:.1f}ms")
    return results


def run_searches(model, doc_embeddings):
    """Run all predefined queries."""
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for query in QUERIES:
        print(f"\n--- Query: '{query}' ---")
        results = search(query, model, doc_embeddings, CORPUS, k=3)
        for r in results:
            print(f"  [{r['rank']}] ({r['score']:.4f}) {r['content'][:90]}...")


def rerank_demo(model, doc_embeddings):
    """BM25 first-pass, then ColBERT rerank."""
    print("\n" + "=" * 60)
    print("HYBRID: BM25 -> ColBERT RERANK")
    print("=" * 60)

    try:
        import bm25s
        import Stemmer

        query = "attention memory optimization"
        stemmer = Stemmer.Stemmer("english")

        # BM25 first stage
        bm25 = bm25s.BM25()
        tokenized = bm25s.tokenize(CORPUS, stemmer=stemmer)
        bm25.index(tokenized)

        tokenized_q = bm25s.tokenize(query, stemmer=stemmer)
        bm25_results, bm25_scores = bm25.retrieve(tokenized_q, k=5)
        candidate_indices = bm25_results[0].tolist()

        print(f"\nQuery: '{query}'")
        print("BM25 top-5:")
        for i, idx in enumerate(candidate_indices):
            print(f"  [{i+1}] (BM25={bm25_scores[0][i]:.1f}) {CORPUS[idx][:80]}...")

        # ColBERT rerank using MaxSim
        q_emb = token_embeddings(model, [query])[0]
        reranked = sorted(
            [(idx, maxsim_score(q_emb, doc_embeddings[idx])) for idx in candidate_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        print("\nColBERT reranked:")
        for i, (idx, score) in enumerate(reranked):
            print(f"  [{i+1}] ({score:.4f}) {CORPUS[idx][:80]}...")

    except ImportError:
        print("bm25s/stemmer not installed. pip install bm25s stemmer")


def save_load_demo(model, doc_embeddings, corpus):
    """Save embeddings to disk and reload."""
    print("\n" + "=" * 60)
    print("SAVE/LOAD DEMO")
    print("=" * 60)

    import pickle, os

    # Save
    save_dir = "./indexes/ml_optimization"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/embeddings.pkl", "wb") as f:
        pickle.dump({"corpus": corpus, "embeddings": doc_embeddings}, f)
    print(f"  Saved to {save_dir}/")

    # Reload
    start = time.time()
    with open(f"{save_dir}/embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Reloaded in {time.time() - start:.2f}s")

    # Verify
    q_emb = token_embeddings(model, ["model quantization"])[0]
    scores = [maxsim_score(q_emb, d) for d in data["embeddings"]]
    top_idx = np.argmax(scores)
    print(f"  Query: 'model quantization'")
    print(f"  Best match ({scores[top_idx]:.4f}): {data['corpus'][top_idx][:80]}...")


if __name__ == "__main__":
    model = load_model()
    doc_embeddings = build_index(CORPUS, model)

    run_searches(model, doc_embeddings)
    rerank_demo(model, doc_embeddings)
    save_load_demo(model, doc_embeddings, CORPUS)

    print("\nDone!")

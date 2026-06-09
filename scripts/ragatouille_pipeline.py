#!/usr/bin/env python3
"""
Complete late-interaction RAG pipeline using RAGatouille (ColBERT).

Demonstrates indexing, search, and reranking.

Install: pip install ragatouille bm25s stemmer
Run: python scripts/ragatouille_pipeline.py
"""

from ragatouille import RAGPretrainedModel
import time
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

EXPECTED_MATCHES = {
    "how to reduce GPU memory during inference": [0, 1, 3, 4, 5],
    "techniques for faster transformer decoding speed": [2, 5, 6, 8],
    "memory efficient attention mechanisms for long sequences": [1, 3, 5],
    "methods to reduce model size for deployment": [4, 7, 9],
    "serving large language models efficiently": [5, 6, 9, 8],
}


def build_and_index(corpus: List[str]) -> RAGPretrainedModel:
    """Index documents with ColBERTv2."""
    print("Loading ColBERTv2 model...")
    start = time.time()
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    print(f"  Model loaded in {time.time() - start:.2f}s")

    print("\nIndexing corpus...")
    start = time.time()
    RAG.index(
        collection=corpus,
        index_name="ml_optimization",
        max_document_length=256,
        split_documents=False,
        overwrite_index=True,
    )
    print(f"  Indexed {len(corpus)} documents in {time.time() - start:.2f}s")
    return RAG


def search(rag: RAGPretrainedModel, queries: List[str], k: int = 5):
    """Search and show results for each query."""
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        start = time.time()
        results = rag.search(query=query, k=k)
        elapsed = time.time() - start

        print(f"  Retrieved in {elapsed*1000:.1f}ms")
        for i, r in enumerate(results):
            rank = r.get("rank", i + 1)
            score = r.get("score", 0.0)
            content = r.get("content", "")[:100]
            print(f"  [{rank}] ({score:.4f}) {content}...")


def rerank_demo(rag: RAGPretrainedModel):
    """Demonstrate ColBERT as a reranker over BM25 candidates."""
    print("\n" + "=" * 60)
    print("RERANKING DEMO (BM25 -> ColBERT)")
    print("=" * 60)

    try:
        import bm25s
        import Stemmer

        query = "attention memory optimization"
        stemmer = Stemmer.Stemmer("english")

        # BM25 first-stage retrieval
        bm25 = bm25s.BM25()
        tokenized = bm25s.tokenize(CORPUS, stemmer=stemmer)
        bm25.index(tokenized)

        tokenized_q = bm25s.tokenize(query, stemmer=stemmer)
        bm25_results, bm25_scores = bm25.retrieve(tokenized_q, k=5)
        candidates = [CORPUS[idx] for idx in bm25_results[0]]

        print(f"\nQuery: '{query}'")
        print(f"BM25 candidates:")
        for i, (doc, score) in enumerate(zip(candidates, bm25_scores[0])):
            print(f"  [{i+1}] (BM25={score:.2f}) {doc[:80]}...")

        # ColBERT rerank
        start = time.time()
        reranked = rag.rerank(query=query, documents=candidates, k=5)
        elapsed = time.time() - start

        print(f"\nColBERT reranked ({elapsed*1000:.1f}ms):")
        for i, r in enumerate(reranked):
            print(f"  [{i+1}] ({r['score']:.4f}) {r['content'][:80]}...")

    except ImportError:
        print("bm25s and/or Stemmer not installed. Skipping reranking demo.")
        print("Install: pip install bm25s stemmer")


def save_load_demo(rag: RAGPretrainedModel):
    """Demonstrate saving and reloading the index."""
    print("\n" + "=" * 60)
    print("SAVE/LOAD DEMO")
    print("=" * 60)

    # Re-index with save
    print("\nRe-indexing and saving...")
    rag.index(
        collection=CORPUS,
        index_name="ml_optimization",
        overwrite_index=True,
    )

    # Reload
    print("Loading from saved index...")
    start = time.time()
    rag_loaded = RAGPretrainedModel.from_index(
        ".ragatouille/colbert/indexes/ml_optimization"
    )
    print(f"  Reloaded in {time.time() - start:.2f}s")

    # Verify with a query
    results = rag_loaded.search(query="model quantization", k=3)
    print("  Query after reload: 'model quantization'")
    for r in results:
        print(f"    [{r['score']:.4f}] {r['content'][:80]}...")


if __name__ == "__main__":
    rag_model = build_and_index(CORPUS)
    search(rag_model, QUERIES, k=5)
    rerank_demo(rag_model)
    save_load_demo(rag_model)

    print("\nDone!")

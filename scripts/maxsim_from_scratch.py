#!/usr/bin/env python3
"""
Complete MaxSim implementation from scratch in NumPy.

Demonstrates the core late-interaction scoring mechanism.
Run: python scripts/maxsim_from_scratch.py
"""

import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of A and every row of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return np.dot(A_norm, B_norm.T)


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """
    Compute ColBERT-style MaxSim score.

    score(Q, D) = sum_i max_j cos(q_i, d_j)

    Each query token independently finds its best match in the document.
    """
    sim = cosine_similarity_matrix(query_emb, doc_emb)
    max_per_query = sim.max(axis=1)
    return float(max_per_query.sum())


def batch_maxsim(query_emb: np.ndarray, doc_embeddings: list) -> np.ndarray:
    """Score one query against multiple documents."""
    scores = np.array([maxsim_score(query_emb, doc) for doc in doc_embeddings])
    return scores


def demonstrate_multi_aspect():
    """Show how late interaction handles multi-aspect queries where
    bi-encoders fail."""
    print("=== Multi-Aspect Matching Demo ===")
    print("Query: 'memory optimization for inference'\n")

    # Simulate query tokens (in practice from ColBERT model)
    query = np.array([
        [1.0, 0.0, 0.0, 0.0],  # "memory"
        [0.0, 1.0, 0.0, 0.0],  # "optimization"
        [0.0, 0.0, 1.0, 0.0],  # "inference"
        [0.2, 0.2, 0.2, 0.8],  # [MASK] augmented token (generic)
    ], dtype=float)

    # Document A: talks about memory during TRAINING (dominant signal)
    doc_a = np.vstack([
        np.random.randn(100, 4) * 0.05,   # noise
        np.array([[0.9, 0.1, 0.0, 0.0]]),  # "memory" match
        np.random.randn(100, 4) * 0.05,   # noise
        np.array([[0.1, 0.8, 0.1, 0.0]]),  # "optimization" weak match
        np.random.randn(50, 4) * 0.05,    # noise
    ])
    doc_a = doc_a / (np.linalg.norm(doc_a, axis=1, keepdims=True) + 1e-10)

    # Document B: talks about memory during INFERENCE (sparse signal, but relevant)
    doc_b = np.vstack([
        np.random.randn(200, 4) * 0.05,  # mostly unrelated content
        np.array([[0.0, 0.0, 0.9, 0.0]]),  # "inference" match (single mention)
        np.random.randn(50, 4) * 0.05,
        np.array([[0.0, 0.1, 0.0, 0.8]]),  # optimization during inference
    ])
    doc_b = doc_b / (np.linalg.norm(doc_b, axis=1, keepdims=True) + 1e-10)

    score_a = maxsim_score(query, doc_a)
    score_b = maxsim_score(query, doc_b)

    print(f"Document A (training-focused):  MaxSim = {score_a:.4f}")
    print(f"Document B (inference-focused): MaxSim = {score_b:.4f}")
    print()

    # Now simulate what a BI-ENCODER would do (mean pooling)
    query_pooled = query.mean(axis=0, keepdims=True)
    doc_a_pooled = doc_a.mean(axis=0, keepdims=True)
    doc_b_pooled = doc_b.mean(axis=0, keepdims=True)

    cos = cosine_similarity_matrix(query_pooled, doc_a_pooled)[0, 0]
    cos_b = cosine_similarity_matrix(query_pooled, doc_b_pooled)[0, 0]

    print("Compare with bi-encoder (mean pooling):")
    print(f"  Document A: cosine = {cos:.4f}")
    print(f"  Document B: cosine = {cos_b:.4f}")
    print()

    if cos > cos_b:
        print("Bi-encoder prefers Document A (training noise dominates)")
    else:
        print("Bi-encoder prefers Document B")
    print()

    if score_a < score_b:
        print("ColBERT prefers Document B (correct — it matches 'inference' token)")
    elif score_a > score_b:
        print("ColBERT prefers Document A")
    else:
        print("ColBERT scores are equal")

    print()
    print("Takeaway: Each query token independently finds its best document match.")
    print("Sparse signals (single mention of 'inference') aren't drowned out.")
    print()


def demonstrate_explainability():
    """Show which query tokens matched which document tokens."""
    print("=== Explainability Demo ===")

    query_tokens = ["memory", "optimization", "during", "inference", "[MASK]"]
    doc_tokens = ["We", "present", "Flash", "Attention", "which", "reduces",
                  "memory", "consumption", "from", "quadratic", "to", "linear",
                  "during", "inference", "."]

    np.random.seed(42)
    d = 16  # small dim for readability
    query_emb = np.random.randn(len(query_tokens), d)
    doc_emb = np.random.randn(len(doc_tokens), d)

    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    doc_emb = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)

    sim = np.dot(query_emb, doc_emb.T)
    best_doc_indices = sim.argmax(axis=1)

    print("Token-level matching:")
    for q_idx, d_idx in enumerate(best_doc_indices):
        print(f"  '{query_tokens[q_idx]:<15}' → '{doc_tokens[d_idx]}'  (cos={sim[q_idx, d_idx]:.3f})")

    contributions = sim.max(axis=1)
    print(f"\nTotal score: {contributions.sum():.3f}")
    print(f"Individual contributions: {dict(zip(query_tokens, [f'{c:.3f}' for c in contributions]))}")


if __name__ == "__main__":
    np.random.seed(42)

    # Basic MaxSim
    print("=== Basic MaxSim ===")
    n_query, n_doc, d = 8, 200, 128
    q_emb = np.random.randn(n_query, d)
    d_emb = np.random.randn(n_doc, d)
    score = maxsim_score(q_emb, d_emb)
    print(f"Query tokens: {n_query}, Doc tokens: {n_doc}, Dim: {d}")
    print(f"MaxSim score: {score:.4f}\n")

    demonstrate_multi_aspect()
    demonstrate_explainability()

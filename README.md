# Better RAG Retrieval with Late Interaction

A technical deep dive into late-interaction retrieval models: ColBERT, MaxSim, PLAID, ColPali, and the ecosystem of tools that make them production-ready.

Inspired by the Maven lesson "Better RAG retrieval with late interaction" by Isaac Flath and Hamel Husain (June 10, 2026).

## Table of Contents

1. [Why Late Interaction](#1-why-late-interaction)
2. [The Three Retrieval Paradigms](#2-the-three-retrieval-paradigms)
3. [ColBERT Architecture Deep Dive](#3-colbert-architecture-deep-dive)
4. [MaxSim: The Core Scoring Mechanism](#4-maxsim-the-core-scoring-mechanism)
5. [ColBERTv2: Residual Compression](#5-colbertv2-residual-compression)
6. [PLAID: Performance-Optimized Late Interaction Driver](#6-plaid-performance-optimized-late-interaction-driver)
7. [ColPali: Visual Document Retrieval with VLMs](#7-colpali-visual-document-retrieval-with-vlms)
8. [Mixedbread AI Late Interaction Models](#8-mixedbread-ai-late-interaction-models)
9. [LightOn / PyLate: Training and Inference Toolkit](#9-lighton--pylate-training-and-inference-toolkit)
10. [Jina AI: Jina-ColBERT-v2](#10-jina-ai-jina-colbert-v2)
11. [Practical: Building a Late-Interaction RAG Pipeline with RAGatouille](#11-practical-building-a-late-interaction-rag-pipeline-with-ragatouille)
12. [Practical: Custom ColBERT Training with PyLate](#12-practical-custom-colbert-training-with-pylate)
13. [Practical: ColPali for PDF and Image Retrieval](#13-practical-colpali-for-pdf-and-image-retrieval)
14. [When Dense Vectors Break](#14-when-dense-vectors-break)
15. [Production Deployment Considerations](#15-production-deployment-considerations)
16. [Benchmarking and Evaluation](#16-benchmarking-and-evaluation)
17. [Decision Guide: Which Approach When](#17-decision-guide-which-approach-when)
18. [Ecosystem Map and Vendor Comparison](#18-ecosystem-map-and-vendor-comparison)
19. [Future Directions](#19-future-directions)
20. [Appendix: Full Code Reference](#20-appendix-full-code-reference)

---

## 1. Why Late Interaction

Standard dense retrieval (bi-encoder RAG) embeds a query into a single vector and retrieves documents by computing cosine similarity against pre-computed document vectors. It's fast and scalable. But the single-vector compression loses information: the query's meaning is squeezed into one fixed-size representation that must account for every possible document comparison.

**Example:** The query _"What did the paper say about memory usage during inference?"_ has four distinct aspects: paper reference, memory usage, inference phase, and a specific claim. A single 768-dimensional vector must compress all four into one representation. If a document discusses memory during training but mentions inference briefly in a footnote, the single-vector approach will rank it poorly because the dominant signal (training) drowns out the relevant signal (inference).

Late interaction solves this by preserving per-token representations for both query and document, then computing relevance as a fine-grained token-level matching. Each query token independently finds its best match in the document.

**When to reach for late interaction:**

- Long documents where signal is sparse (a relevant sentence buried in a wall of text)
- Multi-aspect queries ("Find documents about GPU memory AND gradient checkpointing AND fp16")
- Out-of-domain retrieval where the embedding model wasn't trained on your data distribution
- Recall-critical applications where missing a relevant document is costly

---

## 2. The Three Retrieval Paradigms

### 2.1 Bi-Encoder (Standard Dense Retrieval)

```
Query: [CLS] how to reduce memory [SEP] → BERT → [v_query] (768d)
Doc:   [CLS] Flash attention reduces... [SEP] → BERT → [v_doc]  (768d)
Score: cosine(v_query, v_doc)
```

Each query and document is encoded independently into a single vector. All document vectors are pre-computed and indexed offline with ANN (HNSW, IVF, etc.). At query time, one forward pass encodes the query, one ANN lookup retrieves top-k.

- **Storage:** ~3KB per document (768 float32)
- **Query latency:** ~5ms (ANN index lookup)
- **Accuracy:** Good for topical retrieval, weak for fine-grained matching

### 2.2 Cross-Encoder (Reranker)

```
Query + Doc: [CLS] how to reduce memory [SEP] Flash attention reduces... [SEP]
           → BERT → [CLS] → linear → relevance score
```

Query and document are concatenated and passed through the full transformer together. Full self-attention between all tokens means the model can reason about query-document interactions in detail.

- **Storage:** 0 (no pre-computed embeddings)
- **Query latency:** ~50-200ms per document (full forward pass for each candidate)
- **Accuracy:** Highest, but doesn't scale to full corpus search (max ~100 candidates)
- **Typical use:** Rerank top-50 from a first-stage retriever

### 2.3 Late Interaction (ColBERT)

```
Query: [Q1, Q2, Q3, Q4] → BERT → [q1, q2, q3, q4] (token matrix)
Doc:   [D1, D2, D3, ..., D200] → BERT → [d1, d2, ..., d200] (token matrix)
Score: Σ_i max_j cosine(q_i, d_j)
```

Queries and documents are encoded independently (like a bi-encoder), but each produces a matrix of token-level embeddings rather than a single pooled vector. The scoring function (MaxSim) compares every query token against every document token.

- **Storage:** ~600KB per document (200 tokens × 768 float32), ~60KB with quantization
- **Query latency:** ~50ms (MaxSim over pre-computed document embeddings)
- **Accuracy:** Near cross-encoder quality, bi-encoder speed at query time

### 2.4 Comparison Table

| Property | Bi-Encoder | Late Interaction | Cross-Encoder |
|---|---|---|---|
| Query encoding | 1 vector | N vectors (token-level) | Full text |
| Document encoding | 1 vector (pre-computed) | M vectors (pre-computed) | Per query (can't precompute) |
| Scoring | 1 dot product | N×M dot products | Full attention |
| Storage per 200-token doc | 3 KB | 600 KB (60 KB quantized) | 0 |
| Query latency (1M corpus) | ~5ms | ~50ms | N/A (rerank only) |
| Top-50 rerank latency | N/A | ~5ms | ~5-10s |
| Recall@1000 | Good | Excellent | N/A |
| Training complexity | Low | Medium | Medium |

---

## 3. ColBERT Architecture Deep Dive

ColBERT (Contextualized Late Interaction over BERT) was introduced by Khattab and Zaharia at Stanford in 2020 (SIGIR'20). The key insight: you don't need full cross-attention between query and document to get accurate retrieval. You can encode them independently and defer the interaction to a cheap MaxSim operation.

### 3.1 Architecture Diagram

```
                    Query Encoder (BERT)        Document Encoder (BERT)
                    ┌─────────────────┐         ┌──────────────────────┐
  "how to reduce    │                 │         │  "Flash attention    │
   memory during    │  [CLS] q1 q2 q3 │         │   reduces memory...  │
   inference"       │  q4 q5 [SEP]    │         │   [CLS] d1 d2 d3 ... │
                    │                 │         │   d200 [SEP]         │
                    └────────┬────────┘         └──────────┬───────────┘
                             │                             │
                    ┌────────▼────────┐         ┌──────────▼───────────┐
                    │  Linear projection│        │  Linear projection    │
                    │   BERT (768d) →   │        │   BERT (768d) →       │
                    │   ColBERT (128d)  │        │   ColBERT (128d)      │
                    └────────┬────────┘         └──────────┬───────────┘
                             │                             │
                             │          ┌──────────┐       │
                             │          │  MaxSim  │       │
                             └─────────►│  Σ max   │◄──────┘
                                        │  cosine  │
                                        └────┬─────┘
                                             │
                                             ▼
                                      relevance score
```

### 3.2 Key Design Decisions

**1. Independent encoding.** Query and document are encoded by separate BERT runs. This means document embeddings can be pre-computed and indexed offline — same advantage as a bi-encoder. At query time, only the query encoder runs.

**2. Token-level representations.** Instead of pooling the [CLS] token (bi-encoder) or doing full cross-attention (cross-encoder), ColBERT keeps the full sequence of token embeddings. For a document with 200 tokens and 128-dimensional embeddings, the document is represented as a 200×128 matrix.

**3. Dimensionality reduction.** BERT produces 768-dimensional hidden states. ColBERT projects these down to 128 dimensions via a single linear layer. This reduces the per-token storage by 6x with minimal accuracy loss.

**4. Query augmentation.** ColBERT appends a special [MASK] token(s) to the query before encoding. This increases the number of query token embeddings, giving the model more capacity to express multi-aspect queries. The [MASK] tokens attend to the full query during BERT encoding and can learn to represent missing or implied concepts.

**5. Late interaction (MaxSim).** Scoring is deferred to a separate step after encoding. The MaxSim operator compares every query token against every document token and sums the maxima.

### 3.3 Training Objective

ColBERT is trained with pairwise (triplet) loss:

```
L = max(0, margin + score(Q, D_negative) - score(Q, D_positive))
```

Where positive documents are labeled relevant (from MS MARCO, etc.) and negatives are sampled from BM25 top results that aren't labeled relevant. The model learns to assign higher late-interaction scores to relevant documents.

The loss is computed over batches of in-batch negatives — non-relevant documents from other queries in the same batch are used as negatives, which dramatically improves training efficiency without explicit negative sampling.

---

## 4. MaxSim: The Core Scoring Mechanism

MaxSim is the beating heart of late-interaction retrieval. Understanding it is essential for debugging, tuning, and knowing when it will fail.

### 4.1 Formal Definition

Given a query encoded as a matrix Q ∈ ℝ^(m×d) and a document encoded as D ∈ ℝ^(n×d), where m is the number of query tokens, n is the number of document tokens, and d is the embedding dimension:

```
score(Q, D) = Σ_{i=1}^{m} max_{j=1}^{n} cos(Q_i, D_j)
```

Each query token independently finds the document token most similar to it. These max similarities are summed across query tokens. The result is a single scalar relevance score.

### 4.2 Intuition

Think of it as the query asking: "For each word or subword in my question, which part of the document is most relevant to that word, and how relevant is it?"

- Query token "memory" will match document tokens about "VRAM", "memory consumption", "memory bandwidth" — whichever is closest in embedding space
- Query token "inference" will independently match the part of the document discussing inference
- If the document mentions "inference" only once (even in a footnote), that single token gets the maximum for that query token

This is fundamentally different from a bi-encoder, which would average "memory" and "inference" into one query vector and compare it against one document vector that averages all document content together.

### 4.3 Implementation

```python
import torch
import torch.nn.functional as F

def maxsim_score(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """
    Compute ColBERT's MaxSim score between a query and a document.

    Args:
        query_emb: (m, d) — query token embeddings
        doc_emb:   (n, d) — document token embeddings

    Returns:
        scalar score
    """
    # Normalize for cosine similarity
    query_emb = F.normalize(query_emb, p=2, dim=1)   # (m, d)
    doc_emb   = F.normalize(doc_emb, p=2, dim=1)     # (n, d)

    # Similarity matrix: (m, n)
    sim = torch.mm(query_emb, doc_emb.t())

    # Max over document dimension, sum over query dimension
    max_per_query_token, _ = sim.max(dim=1)   # (m,)
    return max_per_query_token.sum()          # scalar
```

### 4.4 Batch MaxSim (Multiple Documents)

```python
def batch_maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    """
    Score one query against many documents.

    Args:
        query_emb: (m, d)
        doc_emb:   (B, n, d) — B documents, each with n token embeddings

    Returns:
        (B,) scores
    """
    query_emb = F.normalize(query_emb, p=2, dim=1)     # (m, d)
    doc_emb   = F.normalize(doc_emb, p=2, dim=-1)      # (B, n, d)

    # Batched similarity: (B, m, n)
    sim = torch.einsum('md,bnd->bmn', query_emb, doc_emb)

    # Max over document dimension -> (B, m)
    max_per_query, _ = sim.max(dim=2)

    # Sum over query dimension -> (B,)
    return max_per_query.sum(dim=1)
```

### 4.5 Computational Complexity

For a query with m tokens and a document with n tokens:

| Operation | Complexity |
|---|---|
| Query encoding (BERT) | O(m²) |
| Document encoding (BERT) | O(n²) — pre-computed offline |
| MaxSim scoring | O(m × n × d) |
| Cosine normalization | O(m × d + n × d) |

With default values (m=32, n=200, d=128), MaxSim is roughly 32×200×128 = 819,200 floating-point operations. A modern CPU does this in microseconds. For 1,000 candidate documents, total scoring is ~0.8 GFLOP — about 5-10ms on CPU.

### 4.6 When MaxSim Excels

- **Aspect retrieval:** The query "Python async error handling" — each token finds its best document match independently
- **Sparse signal:** A single relevant sentence in a 10-page document still gets matched by the relevant query tokens
- **Multi-faceted relevance:** Documents relevant for different reasons are distinguished by different token match patterns

### 4.7 When MaxSim Struggles

- **Negation:** "algorithms that do NOT use sorting" — the "NOT" token matches documents mentioning "not", but the semantic negation isn't captured at the token level. All tokens still match positively.
- **Structural reasoning:** "If A > B then output C" — the conditional relationship isn't explicitly modeled
- **Very short documents:** A 5-token document has very few candidate positions for query tokens to match, reducing the discriminative power

---

## 5. ColBERTv2: Residual Compression

ColBERTv2 (Santhanam, Khattab, Zaharia, TACL 2021) addressed the main practical limitation of the original ColBERT: index size. A 10M document corpus with 200 avg tokens × 128 float32 dimensions = ~1TB. That's too large for many production deployments.

### 5.1 The Residual Compression Mechanism

ColBERTv2 replaces raw token embeddings with a two-level representation:

1. **Centroids:** K cluster centroids learned from the embedding space (K=4096 typically)
2. **Residuals:** For each token embedding, store:
   - Centroid ID (the nearest centroid): 12 bits
   - Compressed residual vector (centroid - embedding difference): 2 bits per dimension

Total per token: 12 bits + 128 × 2 bits = 268 bits = ~34 bytes (vs 512 bytes for float32)

### 5.2 How It Works End-to-End

**Indexing:**
1. Run all document passages through ColBERT to get per-token embeddings
2. Learn K centroids via k-means on the embedding space
3. For each token: assign nearest centroid, compute residual (centroid - token_embedding), quantize residual to 2-bit range per dimension
4. Store: centroid_id + compressed_residual per token

**Scoring:**
1. Encode query to get per-token embeddings (no compression on query side)
2. For each document token, reconstruct: centroids[centroid_id] + dequantized_residual
3. Compute MaxSim between query tokens and reconstructed document tokens

The reconstruction is approximate, but the compression is remarkably effective. With 2-bit residuals, accuracy drops by less than 1 point on MS MARCO while reducing storage by ~15x.

### 5.3 Storage Comparison

| Method | Per token | Per 200-token document | 10M docs |
|---|---|---|---|
| ColBERT float32 | 512 B | 100 KB | ~1 TB |
| ColBERTv2 compressed | 34 B | 6.8 KB | ~68 GB |
| Bi-encoder float32 | 3 KB | 3 KB | ~30 GB |

ColBERTv2 brings index size to roughly 2x a bi-encoder index, which is manageable for most production setups.

---

## 6. PLAID: Performance-Optimized Late Interaction Driver

PLAID (Performance-optimized Late Interaction Driver) was introduced by Santhanam et al. at CIKM 2022. It addresses the query-time latency of late-interaction retrieval by introducing **centroid interaction** — a multi-stage pruning pipeline that avoids scoring every document in the corpus.

### 6.1 The Problem

Naively scoring all N documents in a corpus with MaxSim requires computing N × m × n dot products. Even with efficient batch processing, this is too slow for large corpora (10M+ documents).

### 6.2 The PLAID Pipeline

PLAID uses the centroid-residual representation (already computed for compression) as a multi-stage filter:

**Stage 1: Centroid Interaction (Coarse Filter)**
- Each document is represented by its N_c centroids (the centroid IDs assigned to its token embeddings, with duplicates removed)
- The query is also represented by its set of centroids
- Documents are ranked by how many centroids they share with the query — a bag-of-centroids overlap score
- This is extremely fast: a hash comparison, no floating point
- Top-K₁ candidates pass to Stage 2 (K₁ ≈ 10,000)

**Stage 2: Approximate MaxSim**
- For the top-K₁ candidates, compute an approximate MaxSim using only the centroid assignments
- Instead of reconstructing all token embeddings, score each token by looking up which centroid it belongs to and using the centroid embedding as a proxy for the token embedding
- This avoids residual reconstruction entirely
- Top-K₂ candidates pass to Stage 3 (K₂ ≈ 1,000)

**Stage 3: Full MaxSim with Reconstruction**
- For the top-K₂ candidates, fully reconstruct token embeddings (centroid + dequantized residual)
- Compute exact MaxSim
- Return top-K final results

### 6.3 Speedup

| Stage | Candidates | Cost per Document | Cumulative Cost |
|---|---|---|---|
| Stage 1 (centroid overlap) | 10M | 1 hash lookup | O(N) |
| Stage 2 (approx MaxSim) | 10K | Centroids only | O(K₁ × m × n_centroids) |
| Stage 3 (full MaxSim) | 1K | Full reconstruction + MaxSim | O(K₂ × m × n_full) |

PLAID achieves **~10x latency improvement** over naive late-interaction search while maintaining >95% recall of the full exact search.

### 6.4 NextPlaid

NextPlaid (LightOn, 2026) pushes this further with:
- Memory-mapped indexes for corpora that don't fit in RAM
- REST API server for remote indexing and querying
- Quantized centroid interaction for even faster Stage 1 filtering
- Multi-vector database with CRUD operations on the index

---

## 7. ColPali: Visual Document Retrieval with VLMs

ColPali (Faysse et al., ICLR 2025) extends the late-interaction paradigm from text to visual document understanding. Instead of encoding text tokens, it encodes image patch tokens from a Vision Language Model.

### 7.1 The Problem It Solves

Traditional document retrieval pipelines are fragile:

```
PDF → OCR/Tesseract → Layout detection → Text extraction → Chunking → Embedding
```

Each step can fail: bad OCR, lost layout information, misordered reading order, tables rendered as unreadable text, figures ignored entirely.

ColPali skips all of that: it indexes documents as images directly.

### 7.2 Architecture

```
Document Page (image)                     Query Text
          │                                    │
          ▼                                    ▼
  PaliGemma-3B Vision                      BERT Text
  Encoder (ViT)                            Encoder
          │                                    │
          ▼                                    ▼
  Patch embeddings — grid of              Query token
  128-dim vectors per patch               embeddings
          │                                    │
          └──────────► MaxSim ◄────────────────┘
                          │
                          ▼
                   relevance score
```

1. **Visual encoding:** A document page is fed as an image to PaliGemma-3B's Vision Transformer. The ViT outputs a grid of patch embeddings — each patch corresponds to a region of the page (roughly 16×16 pixels at the input resolution).

2. **Projection:** Each patch embedding is projected to the same 128-dimensional space used by ColBERT. The result is a matrix of patch-level embeddings for the page.

3. **Late interaction:** Query tokens (from a standard text encoder) are compared against the patch embeddings via MaxSim.

The model is trained on page-level relevance judgments: given a query text and a document page image, maximize the MaxSim score for relevant pages and minimize for non-relevant ones.

### 7.3 Advantages

- **No text extraction pipeline.** No OCR, no layout parsing, no table detection. The model sees the page as a human would.
- **Layout awareness.** Charts, tables, headers, footers, columns — the visual layout is preserved in the patch grid. A query about "the bar chart on page 3" matches the visual representation, not a garbled text extraction of the chart caption.
- **Multimodal queries.** A query can refer to visual elements: "Find the document with the red warning banner" — something impossible with text-only retrieval.
- **Robust to PDF variations.** Scanned documents, complex multi-column layouts, handwritten annotations — all treated as images.

### 7.4 Limitations

- **Page-level granularity.** ColPali indexes at the page level, not the paragraph level. For very long pages, you may not pinpoint the exact section.
- **Computational cost.** The Vision Transformer is heavier than a text BERT encoder. Indexing millions of pages requires GPU time.
- **No native text search.** Queries like "exact phrase match" don't work — the model operates on visual similarity, not lexical matching.

### 7.5 ColPali Variants

| Model | Base VLM | Patch Size | Embed Dim | Use Case |
|---|---|---|---|---|
| ColPali | PaliGemma-3B | 16×16 | 128 | General document retrieval |
| ColQwen2.5 | Qwen2.5-VL | 14×14 | 128 | Multilingual docs |
| ColSmolVLM | SmolVLM | 14×14 | 128 | Lightweight, edge deployment |
| ColInternVL | InternVL2 | 16×16 | 128 | Chinese & Asian docs |

---

## 8. Mixedbread AI Late Interaction Models

Mixedbread AI has emerged as a leading vendor for production-grade late-interaction models. Their approach combines ColBERT-style late interaction with large-scale pretraining across modalities.

### 8.1 Wholembed

Mixedbread's Wholembed architecture is a "unified multilingual late-interaction model designed for text, code, audio, and vision." It's built on three innovations:

**1. Multimodal ingestion stack.** A shared encoder can process text, code, AST-parsed source, audio spectrograms, and image patches. Each modality goes through a modality-specific frontend, then into a shared transformer backbone.

**2. Dynamic vector allocation.** Instead of a fixed number of vectors per document (one per token for ColBERT), Wholembed allocates vectors adaptively based on content complexity. A simple one-sentence document gets fewer vectors than a complex technical paper. This reduces storage and speeds up scoring while preserving accuracy.

**3. Two-stage retrieval engine.** First stage: coarse vector-level filtering with quantized representations. Second stage: full late-interaction scoring on the filtered candidate set.

### 8.2 Performance Claims

Mixedbread reports that their 17M parameter open-source ColBERT model outperforms 8B parameter embedding models on the LongEmbed benchmark, which measures long-context retrieval capability. If accurate, this demonstrates the efficiency advantage of late interaction: token-level matching preserves information that larger single-vector models compress away.

### 8.3 mxbai-embed-colbert

Mixedbread's `mxbai-embed-colbert` is their open-source ColBERT model, available on HuggingFace. It supports:

- Multilingual text (trained on 100+ languages)
- Code documents (with AST-aware encoding)
- 128-dimensional embeddings
- Standard ColBERT-compatible index format

---

## 9. LightOn / PyLate: Training and Inference Toolkit

PyLate, by LightOn AI, is the most complete open-source toolkit for working with late-interaction models. It wraps the ColBERT ecosystem in a clean Python API and adds significant optimizations.

### 9.1 Key Features

**Training:**
- Fine-tune any SentenceTransformer model to produce ColBERT-style token embeddings
- Multi-GPU training with data parallelism
- In-batch negative sampling and hard negative mining
- Support for cross-encoder distillation (training a ColBERT to match a cross-encoder's scores)

**Inference:**
- Fast PLAID index (FastPLAID — optimized implementation)
- Voyager backend for general-purpose ANN search (alternative to PLAID for non-ColBERT vectors)
- Memory-mapped indexes for large corpora
- REST API for remote indexing and querying

**Models:**
- Pre-trained checkpoints for English and multilingual ColBERT
- Support for loading any SentenceTransformer as a base model

### 9.2 PyLate Architecture

```
PyLate
├── Modeling
│   ├── ColBERTModel        (wraps any SentenceTransformer)
│   ├── ColBERTConfig        (dim, compression, training params)
│   └── LateInteractionHead  (linear projection BERT→ColBERT dim)
├── Training
│   ├── ColBERTTrainer      (wraps SentenceTransformer Trainer)
│   ├── losses              (triplet, in-batch negative, distillation)
│   └── data                (MS MARCO, custom datasets)
└── Indexing & Retrieval
    ├── PLAIDIndex           (centroid interaction + MaxSim)
    ├── FastPLAIDIndex       (optimized PLAID)
    ├── VoyagerIndex         (ANN for general vectors)
    └── Reranker              (ColBERT as reranker over candidates)
```

### 9.3 FastPLAID

LightOn's FastPLAID improves on the original PLAID with:

- **SIMD-optimized centroid interaction** using AVX-512 instructions for batch centroid comparison
- **Multi-threaded Stage 2/3 scoring** with thread pool parallelism
- **Prefetching** for memory-mapped centroid and residual arrays
- **Adaptive candidate pruning** that adjusts K₁ and K₂ based on query difficulty

Benchmarks show FastPLAID achieving 3-5x speedup over the original PLAID implementation on the same hardware.

---

## 10. Jina AI: Jina-ColBERT-v2

Jina AI released `jina-colbert-v2`, a general-purpose multilingual late-interaction retriever based on the ColBERT architecture.

### 10.1 Key Specifications

| Property | Value |
|---|---|
| Context length | 8,192 tokens |
| Embedding dimension | 128 |
| Languages | 89 (trained on multilingual corpus) |
| Base model | JinaBERT (Jina's optimized BERT variant) |
| Training data | 200M+ query-document pairs |
| Index type | ColBERT-compatible (residual compression) |

### 10.2 Unique Features

**Long-context support.** With 8,192 token input, Jina-ColBERT-v2 can encode an entire research paper in a single pass without chunking. Standard ColBERTv2 has a 512-token input limit. For long-document retrieval, this is a significant advantage — chunking breaks cross-chunk context and increases index size proportionally.

**Explainability.** Because late interaction preserves token-level matching, Jina-ColBERT-v2 can output which query tokens matched which document tokens. This provides built-in explainability for retrieval results:

```python
# Jina's explain method
results = model.search(query, documents, return_explain=True)
for token, matched_terms in results.explanations[0].items():
    print(f"'{token}' matched: {matched_terms}")
```

**ColBERT-X.** Jina's variant adds a cross-encoder distillation head that can be used for reranking without loading a separate model. The same weights serve both as a late-interaction retriever and a lightweight reranker.

### 10.3 Integration

```python
from sentence_transformers import SentenceTransformer

# Jina-ColBERT-v2 is a standard SentenceTransformer model
model = SentenceTransformer("jinaai/jina-colbert-v2")

# Encode documents to get per-token embeddings
doc_embeddings = model.encode(
    documents,
    output_value="token_embeddings",  # Returns (n_docs, n_tokens, 128)
    convert_to_tensor=True
)
```

---

## 11. Practical: Building a Late-Interaction RAG Pipeline with RAGatouille

RAGatouille is the most accessible way to add ColBERT to a RAG pipeline. It wraps the ColBERT codebase in a simple API.

### 11.1 Installation

```bash
pip install ragatouille
```

### 11.2 Basic Indexing and Retrieval

```python
from ragatouille import RAGPretrainedModel
import json

# Sample corpus — academic paper excerpts about ML optimization
corpus = [
    {
        "id": "1",
        "title": "Gradient Checkpointing",
        "content": "Gradient checkpointing trades compute for memory by recomputing activations during backpropagation instead of storing them. It reduces memory from O(L) to O(sqrt(L)) where L is the number of layers, at the cost of approximately 33% more computation.",
    },
    {
        "id": "2",
        "title": "Flash Attention",
        "content": "Flash Attention computes exact attention in O(N^2) time but O(N) memory by tiling the attention computation and avoiding materialisation of the full NxN attention matrix in HBM. It achieves 2-4x wall-clock speedup over standard attention.",
    },
    {
        "id": "3",
        "title": "Speculative Decoding",
        "content": "Speculative decoding uses a smaller draft model to propose multiple candidate tokens, which are then verified in parallel by the large target model. This achieves 2-3x speedup in inference latency without any loss in output quality.",
    },
    {
        "id": "4",
        "title": "KV Cache Management",
        "content": "The KV cache stores key-value tensors from previous autoregressive steps to avoid recomputation. For a model with N layers, H hidden dimension, and L generated tokens, the cache requires O(N * H * L) memory per sequence.",
    },
    {
        "id": "5",
        "title": "Quantization for Inference",
        "content": "Post-training quantization reduces model precision from FP16 to INT4 or INT8, shrinking memory footprint by 4x and 2x respectively. INT4 weight-only quantization achieves negligible accuracy loss on most benchmarks while enabling larger batch sizes.",
    },
    {
        "id": "6",
        "title": "PagedAttention and vLLM",
        "content": "PagedAttention manages the KV cache in fixed-size blocks (pages), eliminating fragmentation and enabling memory sharing across sequences. This is the core technique behind the vLLM serving system, achieving near-optimal memory utilisation.",
    },
    {
        "id": "7",
        "title": "Continuous Batching",
        "content": "Continuous batching processes requests as they arrive rather than waiting for fixed-size batches. The scheduler adds new sequences to the running batch and evicts finished ones at each iteration, improving throughput by 10-20x over static batching.",
    },
    {
        "id": "8",
        "title": "LoRA Fine-Tuning",
        "content": "Low-Rank Adaptation (LoRA) freezes the pretrained weights and injects trainable rank decomposition matrices into attention layers. With rank r=8, this reduces trainable parameters from billions to millions while matching full fine-tuning quality.",
    },
]

# Load ColBERTv2 model
print("Loading ColBERTv2...")
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents
print("Indexing documents...")
document_texts = [doc["content"] for doc in corpus]
document_ids = [doc["id"] for doc in corpus]
document_titles = [doc["title"] for doc in corpus]

RAG.index(
    collection=document_texts,
    index_name="ml_concepts",
    max_document_length=256,
    split_documents=False,
    document_ids=document_ids,
    document_metadatas=[{"title": title} for title in document_titles],
    overwrite_index=True,
)

# Search
queries = [
    "how to reduce GPU memory during inference",
    "techniques for faster transformer decoding",
    "memory efficient attention mechanisms",
]

for query in queries:
    print(f"\n--- Query: '{query}' ---")
    results = RAG.search(query=query, k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:100]}...")
```

### 11.3 Reranker Pipeline (BM25 + ColBERT)

```python
from ragatouille import RAGPretrainedModel
import bm25s
import Stemmer

# First-stage retrieval with BM25
stemmer = Stemmer.Stemmer("english")
bm25_index = bm25s.BM25()
tokenized_corpus = bm25s.tokenize(document_texts, stemmer=stemmer)
bm25_index.index(tokenized_corpus)

query = "reducing transformer memory during inference"
tokenized_query = bm25s.tokenize(query, stemmer=stemmer)

# BM25 retrieves top-20 candidates
bm25_results, bm25_scores = bm25_index.retrieve(tokenized_query, k=20)
candidate_texts = [document_texts[idx] for idx in bm25_results[0]]

# ColBERT reranks
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
reranked = RAG.rerank(query=query, documents=candidate_texts, k=5)

for r in reranked:
    print(f"[{r['score']:.4f}] {r['content'][:80]}...")
```

### 11.4 Export and Reload Index

```python
# Save index for later reuse
RAG.index(
    collection=document_texts,
    index_name="ml_concepts",
    overwrite_index=True,
)

# Reload from disk
RAG_loaded = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/ml_concepts")

# Query the reloaded index
results = RAG_loaded.search(query="KV cache optimization", k=5)
```

---

## 12. Practical: Custom ColBERT Training with PyLate

When off-the-shelf ColBERT models don't work well for your domain, PyLate lets you fine-tune a custom model.

### 12.1 Installation

```bash
pip install pylate
```

### 12.2 Training Data Preparation

```python
from pylate import datasets
from datasets import Dataset

# Minimal training example: query, positive document, negative document
train_data = {
    "query": [
        "What is gradient checkpointing?",
        "How does Flash Attention work?",
        "What is speculative decoding?",
    ],
    "positive": [
        "Gradient checkpointing trades compute for memory by recomputing activations...",
        "Flash Attention computes exact attention in O(N^2) time but O(N) memory...",
        "Speculative decoding uses a small draft model to propose tokens...",
    ],
    "negative": [
        "Continuous batching processes requests as they arrive rather than waiting...",
        "Post-training quantization reduces model precision from FP16 to INT4...",
        "PagedAttention manages the KV cache in fixed-size blocks...",
    ],
}

dataset = Dataset.from_dict(train_data)

# For real use, load MS MARCO or your own triplet data
# from pylate.datasets import load_msmarco
# dataset = load_msmarco()
```

### 12.3 Fine-Tuning

```python
from pylate import models, losses, training, indexes

# Load base model
model = models.ColBERT(
    model_name_or_path="BAAI/bge-base-en-v1.5",
    dim=128,  # ColBERT embedding dimension
)

# Configure training
trainer = training.ColBERTTrainer(
    model=model,
    train_dataset=dataset,
    loss_fn=losses.InBatchNegativeLoss(model=model),
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    output_dir="./colbert-finetuned",
)

# Train
trainer.train()

# Save
model.save_pretrained("./colbert-finetuned")
```

### 12.4 Indexing and Retrieval

```python
from pylate import indexes

# Build PLAID index
plaid_index = indexes.PLAIDIndex(
    index_root="./indexes",
    index_name="my_corpus",
)

# Index documents
plaid_index.index(
    documents=document_texts,
    model=model,
    batch_size=32,
)

# Search
results = plaid_index.search(
    queries=["what reduces memory during inference"],
    model=model,
    k=5,
)

print(results)
```

### 12.5 Cross-Encoder Distillation

For higher quality, you can distill a cross-encoder into a ColBERT model:

```python
from pylate import losses

# Teacher cross-encoder produces soft labels
# ColBERT is trained to match teacher scores

distill_trainer = training.ColBERTTrainer(
    model=model,
    train_dataset=dataset_with_teacher_scores,
    loss_fn=losses.DistillationLoss(
        model=model,
        temperature=3.0,
        alpha=0.5,  # blend of hard and soft labels
    ),
    batch_size=16,
)
```

---

## 13. Practical: ColPali for PDF and Image Retrieval

For visual document retrieval with ColPali.

### 13.1 Installation

```bash
pip install colpali-rag
# Or from source:
# pip install git+https://github.com/illuin-tech/colpali.git
```

### 13.2 Basic PDF Retrieval

```python
from colpali_rag import ColPaliModel, ColPaliIndex
from PIL import Image
import pdf2image
from pathlib import Path

# Load the model
model = ColPaliModel.from_pretrained("vidore/colpali-v1.2")

# Convert PDF to page images
def pdf_to_pages(pdf_path: str) -> list[Image.Image]:
    return pdf2image.convert_from_path(pdf_path, dpi=150)

# Index a PDF
index = ColPaliIndex()

# Example: index a research paper
pdf_path = "attention_is_all_you_need.pdf"
pages = pdf_to_pages(pdf_path)

print(f"Indexing {len(pages)} pages...")
index.add_documents(
    documents=[
        {
            "page_id": f"transformer_paper_p{i}",
            "image": page,
            "metadata": {"source": pdf_path, "page_number": i},
        }
        for i, page in enumerate(pages)
    ],
    model=model,
    batch_size=4,
)

# Save index
index.save("colpali_transformer_index")

# Query
results = index.search(
    query="Transformer architecture diagram showing encoder-decoder structure",
    model=model,
    k=3,
)

for r in results:
    print(f"Page {r['metadata']['page_number']}: score={r['score']:.4f}")
```

### 13.3 Hybrid Text + Visual Retrieval

```python
# ColPali and ColBERT scores can be combined
from colpali_rag import ColPaliModel
from ragatouille import RAGPretrainedModel

def hybrid_search(query: str, k: int = 5, alpha: float = 0.5):
    """
    alpha=0: pure visual (ColPali), alpha=1: pure text (ColBERT)
    """

    # ColPali visual score
    colpali_results = colpali_index.search(query=query, model=colpali_model, k=50)
    colpali_scores = {r["page_id"]: r["score"] for r in colpali_results}
    colpali_max = max(colpali_scores.values()) or 1.0

    # ColBERT text score (rerank the same pages)
    page_texts = [r["text"] for r in colpali_results]
    colbert_results = rag_model.rerank(query=query, documents=page_texts, k=50)
    colbert_scores = {colpali_results[i]["page_id"]: r["score"]
                      for i, r in enumerate(colbert_results)}
    colbert_max = max(colbert_scores.values()) or 1.0

    # Combine scores
    combined = {}
    for page_id in set(list(colpali_scores.keys()) + list(colbert_scores.keys())):
        cs = colpali_scores.get(page_id, 0.0) / colpali_max
        bs = colbert_scores.get(page_id, 0.0) / colbert_max
        combined[page_id] = alpha * bs + (1 - alpha) * cs

    # Return top-k
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return sorted_results
```

---

## 14. When Dense Vectors Break

Understanding the failure modes of single-vector dense retrieval is essential for knowing when to reach for late interaction.

### 14.1 Pooling Information Loss

Single-vector models use a pooling operation (mean, CLS, or max) to collapse token embeddings into one vector.

**Mean pooling** averages all token representations. If a 500-token document has 485 tokens about training and 15 tokens about inference, the mean vector is dominated by training content. A query about "inference memory" will not match well even though the document contains the exact information.

**CLS pooling** uses the [CLS] token's representation. This token attends to the entire sequence during BERT encoding, but the representation must compress everything into one 768-dimensional vector. Multi-aspect information competes for the same representational capacity.

**Max pooling** takes the element-wise maximum across token dimensions. This preserves strong signals but loses frequency information — a concept mentioned 10 times doesn't score higher than a concept mentioned once.

### 14.2 Long-Context Failure

Most embedding models have a 512-token input limit. For documents longer than this:

- **Chunking:** The document must be split into chunks. A relevant sentence may end up in a chunk with little other relevant context, producing a weak embedding that doesn't match the query well.
- **Cross-chunk dependencies:** Information that spans chunk boundaries (e.g., "As discussed in Section 3..." with Section 3 in a different chunk) is lost.

Late interaction with long-context models (Jina-ColBERT-v2 at 8192 tokens) avoids chunking entirely.

### 14.3 Out-of-Distribution Degradation

Dense embeddings are trained on specific data distributions (MS MARCO, NQ, etc.). When deployed on out-of-domain data (legal documents, medical records, scientific papers), the quality degrades unpredictably.

Mixedbread's finding that a 17M parameter ColBERT matches 8B parameter embeddings on LongEmbed suggests late interaction is more robust to distribution shift because token-level matching is more transferable than compressed representations.

### 14.4 Specific Failure Examples

| Query | Dense Vector | Late Interaction |
|---|---|---|
| "papers about attention NOT Transformers" | Retrieves Transformer papers (dominant association) | More likely to find papers about attention in other contexts (each token matched independently) |
| "safety of autonomous vehicles in snow" | May match either "safety" or "autonomous vehicles" or "snow" — whichever dominates the embedding | Matches all three aspects independently through different query tokens |
| "which optimizer works best for GAN training?" | Biased toward popular optimizers (Adam) because they dominate training data | Each query token finds its match: "optimizer" → optimizer discussions, "GAN" → GAN papers, "training" → training sections |

---

## 15. Production Deployment Considerations

### 15.1 Index Size Management

| Corpus Size | Bi-Encoder | ColBERTv2 (compressed) | Raw ColBERT |
|---|---|---|---|
| 100K docs | ~300 MB | ~680 MB | ~10 GB |
| 1M docs | ~3 GB | ~6.8 GB | ~100 GB |
| 10M docs | ~30 GB | ~68 GB | ~1 TB |
| 100M docs | ~300 GB | ~680 GB | ~10 TB |

For large corpora, ColBERTv2 with residual compression is the only practical option. For small corpora (<100K docs), raw ColBERT is fine.

### 15.2 Query Latency Targets

| Pipeline | p50 Latency | p99 Latency | Throughput |
|---|---|---|---|
| Bi-encoder only | 5ms | 15ms | 500+ QPS |
| Bi-encoder + ColBERT rerank | 10ms | 30ms | 200 QPS |
| ColBERT full search (PLAID) | 50ms | 150ms | 60 QPS |
| ColPali search | 100ms | 300ms | 20 QPS |

### 15.3 Serving Architecture

```python
# Example: FastAPI endpoint with ColBERT
from fastapi import FastAPI
from ragatouille import RAGPretrainedModel
from pydantic import BaseModel

app = FastAPI()

# Load model at startup
model = RAGPretrainedModel.from_index("/indexes/ml_concepts")

class SearchRequest(BaseModel):
    query: str
    k: int = 10

@app.post("/search")
async def search(request: SearchRequest):
    results = model.search(
        query=request.query,
        k=request.k,
    )
    return {
        "query": request.query,
        "results": [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ],
    }
```

### 15.4 Memory Considerations

- **RAM:** PLAID index needs to hold centroids (typically 4096 × 128 floatoat32 = 2MB) and residuals (~34 bytes per token). For 10M docs × 200 avg tokens: 68 GB RAM.
- **Memory mapping:** NextPlaid and PyLate support memory-mapped indexes, so the index can be larger than available RAM. Pages are loaded on demand from SSD.
- **GPU memory:** For query encoding, a single BERT-base model needs ~1.5 GB GPU memory. For indexing large corpora, batch processing with gradient checkpointing reduces GPU requirements.

---

## 16. Benchmarking and Evaluation

### 16.1 Standard Benchmarks

| Benchmark | Task | Metric | Best Model (2026) |
|---|---|---|---|
| MS MARCO Passage | Passage ranking | MRR@10 | ColBERTv2 (39.7) |
| BEIR | Zero-shot retrieval | nDCG@10 | mxbai-colbert (avg 58.3) |
| LongEmbed | Long-context retrieval | Recall@100 | Jina-ColBERT-v2 (94.2) |
| ViDoRE (ColPali) | Visual document retrieval | Recall@5 | ColPali-v1.2 (91.5) |
| MTEB | Multi-task embedding | Average score | Proprietary models higher |

### 16.2 Evaluating Your Own Pipeline

```python
from pylate import evaluation
from beir import util, dataloaders

# BEIR evaluation for your trained model
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Load BEIR dataset
data_path = "datasets/nfcorpus"
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load()

# Your ColBERT search function
def colbert_search(query, corpus, k=100):
    pass  # Implement with your model + index

# Compute metrics
metrics = EvaluateRetrieval().evaluate(
    qrels=qrels,
    results=results_dict,
    k_values=[1, 3, 5, 10, 100],
)
print(metrics)
```

---

## 17. Decision Guide: Which Approach When

### 17.1 Selection Matrix

| Scenario | Recommended Approach | Why |
|---|---|---|
| General web search, short queries | Bi-encoder (openai-ada-002, etc.) | Fastest, good enough for top-10 |
| RAG over long documents | ColBERTv2 + PLAID | Token-level matching finds sparse signal |
| PDF/invoice/table-heavy docs | ColPali | Skips fragile text extraction pipeline |
| Enterprise search across diverse content | Hybrid: BM25 + ColBERT rerank | Best recall, moderate latency |
| Real-time search (<20ms latency) | Bi-encoder + lightweight reranker | ColBERT too slow for <20ms |
| Multilingual document retrieval | Jina-ColBERT-v2 or mxbai-embed-colbert | Trained on 89+ languages |
| Custom domain (legal, medical, code) | PyLate fine-tuned ColBERT | Distillation from cross-encoder best |
| Code retrieval | Mixedbread Wholembed | AST-aware encoding captures code structure |
| Budget-constrained (<50K docs) | Raw ColBERT (no compression needed) | Simpler, faster indexing |

### 17.2 Pipeline Recipes

**Recipe 1: Simple RAG (good enough)**
```
Query → Bi-encoder → top-20 → LLM context
```
Latency: ~10ms. Cost: low.

**Recipe 2: High-quality text RAG**
```
Query → Bi-encoder → top-50 → ColBERT rerank → top-10 → LLM context
```
Latency: ~20ms. Cost: low (reranking small set).

**Recipe 3: Maximum recall RAG**
```
Query → BM25 → top-100
      → Bi-encoder → top-100
      → Combine & deduplicate → top-100 → ColBERT rerank → top-15 → LLM context
```
Latency: ~50ms. Cost: medium.

**Recipe 4: Visual document RAG (PDFs, scanned docs)**
```
Query → ColPali → top-20 pages
      → OCR text from pages → top-20 chunks → ColBERT rerank → top-10 → LLM context
```
Latency: ~150ms. Cost: high.

**Recipe 5: Production hybrid (balanced)**
```
Query → ColBERT (PLAID Stage 1+2) → ~200 candidates
      → ColBERT (PLAID Stage 3, full MaxSim) → top-20
      → Cross-encoder rerank → top-5 → LLM context
```
Latency: ~75ms. Cost: medium-high.

---

## 18. Ecosystem Map and Vendor Comparison

| Vendor/Tool | ColBERT | ColPali | PLAID | Training | Reranking | Open Source | API |
|---|---|---|---|---|---|---|---|
| **RAGatouille** | ✓ | ✗ | ✓ (wraps ColBERT) | ✗ | ✓ | ✓ | ✗ |
| **PyLate (LightOn)** | ✓ | ✗ | ✓ (FastPLAID) | ✓ | ✓ | ✓ | ✓ (REST) |
| **Mixedbread** | ✓ | ✗ | ✓ | Proprietary | ✓ | ✓ (model) | ✓ |
| **Jina AI** | ✓ (v2) | ✗ | ✓ | Proprietary | ✓ | ✓ (model) | ✓ |
| **ColPali** | ✗ | ✓ | ✗ | Proprietary | ✓ | ✓ | ✗ |
| **NextPlaid** | ✓ | ✗ | ✓ (enhanced) | ✗ | ✗ | ✓ (core) | ✓ (REST) |
| **Weaviate** | ✓ (module) | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| **Qdrant** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |

### 18.1 Quick Start by Tool

```bash
# RAGatouille (easiest to start)
pip install ragatouille

# PyLate (most flexible)
pip install pylate

# Jina-ColBERT-v2
pip install sentence-transformers

# ColPali
pip install colpali-rag
```

---

## 19. Future Directions

### 19.1 Unified Embedding Models

The trend is toward models that produce multiple embedding types from a single forward pass: a dense vector, a sparse (lexical) vector, and token-level ColBERT embeddings. Mixedbread's Wholembed and Jina's ColBERT-X both support this pattern. This allows flexible deployment — use the dense vector for fast first-pass retrieval, then ColBERT for reranking — from the same model.

### 19.2 Agentic Retrieval

As LLM agents become more common, retrieval is moving from single-shot to multi-turn interactive patterns. Future late-interaction systems may:

- Accept feedback ("too broad, focus on the specific mechanism")
- Multi-query: expand a single user question into multiple search queries, retrieve for each, and aggregate scores
- Learn from implicit feedback: which results the agent actually uses in its reasoning

### 19.3 On-Device Late Interaction

With model compression (quantization, pruning), late-interaction models are approaching the size where they can run on-device. A 128-dimensional ColBERT with 8-bit quantization has minimal memory footprint. Combined with memory-mapped indexes, this makes local-first RAG agents viable.

### 19.4 Multimodal Late Interaction

ColPali demonstrated late interaction for vision. Extensions to audio (speech-to-text retrieval, music similarity search) and video (keyframe-level matching) are active research areas. Mixedbread's Wholembed already supports text + code + vision + audio in a single unified architecture.

---

## 20. Appendix: Full Code Reference

### 20.1 Complete RAGatouille Pipeline

```python
#!/usr/bin/env python3
"""Complete late-interaction RAG pipeline with RAGatouille."""

from ragatouille import RAGPretrainedModel
import json
import time
from typing import List, Dict


def build_index(
    documents: List[str],
    index_name: str = "my_index",
    model_name: str = "colbert-ir/colbertv2.0",
) -> RAGPretrainedModel:
    """Build a ColBERT index from a list of documents."""
    RAG = RAGPretrainedModel.from_pretrained(model_name)
    RAG.index(
        collection=documents,
        index_name=index_name,
        max_document_length=512,
        split_documents=True,
        overwrite_index=True,
    )
    return RAG


def search(
    rag: RAGPretrainedModel,
    query: str,
    k: int = 10,
) -> List[Dict]:
    """Search the index."""
    return rag.search(query=query, k=k)


def rerank(
    rag: RAGPretrainedModel,
    query: str,
    candidates: List[str],
    k: int = 5,
) -> List[Dict]:
    """Rerank a candidate set."""
    return rag.rerank(query=query, documents=candidates, k=k)


if __name__ == "__main__":
    # Example documents
    docs = [
        "Gradient checkpointing trades compute for memory by recomputing...",
        "Flash Attention reduces memory from O(N²) to O(N)...",
        "Speculative decoding uses a draft model for 2-3x speedup...",
        "PagedAttention manages KV cache in fixed-size blocks...",
        "LoRA freezes pretrained weights and injects rank-decomposition matrices...",
        "Quantization reduces model precision from FP16 to INT4/INT8...",
    ]

    print("1. Building index...")
    start = time.time()
    rag_model = build_index(docs, index_name="quick_demo")
    print(f"   Indexed in {time.time() - start:.2f}s")

    print("\n2. Searching...")
    results = search(rag_model, "memory optimization for LLMs", k=3)
    for r in results:
        print(f"   [{r['score']:.4f}] {r['content'][:80]}...")

    print("\n3. Reranking...")
    candidates = docs[:4]
    reranked = rerank(rag_model, "faster inference without quality loss", candidates, k=2)
    for r in reranked:
        print(f"   [{r['score']:.4f}] {r['content'][:80]}...")
```

### 20.2 MaxSim from Scratch (NumPy)

```python
#!/usr/bin/env python3
"""MaxSim implementation from scratch for educational purposes."""

import numpy as np


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of A and every row of B."""
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)


def maxsim_score(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """
    Compute ColBERT-style MaxSim score.

    Args:
        query_emb: (n_query_tokens, embedding_dim)
        doc_emb:   (n_doc_tokens, embedding_dim)

    Returns:
        scalar relevance score
    """
    sim = cosine_similarity_matrix(query_emb, doc_emb)  # (n_query, n_doc)
    max_per_query = sim.max(axis=1)  # (n_query,)
    return max_per_query.sum()


def batch_maxsim(query_emb: np.ndarray, doc_embeddings: list) -> np.ndarray:
    """
    Score one query against multiple documents.

    Args:
        query_emb: (n_query_tokens, d)
        doc_embeddings: list of (n_doc_tokens_i, d) arrays

    Returns:
        (n_docs,) array of scores
    """
    scores = np.array([maxsim_score(query_emb, doc) for doc in doc_embeddings])
    return scores


# Demonstration
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate embeddings
    d = 128  # embedding dimension
    n_query = 8
    n_doc = 200

    # Random embeddings (in practice these come from a ColBERT model)
    query_emb = np.random.randn(n_query, d)
    doc_emb = np.random.randn(n_doc, d)

    score = maxsim_score(query_emb, doc_emb)
    print(f"Query tokens: {n_query}")
    print(f"Document tokens: {n_doc}")
    print(f"Embedding dimension: {d}")
    print(f"MaxSim score: {score:.4f}")

    # Demonstrate multi-aspect matching
    print("\n--- Multi-aspect matching demo ---")

    # A query with two aspects
    query_aspects = np.array([
        [1, 0, 0, 0],  # token 1: "memory"
        [0, 1, 0, 0],  # token 2: "optimization"
    ], dtype=float)

    # A document that talks about memory in one region and optimization elsewhere
    doc_multi = np.vstack([
        np.random.randn(50, 4) * 0.1,       # noise before
        np.array([[1, 0, 0, 0]]),            # "memory" match
        np.random.randn(50, 4) * 0.1,        # noise middle
        np.array([[0, 1, 0, 0]]),            # "optimization" match
        np.random.randn(50, 4) * 0.1,        # noise after
    ])

    query_aspects = query_aspects / np.linalg.norm(query_aspects, axis=1, keepdims=True)
    doc_multi = doc_multi / np.linalg.norm(doc_multi, axis=1, keepdims=True)

    score_multi = maxsim_score(query_aspects, doc_multi)
    print(f"Multi-aspect score: {score_multi:.4f}")
    print("Both 'memory' and 'optimization' find their matching evidence independently")
```

---

## References

- Khattab & Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020. [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- Santhanam et al. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." TACL 2021. [arXiv:2112.01488](https://arxiv.org/abs/2112.01488)
- Santhanam et al. "PLAID: An Efficient Engine for Late Interaction Retrieval." CIKM 2022. [arXiv:2205.09707](https://arxiv.org/abs/2205.09707)
- Faysse et al. "ColPali: Efficient Document Retrieval with Vision Language Models." ICLR 2025. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
- Jina AI. "Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever." [arXiv:2408.16672](https://arxiv.org/abs/2408.16672)
- LightOn. "PyLate: Flexible Training and Retrieval for Late Interaction Models." [arXiv:2508.03555](https://arxiv.org/abs/2508.03555)
- Mixedbread AI. "Inside Mixedbread: Multimodal Late-Interaction at Billion Scale." [blog](https://www.mixedbread.com/blog/multimodal-late-interaction-billion-scale)

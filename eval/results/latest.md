# Latest evaluation results

## Retrieval (run 20261101-165804-retrieval, split=all)

| Config | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| A0 Baseline: MiniLM, 400w, dense | 0.12 | 0.21 | 0.23 | 0.35 | 0.18 | 22 |
| A1 + parser v2 and chunking v2 (dense) | 0.15 | 0.29 | 0.35 | 0.53 | 0.25 | 23 |
| A2 BM25 only | 0.35 | 0.62 | 0.68 | 0.79 | 0.49 | 9 |
| A3 Hybrid (RRF) | 0.21 | 0.44 | 0.56 | 0.85 | 0.39 | 39 |
| A4 Hybrid + reranker (small) | 0.44 | 0.68 | 0.73 | 0.82 | 0.58 | 2183 |
| A5 Hybrid + reranker (base) | 0.65 | 0.79 | 0.85 | 0.91 | 0.73 | 11112 |

### Recall@5 by category

| Config | cross_quarter R@5 | exact_term R@5 | single_fact R@5 | thematic R@5 |
| --- | --- | --- | --- | --- |
| A0 Baseline: MiniLM, 400w, dense | 0.50 | 0.38 | 0.08 | 0.12 |
| A1 + parser v2 and chunking v2 (dense) | 0.50 | 0.12 | 0.25 | 0.62 |
| A2 BM25 only | 0.00 | 1.00 | 0.75 | 0.75 |
| A3 Hybrid (RRF) | 0.33 | 0.50 | 0.58 | 0.75 |
| A4 Hybrid + reranker (small) | 0.67 | 0.75 | 0.75 | 0.75 |
| A5 Hybrid + reranker (base) | 0.83 | 1.00 | 0.83 | 0.75 |

## Answer quality, core questions (run 20261102-135015-answers)

| System | Faithfulness | Correctness | Citation precision | Abstention | p50 latency |
| --- | --- | --- | --- | --- | --- |
| B0 Naive RAG (dense top-5) | 0.92 | 0.14 | 0.31 | 100% | 50.1s |
| B1 Hybrid + rerank RAG (base reranker) | 0.91 | 0.64 | 0.63 | 100% | 37.4s |
| B2 LangGraph agent (base reranker) | 1.00 | 0.41 | 0.66 | 100% | 49.0s |

### Agent subset (cross_quarter + price_linked)

| System | n | Task success | Tool selection | Avg steps | Tool-error rate |
| --- | --- | --- | --- | --- | --- |
| B0 Naive RAG (dense top-5) | 10 | 0% | 0% | 1.00 | 0% |
| B1 Hybrid + rerank RAG (base reranker) | 10 | 0% | 0% | 1.00 | 0% |
| B2 LangGraph agent (base reranker) | 10 | 40% | 100% | 2.10 | 0% |

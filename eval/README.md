# Evaluation

The point of this folder is to turn "I think it's better" into numbers you can defend.

## 1. Build the question set (this is the real work)

Aim for **40 to 60** questions in `eval/questions.jsonl` (format: `eval/questions.example.jsonl`).

| Category | Count (suggested) | Tests |
| --- | --- | --- |
| `single_fact` | ~12 | one number or fact in one passage |
| `exact_term` | ~8 | product names, tickers, jargon: where BM25 should beat dense |
| `thematic` | ~8 | paraphrased concepts: where dense should beat BM25 |
| `cross_quarter` | ~8 | needs several calls: agent subset |
| `price_linked` | ~4 | needs price data: agent subset |
| `unanswerable` | ~6 | answer is not in the corpus: tests abstention |
| `trap` | ~4 | fiscal vs calendar confusion, missing quarters |

Each answerable question carries **verbatim evidence spans**, not chunk ids, so the same set evaluates every
chunking strategy. Split roughly 60/40 into `"dev"` and `"test"`; tune on dev only.

`python -m eval.draft_questions --n 40` can propose candidates from random passages using your local LLM and auto-rejects
any whose span is not verbatim, but **you must read and fix every one**. Unreviewed model-written questions are not a benchmark.

```bash
python -m eval.verify_evidence      # every span must really occur in its transcript
```

## 2. Run

```bash
python scripts/run_pipeline.py --configs v1 v2     # indexes for A0 (v1) and A1..A5 (v2)
python -m eval.run_eval retrieval                  # A0..A5, writes eval/results/*.json and updates the README table
python -m eval.run_eval answers                    # B0..B2 (needs Ollama: generator + a different-family judge)
python -m eval.run_eval retrieval --rows A0 A3 --split test
```

Retrieval ladder: **A0** v1 baseline · **A1** parser+chunking v2 · **A2** BM25 · **A3** hybrid RRF · **A4** + small reranker ·
**A5** + base reranker. Answer rows: **B0** naive RAG · **B1** hybrid+rerank RAG · **B2** agent.

## 3. Metrics

* **Recall@k / MRR / coverage@k**: a chunk is relevant if it is in the evidence transcript and contains the normalized span
  (or >=80% of its tokens with every number intact). Reported with 95% bootstrap CIs.
* **Faithfulness**: judge extracts claims and checks each against the retrieved context only.
* **Correctness**: exact match on numbers, then a judge for the rest (0 / 0.5 / 1).
* **Citation precision**: fraction of cited chunks that contain evidence. **Abstention**: correct refusals on unanswerable questions.
* **Agent**: task success, tool-selection accuracy, steps, tool-error rate. Cost is 0 (local models).

## 4. Judge hygiene (free judge)

Judge and generator come from **different model families** by default (`JUDGE_MODEL=qwen2.5:7b` vs `llama3.2:3b`),
temperature 0, prompts versioned in `eval/judges.py`, every judgment cached in `eval/cache/`. Before quoting a number,
run `audit_sample` to export ~10 judged items, label them yourself, and report the agreement rate next to the metric.
Small local judges can be noisy, so treat judge-based numbers as indicative and retrieval metrics (no judge involved) as firm.

## 5. CI

`python -m eval.smoke` runs on invented fixtures with deterministic fake components and fails the build if hybrid
Recall@5 drops below `eval/baseline_smoke.json` minus 0.05. Update the baseline deliberately with `--update`.

# Evaluation

The point of this folder is to turn "I think it's better" into numbers you can defend.

## 1. Build the question set (this is the real work)

`eval/questions.jsonl` holds **48** questions (format: `eval/questions.example.jsonl`): 29 dev, 19 test.

| Category | Count | Tests |
| --- | --- | --- |
| `single_fact` | 12 | one number or fact in one passage |
| `exact_term` | 8 | product names, tickers, jargon: where BM25 should beat dense |
| `thematic` | 8 | paraphrased concepts: where dense should beat BM25 |
| `cross_quarter` | 6 | needs several calls: agent subset |
| `price_linked` | 4 | needs price data: agent subset |
| `unanswerable` | 6 | answer is not in the corpus: tests abstention |
| `trap` | 4 | fiscal vs calendar confusion, missing quarters |

Each answerable question carries **verbatim evidence spans**, not chunk ids, so the same set evaluates every
chunking strategy. Split roughly 60/40 into `"dev"` and `"test"`; tune on dev only. Every question names the company
and the call date in its text, because the retrieval eval runs **without metadata filters**.

**Gold-answer rule.** A gold answer contains only the facts the question asks for: no extra figures, no call dates and no
"Q2"/"Q4" labels. `numeric_exact_ok` demands every number in the gold (dates and quarter digits included), so extras make
correct answers score 0. This rule was applied once to all 48 questions, dev and test alike, before the final run.

`python -m eval.draft_questions --kind single_fact --n 24 --seed 1` (also `exact_term`, `thematic`) proposes candidates: the
**evidence span is picked in code** (a real sentence of the chunk, so it is verbatim by construction) and the local LLM only
writes the question and a short gold answer. Drafts are auto-rejected when the question does not name company and call date,
is vague, leaks the answer, or the gold has a number that is not in the span. Analyst turns, operator turns and boilerplate
are skipped. **You must still read and fix every candidate**; unreviewed model-written questions are not a benchmark.
`python -m eval.merge_candidates <file> --keep sf001 ...` moves approved ones into `questions.jsonl`; spans go to the
git-ignored `eval/evidence.private.jsonl`. `cross_quarter` and `price_linked` questions are written by hand (two spans from
two calls; gold numbers from the price tool).

```bash
python -m eval.verify_evidence      # every span must really occur in its transcript
```

## 2. Run

```bash
python scripts/run_pipeline.py --configs v1 v2     # indexes for A0 (v1) and A1..A5 (v2)
python -m eval.run_eval retrieval                  # A0..A5, writes eval/results/*.json and updates the README table
python -m eval.run_eval answers                    # B0..B2 (needs Ollama: generator + a different-family judge)
python -m eval.run_eval retrieval --rows A0 A3 --split test
python -m eval.run_eval answers --rows B2          # a partial rerun merges into the previous results (omit --limit)
python -m eval.run_eval tables                     # re-render the README tables from the latest results
```

Analysis tools (read the saved results, no model calls):

```bash
python -m eval.resplit               # correctness / faithfulness / citation precision by split, bootstrap CIs, paired differences
python -m eval.inspect_answers --row B1 --only-wrong   # question, gold and answer per failing item
python -m eval.recheck_abstention    # answers that look like a misspelled abstention but were counted as answered
python -m eval.audit --row B1 --n 10 --seed 7           # blind sample sheet; add --verdicts 1 0 ... to compare with the judge
```

Retrieval ladder: **A0** v1 baseline · **A1** parser+chunking v2 · **A2** BM25 · **A3** hybrid RRF · **A4** + small reranker ·
**A5** + base reranker. Answer rows: **B0** naive RAG (dense top-5) · **B1** hybrid + base-reranker RAG · **B2** agent (base reranker).

## 3. Metrics

* **Recall@k / MRR / coverage@k**: a chunk is relevant if it is in the evidence transcript and contains the normalized span
  (or >=80% of its tokens with every number intact). Reported with 95% bootstrap CIs.
* **Faithfulness**: judge extracts claims and checks each against the retrieved context only.
* **Correctness**: exact match on numbers, then a judge for the rest (0 / 0.5 / 1). An abstention on an answerable question
  scores 0.
* **Headline answer columns use the core questions only** (single_fact, exact_term, thematic). cross_quarter and price_linked
  need several calls or the price tool, which plain RAG cannot do, so they are reported in a separate agent-subset table.
* **Citation precision**: fraction of cited chunks that contain evidence. **Abstention**: correct refusals on the unanswerable and trap questions; the marker check accepts a misspelled
  `INSUFFICIENT_EVIDENCE`. The agent's citations are scored against the full chunk text.
* **Agent**: task success, tool-selection accuracy, steps, tool-error rate. Cost is 0 (local models).

## 4. Judge hygiene (free judge)

Judge and generator come from **different model families** by default (`JUDGE_MODEL=qwen2.5:3b` vs generator `mistral:latest`),
temperature 0, prompts versioned in `eval/judges.py`, every judgment cached in `eval/cache/`. Before quoting a number,
run `python -m eval.audit` (a blind sheet of random answered questions with the judge score hidden), label them yourself, and
report the agreement rate next to the metric. The audit run for this repo: 10/10 agreement for B1 and 8/10 for B2, both
disagreements judge false negatives on correct answers, so the judge is conservative. Small local judges can be noisy, so treat
judge-based numbers as indicative and retrieval metrics (no judge involved) as firm.

## 5. CI

`python -m eval.smoke` runs on invented fixtures with deterministic fake components and fails the build if hybrid
Recall@5 drops below `eval/baseline_smoke.json` minus 0.05. Update the baseline deliberately with `--update`.

# Earnings Call Intelligence Platform (v2)

Hybrid search, cited question answering and a tool-using agent over corporate earnings call transcripts, with an
evaluation harness that measures whether each upgrade actually helped.

**Entirely free to run.** Open-weight models on CPU, a local LLM through [Ollama](https://ollama.com), SQLite and
ChromaDB, price data from yfinance. No paid API is required anywhere, including the evaluation judge.

> Reports what was said on earnings calls. **Not investment advice.** The "tone signal" describes language used
> on a call; it is not a prediction of anything.

**Stack:** Python · FastAPI · LangGraph · ChromaDB · BM25 · sentence-transformers · FinBERT · Ollama · SQLite · Streamlit · Docker

---

## What's in v2

| Area | v1 | v2 |
| --- | --- | --- |
| Retrieval | dense only, 400-word chunks (many truncated by the 256-token embedder) | dense + BM25 → RRF fusion → cross-encoder rerank; 180-word turn-aware chunks with speaker/role/section and a context header; metadata filters |
| Parsing | regex that stripped speaker labels and boilerplate mid-sentence | structure-first turn parser, prepared-vs-Q&A split, whole-paragraph boilerplate removal, number normalization, header-vs-manifest validation |
| Summaries | first 1,200 words only (none of the Q&A), quotes never checked | map-reduce over the whole call; quotes kept only if verbatim; "tone" label, not a "signal" |
| Q&A | none | cited RAG answers with an abstention path |
| Agent | none | LangGraph: route → tool loop → synthesize → verify; 6 tools incl. price reaction; hard 6-step cap |
| Serving | Streamlit loads FinBERT and ChromaDB in-process | FastAPI (typed, rate-limited, SSE) + thin Streamlit client; FinBERT runs offline only |
| Data integrity | duplicate rows on re-run, stale vectors skipped | UNIQUE constraints + upserts, delete-then-insert re-indexing, config-hash-named vector collections |
| Quality evidence | none | question set + metrics + bootstrap CIs + ablation ladder + CI regression gate |

## Results

**Setup.** 40 call transcripts, 5 companies (AAPL, AMZN, GOOGL, MSFT, NVDA), 8 consecutive quarters each. 48 hand-reviewed
questions (29 dev / 19 test): 12 single_fact, 8 exact_term, 8 thematic, 6 cross_quarter, 4 price_linked, 6 unanswerable,
4 trap. Generator: `mistral:latest` (7B) through Ollama. Judge: `qwen2.5:3b` (a different model family).

The three tables below are **generated** by `python -m eval.run_eval` (retrieval run `20261101-165804`, answers run
`20261102-135015`); nothing in them is typed by hand. The test-split figures and the reading of the results in the text
come from the same runs via `python -m eval.resplit`.

### Retrieval ablation (Recall@k = did a top-k chunk contain the evidence span)

34 questions that carry evidence spans, all splits combined, **no metadata filters** (each question names the company and
call date in its text). 95% bootstrap CIs are stored in the results JSON (`ci95_recall@5`).

<!-- RETRIEVAL_TABLE_START -->
| Config | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| A0 Baseline: MiniLM, 400w, dense | 0.12 | 0.21 | 0.23 | 0.35 | 0.18 | 22 |
| A1 + parser v2 and chunking v2 (dense) | 0.15 | 0.29 | 0.35 | 0.53 | 0.25 | 23 |
| A2 BM25 only | 0.35 | 0.62 | 0.68 | 0.79 | 0.49 | 9 |
| A3 Hybrid (RRF) | 0.21 | 0.44 | 0.56 | 0.85 | 0.39 | 39 |
| A4 Hybrid + reranker (small) | 0.44 | 0.68 | 0.73 | 0.82 | 0.58 | 2183 |
| A5 Hybrid + reranker (base) | 0.65 | 0.79 | 0.85 | 0.91 | 0.73 | 11112 |
<!-- RETRIEVAL_TABLE_END -->

### Answer quality, core questions (single_fact + exact_term + thematic; n = 28, dev + test)

<!-- ANSWER_TABLE_START -->
| System | Faithfulness | Correctness | Citation precision | Abstention | p50 latency |
| --- | --- | --- | --- | --- | --- |
| B0 Naive RAG (dense top-5) | 0.92 | 0.14 | 0.31 | 100% | 50.1s |
| B1 Hybrid + rerank RAG (base reranker) | 0.91 | 0.64 | 0.63 | 100% | 37.4s |
| B2 LangGraph agent (base reranker) | 1.00 | 0.41 | 0.66 | 100% | 49.0s |
<!-- ANSWER_TABLE_END -->

### Agent subset (cross_quarter + price_linked, n = 10)

Plain RAG cannot call the price tool or combine two calls, so these questions are reported separately and are not in the
headline columns above.

<!-- AGENT_TABLE_START -->
| System | n | Task success | Tool selection | Avg steps | Tool-error rate |
| --- | --- | --- | --- | --- | --- |
| B0 Naive RAG (dense top-5) | 10 | 0% | 0% | 1.00 | 0% |
| B1 Hybrid + rerank RAG (base reranker) | 10 | 0% | 0% | 1.00 | 0% |
| B2 LangGraph agent (base reranker) | 10 | 40% | 100% | 2.10 | 0% |
<!-- AGENT_TABLE_END -->

### What the numbers support

* **Retrieval: the reranker is the step that matters.** Recall@5 goes 0.23 (A0) → 0.35 (A1) → 0.56 (hybrid, A3) → 0.73 (small
  reranker, A4) → 0.85 (base reranker, A5). BM25 alone (0.68) beats plain hybrid at R@5 but not at R@10 (0.79 vs 0.85): the
  right passages are in the candidate pool, the reranker fixes the order. A0 to A1 changes the parser and the chunking
  together, so the gain cannot be attributed to either. A5 is about 5x slower than A4 (p50 11 s vs 2 s on CPU).
* **Answers: better retrieval gives better answers.** Same generator, hybrid + base reranker (B1) vs dense top-5 (B0). Test
  split, core questions, n = 11, correctness (judge) with 95% bootstrap CI: **B0 0.09 [0.00, 0.27]**, **B1 0.68 [0.41, 0.91]**,
  **B2 (agent) 0.50 [0.23, 0.77]**. Paired B1 − B0: **+0.59 [0.32, 0.86]** on test and +0.50 [0.32, 0.68] on all 28.
* **The agent did not beat plain hybrid RAG on core questions** (B2 − B1: −0.18 [−0.45, 0.14] on test, −0.23 [−0.43, −0.04] on
  all 28), so say "no detectable gain, possibly lower". Its value is on cross_quarter and price_linked questions, which plain RAG
  cannot answer at all (see the agent-subset table).
* **Abstention** on the 10 unanswerable and trap questions was 10/10 for all three systems.

### How far to trust them

* **Small samples.** 11 core test questions, 6 cross_quarter, 4 price_linked. Intervals are wide; differences smaller than
  the interval are noise. Retrieval metrics involve no judge and are the firmest numbers here.
* **The judge is a 3B model.** A blind hand audit of 10 random answered core questions per system found the rater and the judge
  agreeing on **10/10 (B1)** and **8/10 (B2)**. Both disagreements were judge false negatives (correct answers scored 0), so
  judge-based correctness is conservative, more so for B2. The rater also wrote the questions.
* **Scoring was fixed after looking at results.** After a first partial run on the test questions I (1) audited all gold
  answers for extra figures, call dates and quarter labels the question did not ask for (12 changed), (2) moved cross_quarter
  and price_linked out of the headline columns, (3) made the agent's citation check use the full chunk text instead of a
  shortened snippet, and (4) made the abstention check accept a misspelled marker (one `INSUFFICENT_EVIDENCE` was being counted
  as an answer). No retrieval or model setting was tuned on the test split, but these scoring fixes were made after seeing
  part of it. B1 and B2 use the base reranker, chosen from the retrieval ablation.
* **"Verified against evidence" is a mechanical check** (numbers and quotes appear in retrieved text, dates match the cited call,
  periods are filtered). It does not prove an answer is relevant or correct.

Per-category tables are in `eval/results/latest.md`. Protocol and reproduction steps: [eval/README.md](eval/README.md).

---

## Quickstart (all free)

```bash
git clone < https://github.com/SarthakDaga/AI-Earnings-Call-Intelligence-v2.git > && cd AI-Earnings-Call-Intelligence-v2
python -m venv venv && source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**1. Add transcripts** to `data/transcripts/` named `TICKER_Q#_YYYY.txt`, and one row each to `data/manifest.csv`
(`ticker,fiscal_quarter,fiscal_year,call_date,source_url`). The call date powers the price-reaction tool.

```bash
python scripts/validate_corpus.py        # catches header/manifest mismatches, missing Q&A, truncated questions
```

**2. Process** (FinBERT sentiment + chunk + embed; 5-15 min on CPU for a handful of calls, then cached):

```bash
python scripts/run_pipeline.py --configs v1 v2     # v1 index is only needed to reproduce the A0 baseline
```

**3. Install a free local LLM** (for summaries, Q&A and the agent):

```bash
# https://ollama.com
ollama pull mistral:latest     # generator used for the reported results (7B; slow on CPU, set LLM_TIMEOUT=600)
ollama pull qwen2.5:3b         # judge for evals (different family from the generator on purpose)
# faster but weaker generator: ollama pull llama3.2:3b
```

**4. Run**

```bash
uvicorn src.api.main:app --port 7860      # API + docs at http://localhost:7860/docs
streamlit run app/streamlit_app.py        # UI at http://localhost:8501
```

Retrieval, search and sentiment work without any LLM. Only summaries, `/v1/ask` and the agent need one.

### Free LLM options

| `LLM_PROVIDER` | What | Cost |
| --- | --- | --- |
| `ollama` (default) | local models via Ollama | free, private |
| `openai_compat` | any OpenAI-compatible endpoint; several vendors (Groq, Google AI Studio, OpenRouter `:free` models) offer free tiers. Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL` | free tier only, with rate limits that change: check the vendor |

The same switch exists for the eval judge (`JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY`).
Everything is configured in [`src/config.py`](src/config.py) through environment variables.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | model/index readiness |
| `GET /v1/transcripts` | list calls |
| `GET /v1/transcripts/{id}/sentiment?section=prepared\|qa` | cached FinBERT results |
| `POST /v1/search` | `{query, k, mode: dense\|bm25\|hybrid\|hybrid_rerank, filters}` |
| `POST /v1/ask` | cited answer or `INSUFFICIENT_EVIDENCE` |
| `POST /v1/agent/chat` | SSE stream: `route`, `thought`, `tool_call`, `tool_result`, `verify`, `answer`, `done` |
| `GET/POST /v1/transcripts/{id}/summary` | fetch cached; POST generates in the background (needs `X-API-Key`) |
| `POST /v1/transcripts` | upload (disabled unless `ENABLE_UPLOADS=1` and `ADMIN_API_KEY` set) |

Per-IP rate limits (`RATE_LIMIT_PER_MIN`, LLM-backed routes get a third of the budget), request IDs, JSON logs,
one error shape: `{"error": {"code", "message", "request_id"}}`. Write endpoints are **off** unless `ADMIN_API_KEY` is set.

## The agent

```
route ─┬─ simple lookup ──▶ rag ───────────────┐
       └─ multi-step ─────▶ agent ⇄ tools ─────┴▶ synthesize ⇄ verify ─▶ answer
```

* **Routing** is a cheap heuristic, not an LLM call, so simple questions stay fast.
* **The loop uses JSON actions, not native tool-calling**, so it works with any small free local model.
* **Tools:** `list_transcripts`, `search_transcripts`, `get_sentiment`, `get_sentiment_trend`,
  `get_price_reaction`, `get_price_history`. Arguments are Pydantic-validated, calls time out after 10 s, and errors
  come back as observations the agent can react to.
* **Verification:** every number and quoted phrase in the answer must appear in tool output. One retry, then it abstains.
  This is deliberately strict: a number the model *computed* itself will be rejected.
* **Hard cap** of 6 tool calls (`AGENT_MAX_STEPS`); per-thread memory for follow-ups.
* **Price convention:** calls happen after market close, so the baseline is the last close on or before the call date and
  the reaction is the move over the next 1 and 5 trading days.

## Project layout

```
app/streamlit_app.py     thin UI over the API
src/config.py            all settings (env-driven); ChunkConfig + hash-versioned collections
src/ingestion/           loader (+manifest), turns.py (v2 parser), parser.py (v1 cleaner, kept for A0)
src/preprocessing/       tokenizer.py (sentences, v1 chunks), chunker.py (v2 turn-aware chunks)
src/retrieval/           bm25, dense (Chroma), fusion (RRF), reranker, retriever, index builder
src/sentiment/           FinBERT (lazy-loaded), section-aware aggregation
src/summarization/       map-reduce summarizer with verbatim-quote check
src/llm/                 provider interface: ollama | openai_compat | fake
src/qa/                  cited RAG answerer
src/agent/               LangGraph graph, tools, price provider
src/api/                 FastAPI app, routers, schemas, auth, rate limiting
eval/                    metrics, judges, run_eval, smoke gate, question tooling
scripts/                 run_pipeline, validate_corpus, check_truncation, download_models, fetch_data
tests/                   unit + integration tests; deterministic fakes, no model downloads
```

## Testing & CI

```bash
make install && make test      # no torch needed: tests use a hashing embedder and scripted fake LLM
make lint
make smoke                     # retrieval regression gate (also in CI)
```

The smoke gate runs on a tiny **invented** corpus (`tests/fixtures`), so every mode scores near 100% there. It protects
the retrieval *plumbing* from regressions; it says nothing about quality on real calls. Quality lives in the eval tables above.

## Data & licensing

Earnings call transcripts are **third-party copyrighted content.** This repo does not include any and the image does not
bake any in.

* `data/transcripts/*.txt`, the SQLite DB and the vector store are git-ignored. If an earlier version of this repo
  committed transcripts, purge them from history before publishing:
  `git filter-repo --path data/transcripts --invert-paths`.
* `data/manifest.csv` (ticker, period, call date, source URL) is metadata and is committed.
* Evidence spans in the eval set are short verbatim excerpts; keep them in `eval/evidence.private.jsonl` (git-ignored)
  if you prefer not to publish excerpts.
* For a public demo, host a private dataset on Hugging Face (free) and set `HF_DATASET_REPO` + `HF_TOKEN`;
  `scripts/fetch_data.py` pulls it at container start. Alternatively demo with transcripts you have the right to share.

## Deployment (free options)

* **Local / any VM:** `docker compose up --build` (add `--profile ollama` to run Ollama in a container).
* **Hugging Face Spaces (free CPU, Docker SDK):** set `MODE=all`, `PORT=7860`, secrets `HF_DATASET_REPO`, `HF_TOKEN`.
  Spaces cannot run Ollama, so either point `LLM_PROVIDER=openai_compat` at a free-tier endpoint, or run retrieval-only
  (search and sentiment work; summaries and the agent will report the LLM as unavailable). Free CPU Spaces sleep when
  idle, so expect a cold start. yfinance may be rate-limited from shared cloud IPs; the price tool falls back to its
  SQLite cache, then to an optional `data/prices_snapshot.csv` (`ticker,date,close`).

## Known limitations

* **Fiscal vs calendar quarters.** AAPL, MSFT and NVDA use fiscal quarters; AMZN and GOOGL use calendar quarters
  (`NVDA_Q1_2025` is the May 2024 call). Always compare by call date, never by "Q1" across companies.
* **Nine transcripts have no speaker labels** (only "Operator" is labelled): GOOGL Q3 2024, Q4 2024, Q1 2025, Q2 2025,
  Q3 2025, Q4 2025, Q1 2026 and NVDA Q1 2026, Q2 2026. They are kept (best copies available) and used for search, sentiment
  and summaries, but no who-said-what question was written on them.
* Speaker **roles are inferred**; transcript sources label speakers inconsistently. Treat `role` filters as best-effort.
  Some transcripts contain speech-to-text errors, and `NVDA_Q4_2026` has one cut-off line.
* **Small local models make mistakes.** They copy passages almost verbatim, can skip tool calls, and sometimes state things the
  context does not support. The plain-RAG rows do not filter by call date, so they sometimes answer from the wrong call
  (for example giving another quarter's figure). The agent route does infer ticker and fiscal period.
* The verifier checks dates, periods and numbers, not every claim, and its strictness trades recall for safety: legitimate
  derived numbers may be refused.
* **The evaluation set is small** (48 questions, 11 core test questions) and written by one person, so confidence intervals are
  wide. A free local judge is weaker than a frontier model: see the audit above and `eval/README.md`.
* The summarizer on a CPU model is slow (one call per ~700 words) and its tone label is coarse.
* The UI keeps only question and answer for past turns (trace and citations only for the live answer).
* yfinance is an unofficial scraper and can break or rate-limit without notice; price calls are cached in SQLite.
* The Docker image has not been built in CI against a registry; `docker build` runs in the CI workflow.

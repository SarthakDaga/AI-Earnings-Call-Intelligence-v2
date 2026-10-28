"""
eval/run_eval.py   -   one command reproduces every table in the README.

    python -m eval.run_eval retrieval              # rows A0..A5
    python -m eval.run_eval answers                # rows B0..B2 (needs Ollama or a free hosted LLM)
    python -m eval.run_eval tables                 # re-render README tables from latest results
    python -m eval.run_eval retrieval --rows A0 A3 --split dev

Results are written to eval/results/<run_id>.json. Retrieval rows need the
indexes built first:  python scripts/run_pipeline.py --configs v1 v2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics  # noqa: E402
from src import config  # noqa: E402

RESULTS = config.EVAL_DIR / "results"

RETRIEVAL_ROWS = {
    "A0": ("A0 Baseline: MiniLM, 400w, dense", dict(cfg="v1", mode="dense", reranker=None)),
    "A1": ("A1 + parser v2 and chunking v2 (dense)", dict(cfg="v2", mode="dense", reranker=None)),
    "A2": ("A2 BM25 only", dict(cfg="v2", mode="bm25", reranker=None)),
    "A3": ("A3 Hybrid (RRF)", dict(cfg="v2", mode="hybrid", reranker=None)),
    "A4": ("A4 Hybrid + reranker (small)", dict(cfg="v2", mode="hybrid_rerank", reranker=config.RERANKER_SMALL)),
    "A5": ("A5 Hybrid + reranker (base)", dict(cfg="v2", mode="hybrid_rerank", reranker=config.RERANKER_BASE)),
}
ANSWER_ROWS = {
    "B0": ("B0 Naive RAG (dense top-5)", dict(kind="rag", cfg="v1", mode="dense")),
    "B1": ("B1 Hybrid + rerank RAG", dict(kind="rag", cfg="v2", mode="hybrid_rerank")),
    "B2": ("B2 LangGraph agent", dict(kind="agent", cfg="v2", mode="hybrid_rerank")),
}

_embedders: dict = {}


def build_retriever(spec: dict):
    from src.database import queries
    from src.retrieval.embedder import Embedder
    from src.retrieval.index import build_index
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import Retriever
    cfg = config.get_chunk_config(spec["cfg"])
    emb = _embedders.setdefault(cfg.embed_model, Embedder(cfg.embed_model))
    ts = [queries.get_transcript_by_id(t["id"]) for t in queries.get_all_transcripts()]
    build_index(cfg, emb, ts, progress=lambda *_: None)  # no-op when already indexed
    rr = CrossEncoderReranker(spec["reranker"]) if spec.get("reranker") else None
    return Retriever(cfg, embedder=emb, reranker=rr).ensure_ready()


# ── retrieval ──
def eval_retrieval(questions: list[dict], retriever, mode: str) -> list[dict]:
    rows = []
    for q in questions:
        if not q["answerable"] or not q["evidence"]:
            continue
        hits, ms = retriever.timed_search(q["question"], k=10, mode=mode)
        rows.append({"id": q["id"], "category": q["category"], "split": q["split"],
                     "latency_ms": ms, **metrics.score_retrieval(hits, q["evidence"])})
    return rows


def summarize_retrieval(rows: list[dict]) -> dict:
    out = {"all": metrics.aggregate_retrieval(rows),
           "by_category": metrics.by_category(rows, metrics.aggregate_retrieval)}
    for sp in ("dev", "test"):
        out[sp] = metrics.aggregate_retrieval([r for r in rows if r["split"] == sp])
    return out


# ── answers ──
def make_runner(spec: dict, retriever, llm):
    """Returns fn(question) -> dict(answer, abstained, cited_hits, context, tools_used, steps, tool_errors)."""
    from src.qa.rag import answer_question, format_context
    if spec["kind"] == "rag":
        def run(q):
            r = answer_question(q["question"], retriever, llm, mode=spec["mode"], k=5)
            cited = [h for h in r["hits"] if h["chunk_id"] in set(r["citations"])]
            return {"answer": r["answer"], "abstained": r["abstained"], "cited_hits": cited,
                    "context": format_context(r["hits"])[0] if r["hits"] else "",
                    "tools_used": [], "steps": 1, "tool_errors": 0}
        return run
    from src.agent import Agent, ToolBox
    agent = Agent(ToolBox(retriever), llm)

    def run(q):
        r = agent.run(q["question"])
        cited = [{"ticker": c["ticker"], "quarter": c["quarter"], "year": c["year"], "text": c["snippet"],
                  "chunk_id": c["chunk_id"]} for c in r["citations"]]
        errs = sum(1 for e in r["trace"] if e["type"] == "tool_result" and e.get("error"))
        return {"answer": r["answer"], "abstained": r["abstained"], "cited_hits": cited,
                "context": "\n".join(r["evidence"]), "tools_used": r["tools_used"],
                "steps": r["steps"], "tool_errors": errs}
    return run


def eval_answers(questions: list[dict], runner, judge, retriever=None) -> list[dict]:
    import time
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        out = runner(q)
        ms = (time.perf_counter() - t0) * 1000
        rec = {"id": q["id"], "category": q["category"], "split": q["split"], "question": q["question"],
               "gold": q.get("gold_answer", ""), "answer": out["answer"], "abstained": out["abstained"],
               "answerable": q["answerable"], "latency_ms": ms, "agent_subset": q["agent_subset"],
               "tools_used": out["tools_used"], "steps": out["steps"], "tool_errors": out["tool_errors"],
               "faithfulness": None, "correctness": None, "citation_precision": None}
        if q["answerable"]:
            if out["abstained"]:
                rec["correctness"] = 0.0
            else:
                rec["faithfulness"] = judge.faithfulness(out["answer"], out["context"])
                gold = q.get("gold_answer", "")
                if gold and not metrics.numeric_exact_ok(gold, out["answer"]):
                    rec["correctness"] = 0.0
                elif gold:
                    rec["correctness"] = judge.correctness(q["question"], gold, out["answer"])
                if q["evidence"]:
                    rec["citation_precision"] = metrics.citation_precision(out["cited_hits"], q["evidence"])
            if q["expected_tools"]:
                rec["tool_ok"] = metrics.tool_selection_ok(q["expected_tools"], out["tools_used"])
        rows.append(rec)
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def aggregate_answers(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    agg = {"n": len(rows), "faithfulness": _mean(r["faithfulness"] for r in ans),
           "correctness": _mean(r["correctness"] for r in ans),
           "citation_precision": _mean(r["citation_precision"] for r in ans),
           "abstention": _mean(float(r["abstained"]) for r in un),
           "p50_ms": round(metrics.percentile([r["latency_ms"] for r in rows], 50), 1),
           "ci95_correctness": metrics.bootstrap_ci([r["correctness"] for r in ans])}
    ag = [r for r in rows if r["agent_subset"] and r["answerable"]]
    if ag:
        agg["agent"] = {"n": len(ag),
                        "task_success": _mean(float((r["correctness"] or 0) >= 0.5 and not r["abstained"]) for r in ag),
                        "tool_selection": _mean(float(r["tool_ok"]) for r in ag if "tool_ok" in r),
                        "avg_steps": _mean(r["steps"] for r in ag),
                        "tool_error_rate": _mean(float(r["tool_errors"] > 0) for r in ag),
                        "cost_per_question_usd": 0.0}  # local/free models
    return agg


# ── results + tables ──
def save(kind: str, rows: dict, split: str, meta: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{kind}"
    p = RESULTS / f"{run_id}.json"
    p.write_text(json.dumps({"run_id": run_id, "kind": kind, "split": split, "meta": meta, "rows": rows}, indent=1))
    return p


def latest(kind: str) -> dict | None:
    fs = sorted(RESULTS.glob(f"*-{kind}.json"))
    return json.loads(fs[-1].read_text()) if fs else None


def _f(x, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def render_retrieval(res: dict) -> str:
    L = ["| Config | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, (label, _) in RETRIEVAL_ROWS.items():
        r = res["rows"].get(key)
        if r:
            a = r["summary"]["all"]
            L.append(f"| {label} | {_f(a['recall@1'])} | {_f(a['recall@3'])} | {_f(a['recall@5'])} | "
                     f"{_f(a['recall@10'])} | {_f(a['mrr'])} | {a['p50_ms']:.0f} |")
    return "\n".join(L)


def render_retrieval_by_category(res: dict) -> str:
    cats = sorted({c for r in res["rows"].values() for c in r["summary"]["by_category"]})
    L = ["| Config | " + " | ".join(f"{c} R@5" for c in cats) + " |", "| --- |" + " --- |" * len(cats)]
    for key, (label, _) in RETRIEVAL_ROWS.items():
        r = res["rows"].get(key)
        if r:
            bc = r["summary"]["by_category"]
            L.append(f"| {label} | " + " | ".join(_f(bc[c]["recall@5"]) if c in bc else "-" for c in cats) + " |")
    return "\n".join(L)


def render_answers(res: dict) -> str:
    L = ["| System | Faithfulness | Correctness | Citation precision | Abstention | p50 latency |",
         "| --- | --- | --- | --- | --- | --- |"]
    for key, (label, _) in ANSWER_ROWS.items():
        r = res["rows"].get(key)
        if r:
            a = r["summary"]
            L.append(f"| {label} | {_f(a['faithfulness'])} | {_f(a['correctness'])} | {_f(a['citation_precision'])} | "
                     f"{_f(a['abstention'], True)} | {a['p50_ms'] / 1000:.1f}s |")
    return "\n".join(L)


def update_readme() -> None:
    readme = config.ROOT / "README.md"
    text = readme.read_text()
    for kind, marker, render in (("retrieval", "RETRIEVAL_TABLE", render_retrieval),
                                 ("answers", "ANSWER_TABLE", render_answers)):
        res = latest(kind)
        if res:
            text = re.sub(rf"(<!-- {marker}_START -->).*?(<!-- {marker}_END -->)",
                          lambda m, fn=render, r=res: f"{m.group(1)}\n{fn(r)}\n{m.group(2)}", text, flags=re.S)
    readme.write_text(text)
    out = ["# Latest evaluation results\n"]
    r = latest("retrieval")
    if r:
        out += [f"## Retrieval (run {r['run_id']}, split={r['split']})\n", render_retrieval(r),
                "\n### Recall@5 by category\n", render_retrieval_by_category(r), ""]
    a = latest("answers")
    if a:
        out += [f"## Answer quality (run {a['run_id']})\n", render_answers(a), ""]
    (RESULTS / "latest.md").write_text("\n".join(out))
    print("README tables updated; full tables in eval/results/latest.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("retrieval", "answers"):
        p = sub.add_parser(name)
        p.add_argument("--rows", nargs="*")
        p.add_argument("--split", default="all", choices=["all", "dev", "test"])
        p.add_argument("--questions")
        p.add_argument("--limit", type=int)
    sub.add_parser("tables")
    args = ap.parse_args(argv)

    if args.cmd == "tables":
        return update_readme()

    try:
        qs = metrics.load_questions(args.questions, args.split)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if args.limit:
        qs = qs[:args.limit]
    print(f"{len(qs)} questions ({args.split})")
    out = {}
    if args.cmd == "retrieval":
        for key in args.rows or RETRIEVAL_ROWS:
            label, spec = RETRIEVAL_ROWS[key]
            print(f"\n== {label}")
            rows = eval_retrieval(qs, build_retriever(spec), spec["mode"])
            out[key] = {"label": label, "spec": spec, "summary": summarize_retrieval(rows), "per_question": rows}
            a = out[key]["summary"]["all"]
            print(f"   R@5={a.get('recall@5')} MRR={a.get('mrr')} p50={a.get('p50_ms')}ms  CI95 R@5={a.get('ci95_recall@5')}")
    else:
        from eval.judges import Judge
        from src.llm import get_llm
        llm, judge = get_llm(), Judge()
        print(f"generator={llm.provider}:{llm.model}  judge={judge.llm.provider}:{judge.llm.model}")
        if judge.llm.model.split(":")[0] == llm.model.split(":")[0]:
            print("WARNING: judge and generator look like the same model family; set JUDGE_MODEL.")
        for key in args.rows or ANSWER_ROWS:
            label, spec = ANSWER_ROWS[key]
            print(f"\n== {label}")
            rows = eval_answers(qs, make_runner(spec, build_retriever({**spec, "reranker": None}), llm), judge)
            out[key] = {"label": label, "spec": spec, "summary": aggregate_answers(rows), "per_question": rows}
            print("  ", {k: v for k, v in out[key]["summary"].items() if k != "agent"})
        print(f"judge failures (unparseable): {judge.failures}")
    path = save(args.cmd, out, args.split, {"generator": config.LLM_MODEL, "judge": config.JUDGE_MODEL})
    print(f"\nwrote {path}")
    update_readme()


if __name__ == "__main__":
    main()

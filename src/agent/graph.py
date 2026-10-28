"""
src/agent/graph.py

LangGraph StateGraph (WS4):

    route -> (rag | agent <-> tools) -> synthesize -> verify -> END
                                            ^______retry once____|

* route      cheap heuristic (no LLM call): simple lookups skip the tool loop
* agent      LLM replies with a JSON action; works with ANY model (no native
             tool-calling needed), which matters for small free local models
* tools      Pydantic-validated, 10 s timeout, errors returned as observations
* synthesize cited answer from collected evidence only
* verify     every number and quoted phrase must appear in tool output;
             one retry, then abstain
Hard cap: AGENT_MAX_STEPS tool calls (default 6). A MemorySaver checkpointer
keeps per-thread history for follow-up questions.
"""

from __future__ import annotations

import operator
import queue
import re
import threading
import uuid
from typing import Annotated, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src import config as app_config
from src.agent.tools import (
    TOOL_SPECS,
    RunContext,
    ToolBox,
    infer_call_year,
    infer_fiscal_period,
    infer_ticker,
    sanitize_args,
    tool_descriptions,
)
from src.database import queries
from src.ingestion.turns import normalize_numbers
from src.llm.provider import LLMError, extract_json
from src.qa.rag import ABSTAIN, DISCLAIMER, extract_citations, is_abstention

MULTI_STEP_RE = re.compile(
    r"\b(compare|compared|comparison|change[sd]?|trend|between|versus|vs\.?|year[- ]over[- ]year|yoy|over time|"
    r"first and last|stock|share price|price|moved|reaction|sentiment|tone|across|each of|all (?:five )?companies|"
    r"how has|how did .* (?:change|evolve))\b", re.I)


SOFT_PREFIXES = ("answer uses fiscal labels", "citation mismatch")  # labelling/citation problems: keep answer + note
PRICE_TOOLS = {"get_price_reaction", "get_price_history"}
PRICE_Q_RE = re.compile(r"\b(stock|stocks|share price|shares|price|prices|priced|rally|rallied|fell|rose|"
                        r"trading|market reaction|reaction|moved)\b", re.I)


class AgentState(TypedDict, total=False):
    question: str
    history: Annotated[list, operator.add]
    route: str
    pending: Optional[dict]
    steps: int
    attempts: int
    feedback: str
    dups: int
    answer: str
    abstained: bool
    last_calls: list


def classify(question: str) -> str:
    return "agent" if MULTI_STEP_RE.search(question) else "rag"


AGENT_SYSTEM = f"""You are a research agent answering questions about company earnings calls.
You can call these tools:
{tool_descriptions()}

Reply with ONE JSON object and nothing else:
{{"thought": "short reasoning", "action": "<tool name or final>", "args": {{...}}}}

Rules:
- To find out WHAT WAS SAID, always use search_transcripts. list_transcripts cannot search.
- Search once per company or period you need (use the ticker argument); then answer with action "final".
- quarter/year are the company's FISCAL labels; the call date is the real calendar date.
- Use get_price_reaction / get_price_history ONLY when the question asks about the stock or share price.
- Do not repeat a tool call. Use action "final" when you have enough evidence or the evidence cannot answer.

Examples of correct replies:
{{"thought": "need Azure commentary", "action": "search_transcripts", "args": {{"query": "Azure capacity constraints", "ticker": "MSFT", "k": 5}}}}
{{"thought": "tone over time", "action": "get_sentiment_trend", "args": {{"ticker": "AAPL"}}}}
{{"thought": "stock move after a call", "action": "get_price_reaction", "args": {{"ticker": "NVDA", "call_date": "2025-08-27"}}}}"""

SYNTH_SYSTEM = f"""You write the final answer to a question about earnings calls using ONLY the evidence below.
Rules:
- Cite passages with their labels like [c1][c2]. Numbers from tools (sentiment, price moves) need no label.
- Quote numbers exactly as they appear in the evidence. Do not compute new numbers.
- If the evidence does not answer the question, reply exactly: {ABSTAIN}
- Report what was said; never give investment advice.
- Name every period by its CALL DATE only, e.g. "on the 2024-08-28 call". Never write labels like "Q3 2025":
  company fiscal years differ from calendar years.
- Attach each figure to the call date it came from, and list calls oldest first.
- Be concise (2-6 sentences)."""


# ── verification ──
# (?<![A-Za-z\d\[]) so digits inside identifiers (GB200, H100, FY25) are never read as numbers
_NUM_RE = re.compile(r"(?<![A-Za-z\d\[])\$?\d[\d,]*(?:\.\d+)?%?")
_QUOTE_RE = re.compile(r"[\"“]([^\"”]{15,})[\"”]")


def _n(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9. %$]", " ", normalize_numbers(s).lower())).strip()


def _has_number(haystack: str, core: str) -> bool:
    """Whole-number match: '5' must not match inside '4.5' or '15'."""
    return re.search(rf"(?<![\d.]){re.escape(core)}(?![\d]|\.\d)", haystack) is not None


def verify_answer(answer: str, evidence_text: str, question: str, evidence: dict | None = None,
                  dates: dict | None = None, other_text: str = "") -> list[str]:
    """Return a list of problems (empty = pass)."""
    problems = []
    ev, q = _n(evidence_text), _n(question)
    stripped = re.sub(r"\[c\d+\]", "", answer)
    for m in _NUM_RE.findall(stripped):
        core = m.strip("$%").replace(",", "").rstrip(".")
        if not core or (core.isdigit() and len(core) == 1 and not m.startswith("$") and not m.endswith("%")):
            continue
        if _has_number(q.replace(",", ""), core):
            continue
        if not _has_number(ev.replace(",", ""), core):
            problems.append(f"number '{m}' not found in the evidence")
    if infer_call_year(question) and re.search(r"\bQ[1-4]\s*(?:FY\s*)?20\d\d\b", stripped, re.I):
        problems.append("answer uses fiscal labels like 'Q3 2025'; name each period by its call date (YYYY-MM-DD)")
    for qt in _QUOTE_RE.findall(stripped):
        if _n(qt) not in ev:
            problems.append(f"quote \"{qt[:50]}...\" not found in the evidence")
    if evidence:
        problems += _check_citations(answer, evidence, dates or {}, other_text, q)
    return problems


def _check_citations(answer: str, evidence: dict, dates: dict, other_text: str, q: str) -> list[str]:
    """Per sentence: a call date must be the call date of the passage cited in that sentence (hard), and a number
    should appear in the passage it cites (soft: it may sit in a neighbouring passage of the same call)."""
    problems: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        labs = [lab for lab in dict.fromkeys(re.findall(r"\[(c\d+)\]", sent)) if lab in evidence]
        if not labs:
            continue
        clean = re.sub(r"\[c\d+\]", "", sent)
        allowed = {dates.get((evidence[lab]["ticker"], evidence[lab]["quarter"], evidence[lab]["year"]))
                   for lab in labs} - {None}
        for d in re.findall(r"\d{4}-\d{2}-\d{2}", clean):
            if allowed and d not in allowed:
                problems.append(f"date {d} does not match the call date of {', '.join(labs)} "
                                f"({', '.join(sorted(allowed))}); name the date of the passage you cite")
        cited = _n(" ".join(evidence[lab]["text"] for lab in labs) + " " + other_text).replace(",", "")
        for m in _NUM_RE.findall(re.sub(r"\d{4}-\d{2}-\d{2}", "", clean)):
            core = m.strip("$%").replace(",", "").rstrip(".")
            if not core or (core.isdigit() and len(core) == 1 and not m.startswith("$") and not m.endswith("%")):
                continue
            if _has_number(q.replace(",", ""), core) or _has_number(cited, core):
                continue
            problems.append(f"citation mismatch: number '{m}' is not in {', '.join(labs)}, the passage it is cited to")
    return problems


class Agent:
    def __init__(self, toolbox: ToolBox, llm, max_steps: int | None = None, checkpointer=None):
        self.tools = toolbox
        self.llm = llm
        self.max_steps = max_steps or app_config.AGENT_MAX_STEPS
        self.graph = self._build(checkpointer or MemorySaver())

    # ── nodes ──
    @staticmethod
    def _ctx(config: RunnableConfig) -> RunContext:
        return config["configurable"]["ctx"]

    def _route(self, state: AgentState, config: RunnableConfig):
        route = classify(state["question"])
        self._ctx(config).event("route", route=route)
        return {"route": route}

    def _tickers(self) -> set[str]:
        return {r["ticker"] for r in queries.get_all_transcripts()}

    def _rag(self, state: AgentState, config: RunnableConfig):
        ctx = self._ctx(config)
        args = {"query": state["question"], "k": 5}
        tk = infer_ticker(state["question"], self._tickers())
        if tk:
            args["ticker"] = tk
        if infer_call_year(state["question"]):
            args["call_year"] = infer_call_year(state["question"])
        fp = infer_fiscal_period(state["question"])
        if fp:  # "Q3 2025" is a fiscal label: search only that call (a missing call then abstains correctly)
            args["quarter"], args["year"] = fp
        ctx.event("tool_call", tool="search_transcripts", args=args)
        obs = self.tools.call("search_transcripts", args, ctx)
        ctx.tools_used.append("search_transcripts")
        ctx.evidence_text.append(obs)
        ctx.event("tool_result", tool="search_transcripts", result=obs[:600])
        return {"steps": 1}

    def _fallback(self, state: AgentState, ctx: RunContext) -> None:
        """The model stopped without any real evidence (it only listed transcripts, or its calls kept failing).
        Search with the user's own question once, so a weak model degrades to plain cited RAG instead of
        abstaining."""
        if ctx.substantive or ctx.fell_back:
            return
        ctx.fell_back = True
        args = {"query": state["question"], "k": 8}
        tk = infer_ticker(state["question"], self._tickers())
        if tk:
            args["ticker"] = tk
        if infer_call_year(state["question"]):
            args["call_year"] = infer_call_year(state["question"])
        fp = infer_fiscal_period(state["question"])
        if fp:  # "Q3 2025" is a fiscal label: search only that call (a missing call then abstains correctly)
            args["quarter"], args["year"] = fp
        ctx.event("limit", message="the model never ran a search, so searching with your question directly")
        ctx.event("tool_call", tool="search_transcripts", args=args)
        obs = self.tools.call("search_transcripts", args, ctx)
        ctx.tools_used.append("search_transcripts")
        ctx.evidence_text.append(obs)
        ctx.event("tool_result", tool="search_transcripts", result=obs[:600], error=obs.startswith("ERROR"))

    def _agent(self, state: AgentState, config: RunnableConfig):
        ctx = self._ctx(config)
        if state.get("steps", 0) >= self.max_steps:
            ctx.event("limit", message=f"step cap ({self.max_steps}) reached")
            self._fallback(state, ctx)
            return {"pending": None}
        obs_text = "\n\n".join(f"Observation {i}: {o[:1800]}" for i, o in enumerate(ctx.evidence_text, 1)) or "(none yet)"
        hist = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in (state.get("history") or [])[-2:])
        user = (f"{'Earlier conversation:' + chr(10) + hist + chr(10) + chr(10) if hist else ''}"
                f"Calls available (fiscal labels):\n{self.tools.catalog()}\n\n"
                f"Question: {state['question']}\n\nObservations so far:\n{obs_text}\n\nNext JSON action:")
        try:
            raw = self.llm.chat([{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user}],
                                temperature=0.0, max_tokens=300, json_mode=True)
            act = extract_json(raw)
        except LLMError as e:
            if e.kind == "parse":
                ctx.evidence_text.append("ERROR: your last reply was not valid JSON. Reply with JSON only.")
                ctx.event("tool_result", tool="(parse)", result="invalid JSON from model")
                return {"pending": {"tool": "__retry__"}, "steps": state.get("steps", 0) + 1}
            ctx.event("error", message=str(e))
            self._fallback(state, ctx)
            return {"pending": None}
        name = str(act.get("action", "final")).strip()
        if name in ("final", "answer", "finish", ""):
            ctx.event("thought", text=str(act.get("thought", ""))[:300], action="final")
            self._fallback(state, ctx)
            return {"pending": None}
        args = act.get("args") if isinstance(act.get("args"), dict) else {}
        if name in TOOL_SPECS:
            args = sanitize_args(name, args)
        if name in PRICE_TOOLS and not PRICE_Q_RE.search(state["question"]):
            ctx.evidence_text.append(f"ERROR: the question does not ask about the stock price, so {name} was skipped. "
                                     f"Use search_transcripts, or reply with action final.")
            ctx.event("tool_result", tool=name, result="skipped: the question is not about stock prices")
            return {"pending": {"tool": "__retry__"}, "steps": state.get("steps", 0) + 1}
        core = {k: v for k, v in args.items() if k not in ("windows", "k")}  # same call, different knobs = duplicate
        key = name if name == "list_transcripts" else f"{name}:{sorted(core.items(), key=str)}"
        repeat = False
        if name == "search_transcripts":  # a weak model re-words the query forever: 2 searches per filter set is enough
            fk = "F:" + "|".join(str(core.get(x)) for x in ("ticker", "quarter", "year", "call_year", "section"))
            repeat = sum(1 for k in (state.get("last_calls") or []) if k.startswith(fk)) >= 2
            key = f"{fk}#{key}"
        if repeat or key in (state.get("last_calls") or []):
            dups = state.get("dups", 0) + 1
            if dups >= 2 or ctx.substantive:  # a model that keeps repeating itself will not improve: answer now
                ctx.event("limit", message="the model is repeating itself, so answering with the evidence gathered")
                self._fallback(state, ctx)
                return {"pending": None, "dups": dups}
            ctx.evidence_text.append(f"ERROR: duplicate call to {name}. You already have that result. "
                                     f"Reply with action final, or call a different tool.")
            ctx.event("tool_result", tool=name, result="duplicate call skipped")
            return {"pending": {"tool": "__retry__"}, "steps": state.get("steps", 0) + 1, "dups": dups}
        ctx.event("thought", text=str(act.get("thought", ""))[:300], action=name)
        return {"pending": {"tool": name, "args": args}, "last_calls": (state.get("last_calls") or []) + [key]}

    def _tools(self, state: AgentState, config: RunnableConfig):
        ctx = self._ctx(config)
        p = state["pending"]
        if p["tool"] == "__retry__":
            return {"pending": None}
        if p["tool"] == "search_transcripts":
            cy = infer_call_year(state["question"])
            if cy and "call_year" not in p["args"]:  # a bare "2025" from the user means the calendar year
                p["args"] = {**{k: v for k, v in p["args"].items() if k != "year"}, "call_year": cy}
            fp = infer_fiscal_period(state["question"])
            if fp and not ({"quarter", "year"} & set(p["args"])):
                p["args"] = {**p["args"], "quarter": fp[0], "year": fp[1]}
        ctx.event("tool_call", tool=p["tool"], args=p["args"])
        obs = self.tools.call(p["tool"], p["args"], ctx)
        ctx.tools_used.append(p["tool"])
        ctx.evidence_text.append(obs)
        ctx.event("tool_result", tool=p["tool"], result=obs[:600], error=obs.startswith("ERROR"))
        return {"steps": state.get("steps", 0) + 1, "pending": None}

    def _synthesize(self, state: AgentState, config: RunnableConfig):
        ctx = self._ctx(config)
        good = [o for o in ctx.evidence_text if not o.startswith("ERROR") and not o.startswith("No ")]
        if not good:
            return {"answer": ABSTAIN}
        ev = "\n\n".join(good)[:7000]
        fb = f"\n\nYour previous draft had problems: {state['feedback']}. Fix them or reply {ABSTAIN}." \
            if state.get("feedback") else ""
        try:
            ans = self.llm.chat([{"role": "system", "content": SYNTH_SYSTEM},
                                 {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {state['question']}{fb}"}],
                                temperature=0.0, max_tokens=400).strip()
        except LLMError as e:
            ctx.event("error", message=str(e))
            return {"answer": ABSTAIN}
        return {"answer": ans}

    def _verify(self, state: AgentState, config: RunnableConfig):
        ctx = self._ctx(config)
        ans = state["answer"]
        if is_abstention(ans):
            ctx.event("verify", ok=True, note="abstained")
            return {"answer": ABSTAIN, "abstained": True, "feedback": ""}
        dates = self._dates()
        other = "\n".join(o for o in ctx.evidence_text if not re.search(r"^\[c\d+\]", o, re.M))
        problems = verify_answer(ans, "\n".join(ctx.evidence_text), state["question"], ctx.evidence, dates, other)
        hard = [p for p in problems if not p.startswith(SOFT_PREFIXES)]
        ctx.event("verify", ok=not problems, problems=problems)
        if not problems:
            return {"abstained": False, "feedback": ""}
        if state.get("attempts", 0) < 1:
            fb = "; ".join(problems[:3])
            if any(p.startswith(("date ", "answer uses fiscal", "citation mismatch")) for p in problems):
                fb += ". Call dates of your sources: " + ", ".join(
                    f"[{lab}]={dates.get((h['ticker'], h['quarter'], h['year'])) or 'unknown'}"
                    for lab, h in ctx.evidence.items())
            return {"attempts": state.get("attempts", 0) + 1, "feedback": fb}
        if not hard:  # only labelling / citation problems: keep the verified answer and say what to double-check
            ctx.event("limit", message="kept the answer, with a note about period labels / citations")
            return {"answer": ans + "\n\n" + self._legend(ans, ctx, problems), "abstained": False, "feedback": ""}
        return {"answer": ABSTAIN, "abstained": True, "feedback": ""}

    def _dates(self) -> dict:
        return {(r["ticker"], r["quarter"], r["year"]): r["date"] for r in queries.get_all_transcripts()}

    def _legend(self, ans: str, ctx: RunContext, problems: list[str]) -> str:
        dates = self._dates()
        labs = [lab for lab in dict.fromkeys(re.findall(r"\[(c\d+)\]", ans)) if lab in ctx.evidence]
        labs = labs or list(ctx.evidence)[:5]
        rows = [f"[{lab}] {ctx.evidence[lab]['ticker']} call "
                f"{dates.get((ctx.evidence[lab]['ticker'], ctx.evidence[lab]['quarter'], ctx.evidence[lab]['year'])) or 'date unknown'}"
                f" (fiscal {ctx.evidence[lab]['quarter']} {ctx.evidence[lab]['year']})" for lab in labs]
        note = []
        if any(p.startswith("answer uses fiscal") for p in problems):
            note.append("Labels such as 'Q3 2025' are the company's fiscal quarters, not calendar quarters.")
        if any(p.startswith("citation mismatch") for p in problems):
            note.append("Some figures may belong to a different passage than the one cited; check them in the sources.")
        return "Note: " + " ".join(note) + (" Sources: " + "; ".join(rows) + "." if rows else "")

    # ── wiring ──
    def _build(self, checkpointer):
        g = StateGraph(AgentState)
        for name, fn in [("route", self._route), ("rag", self._rag), ("agent", self._agent),
                         ("tools", self._tools), ("synthesize", self._synthesize), ("verify", self._verify)]:
            g.add_node(name, fn)
        g.set_entry_point("route")
        g.add_conditional_edges("route", lambda s: s["route"], {"rag": "rag", "agent": "agent"})
        g.add_edge("rag", "synthesize")
        g.add_conditional_edges("agent", lambda s: "tools" if s.get("pending") else "synthesize",
                                {"tools": "tools", "synthesize": "synthesize"})
        g.add_edge("tools", "agent")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", lambda s: "synthesize" if s.get("feedback") else END,
                                {"synthesize": "synthesize", END: END})
        return g.compile(checkpointer=checkpointer)

    # ── public API ──
    def run(self, question: str, thread_id: str | None = None, emit=None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex[:12]
        ctx = RunContext(emit=emit)
        cfg = {"configurable": {"thread_id": thread_id, "ctx": ctx}, "recursion_limit": 60}
        state = self.graph.invoke(
            {"question": question, "steps": 0, "attempts": 0, "dups": 0, "pending": None, "feedback": "",
             "answer": "", "abstained": False, "last_calls": []}, cfg)
        answer = state.get("answer") or ABSTAIN
        abstained = bool(state.get("abstained")) or is_abstention(answer)
        labels = {lab: h["chunk_id"] for lab, h in ctx.evidence.items()}
        cited = [] if abstained else extract_citations(answer, labels)
        by_id = {h["chunk_id"]: (lab, h) for lab, h in ctx.evidence.items()}
        citations = [{"label": by_id[c][0], "chunk_id": c, "ticker": by_id[c][1]["ticker"],
                      "quarter": by_id[c][1]["quarter"], "year": by_id[c][1]["year"],
                      "speaker": by_id[c][1]["speaker"], "snippet": by_id[c][1]["text"][:240]} for c in cited]
        self.graph.update_state({"configurable": {"thread_id": thread_id}},
                                {"history": [{"q": question, "a": answer}]}, as_node="verify")
        result = {"answer": answer, "abstained": abstained, "citations": citations,
                  "route": state.get("route"), "steps": state.get("steps", 0), "tools_used": ctx.tools_used,
                  "trace": ctx.trace, "thread_id": thread_id, "disclaimer": DISCLAIMER,
                  "evidence": list(ctx.evidence_text)}
        ctx.event("answer", **{k: result[k] for k in ("answer", "abstained", "citations")})
        return result

    def stream(self, question: str, thread_id: str | None = None):
        """Generator of trace events (route, thought, tool_call, tool_result, verify, answer, done)."""
        q: queue.Queue = queue.Queue()
        SENTINEL = object()
        thread_id = thread_id or uuid.uuid4().hex[:12]

        def worker():
            try:
                self.run(question, thread_id, emit=q.put)
            except Exception as e:  # surface as an event instead of a dead stream
                q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                q.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            yield item
        yield {"type": "done", "thread_id": thread_id}

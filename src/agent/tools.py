"""
src/agent/tools.py

Agent tools (WS4). Each tool has a Pydantic argument model, returns a compact
observation, and registers evidence under citation labels (c1, c2...) held in a
per-run RunContext (so one ToolBox is safe to share across requests).
Tool errors are returned as observations, never raised, so the agent can retry
or relay them.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import queue
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from src import config
from src.agent.prices import PriceError, PriceProvider
from src.database import queries
from src.sentiment.finbert import aggregate_by_section


@dataclass
class RunContext:
    """Per-question state: evidence registry, trace, tool usage, event sink."""
    emit: Optional[Callable[[dict], None]] = None
    trace: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)       # label -> hit dict
    evidence_text: list = field(default_factory=list)  # every observation text (for verification)
    tools_used: list = field(default_factory=list)
    q: queue.Queue | None = None
    substantive: int = 0      # successful search/sentiment/price calls (list_transcripts does not count)
    fell_back: bool = False   # the direct-search fallback already ran

    def event(self, kind: str, **data) -> None:
        e = {"type": kind, **data}
        self.trace.append(e)
        if self.emit:
            self.emit(e)

    def register(self, hit: dict) -> str:
        for lab, h in self.evidence.items():
            if h["chunk_id"] == hit["chunk_id"]:
                return lab
        lab = f"c{len(self.evidence) + 1}"
        self.evidence[lab] = hit
        return lab


# ── argument models ──
class ListArgs(BaseModel):
    ticker: Optional[str] = None
    year: Optional[int] = None


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    ticker: Optional[str] = None
    quarter: Optional[str] = Field(None, pattern=r"^[Qq][1-4]$")
    year: Optional[int] = Field(None, ge=2000, le=2100)       # FISCAL year label of the transcript
    call_year: Optional[int] = Field(None, ge=2000, le=2100)  # CALENDAR year the call took place
    section: Optional[str] = Field(None, pattern=r"^(prepared|qa)$")
    k: int = Field(5, ge=1, le=8)


class SentimentArgs(BaseModel):
    ticker: str
    quarter: str = Field(pattern=r"^[Qq][1-4]$")
    year: int
    section: Optional[str] = Field(None, pattern=r"^(prepared|qa)$")


class TrendArgs(BaseModel):
    ticker: str


class ReactionArgs(BaseModel):
    ticker: str
    call_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    quarter: Optional[str] = Field(None, pattern=r"^[Qq][1-4]$")
    year: Optional[int] = None
    windows: list[int] = Field(default_factory=lambda: [1, 5])


class HistoryArgs(BaseModel):
    ticker: str
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


TOOL_SPECS = {
    "list_transcripts": ("List which calls exist (ticker, quarter, year, call date); optional ticker/year filter. "
                         "It NEVER returns what was said and cannot search - use search_transcripts for that.", ListArgs),
    "search_transcripts": ("Search transcript passages. Args: query, optional ticker, quarter (Q1-Q4), "
                           "year (fiscal label), call_year (calendar year of the call), section (prepared|qa), k. "
                           "Use call_year when the user asks about a calendar year like 'in 2025'.", SearchArgs),
    "get_sentiment": ("FinBERT sentiment for one call. Args: ticker, quarter, year, optional section.", SentimentArgs),
    "get_sentiment_trend": ("FinBERT net sentiment for every available quarter of a ticker.", TrendArgs),
    "get_price_reaction": ("Stock move after an earnings call over 1-day and 5-day windows. Args: ticker and "
                           "either call_date (YYYY-MM-DD) or quarter+year.", ReactionArgs),
    "get_price_history": ("Summary of closing prices between two dates. Args: ticker, start, end (YYYY-MM-DD).",
                          HistoryArgs),
}


SUBSTANTIVE_TOOLS = {"search_transcripts", "get_sentiment", "get_sentiment_trend", "get_price_reaction",
                     "get_price_history"}

COMPANY_ALIASES = {"nvidia": "NVDA", "microsoft": "MSFT", "azure": "MSFT", "apple": "AAPL", "amazon": "AMZN",
                   "aws": "AMZN", "alphabet": "GOOGL", "google": "GOOGL", "meta": "META", "tesla": "TSLA"}
_PLACEHOLDERS = {"", "[]", "{}", "all", "any", "none", "null", "n/a", "na", "*", "unknown", "nan"}


def infer_call_year(question: str) -> int | None:
    """'across 2025', 'in 2025', 'during 2025' -> 2025, meaning the CALENDAR year of the call. Not inferred when
    the user speaks in fiscal labels ('Q3 2025', 'FY2025', 'fiscal 2025')."""
    if re.search(r"\b(q[1-4]|fy|fiscal)\b|\bfy\d", question, re.I):
        return None
    m = re.search(r"\b(?:in|during|across|throughout|over|for|of|since|by|through)\s+(?:the\s+)?(?:year\s+)?(20\d{2})\b",
                  question, re.I)
    return int(m.group(1)) if m else None


def infer_fiscal_period(question: str) -> tuple[str, int] | None:
    """'Q3 2025', 'fiscal Q3 2025', 'Q3 FY2025', 'FY2025 Q3' -> ('Q3', 2025): the company's FISCAL label, which is
    how transcript files are named. None unless exactly one such period is named."""
    found = set()
    for m in re.finditer(r"\bQ([1-4])\s*(?:FY\s*)?(20\d{2})\b|\bQ([1-4])\s*FY\s*(\d{2})\b", question, re.I):
        q, y = (m.group(1), int(m.group(2))) if m.group(1) else (m.group(3), 2000 + int(m.group(4)))
        found.add((f"Q{q}", y))
    for m in re.finditer(r"\b(?:FY|fiscal(?:\s+year)?)\s*(20\d{2})\s*,?\s*Q([1-4])\b", question, re.I):
        found.add((f"Q{m.group(2)}", int(m.group(1))))
    return next(iter(found)) if len(found) == 1 else None


def infer_ticker(question: str, known: set[str] | list[str]) -> str | None:
    """The one ticker a question is clearly about (company name or symbol), else None."""
    known = {k.upper() for k in known}
    found = {COMPANY_ALIASES[w] for w in re.findall(r"[a-z]+", question.lower()) if w in COMPANY_ALIASES}
    found |= {w.upper() for w in re.findall(r"[A-Za-z]+", question) if w.upper() in known}
    found &= known
    return next(iter(found)) if len(found) == 1 else None


def sanitize_args(name: str, args: dict | None) -> dict:
    """Small local models send messy arguments ('[]' for empty, 'Q3 2025' for a quarter, '[1, 5]' for a
    list, 'NVIDIA' for a ticker). Normalize what is unambiguous and drop what is not, so a sloppy-but-obvious
    call still works instead of burning one of the agent's few steps on a validation error."""
    out: dict = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            v = v.strip()
            if v.lower() in _PLACEHOLDERS:
                continue
        if v is None or (isinstance(v, (list, dict)) and not v):
            continue
        out[k] = v

    if isinstance(out.get("ticker"), str):
        t = out["ticker"].strip()
        out["ticker"] = COMPANY_ALIASES.get(t.lower(), re.sub(r"[^A-Za-z.\-]", "", t).upper())
    if "quarter" in out:
        q = str(out["quarter"])
        m = re.search(r"\b[Qq]([1-4])\b", q) or re.fullmatch(r"\s*([1-4])\s*", q)
        y = re.search(r"\b(20\d{2})\b", q)
        if y and "year" not in out:
            out["year"] = y.group(1)
        if m:
            out["quarter"] = f"Q{m.group(1)}"
        else:
            out.pop("quarter")
    for key in ("year", "call_year"):
        if key in out:
            y = re.search(r"20\d{2}", str(out[key]))
            if y:
                out[key] = int(y.group())
            else:
                out.pop(key)
    if "section" in out:
        sec = str(out["section"]).lower()
        if sec in ("qa", "q&a", "q and a") or "question" in sec:
            out["section"] = "qa"
        elif "prep" in sec:
            out["section"] = "prepared"
        else:
            out.pop("section")
    if "k" in out:
        try:
            out["k"] = max(1, min(8, int(float(str(out["k"])))))
        except ValueError:
            out.pop("k")
    if "windows" in out:
        w = out["windows"]
        if isinstance(w, str):
            try:
                w = json.loads(w)
            except ValueError:
                w = re.findall(r"\d+", w)
        if isinstance(w, (int, float)):
            w = [w]
        try:
            w = [int(x) for x in w if 1 <= int(x) <= 30]
        except (TypeError, ValueError):
            w = []
        if w:
            out["windows"] = w
        else:
            out.pop("windows")
    if name != "list_transcripts":  # unknown keys are noise from the model; drop them
        fields = set(TOOL_SPECS[name][1].model_fields)
        out = {k: v for k, v in out.items() if k in fields}
    return out


def tool_descriptions() -> str:
    lines = []
    for name, (desc, model) in TOOL_SPECS.items():
        props = ", ".join(model.model_json_schema().get("properties", {}).keys())
        lines.append(f"- {name}({props}): {desc}")
    return "\n".join(lines)


class ToolBox:
    def __init__(self, retriever, prices: PriceProvider | None = None, timeout: int | None = None):
        self.retriever = retriever
        self.prices = prices or PriceProvider()
        self.timeout = timeout or config.TOOL_TIMEOUT_S
        self._pool = cf.ThreadPoolExecutor(max_workers=4)

    # ── dispatcher with validation + timeout ──
    def call(self, name: str, args: dict, ctx: RunContext) -> str:
        if name not in TOOL_SPECS:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOL_SPECS)}"
        args = sanitize_args(name, args)
        if name == "list_transcripts" and set(args) - {"ticker", "year"}:
            return ("ERROR: list_transcripts only accepts ticker and year and never returns what was said. "
                    "To find what management said, call search_transcripts with a query "
                    "(and optional ticker/year/section).")
        try:
            parsed = TOOL_SPECS[name][1](**args)
        except ValidationError as e:
            msgs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            return f"ERROR: invalid arguments for {name}: {msgs}"
        fut = self._pool.submit(getattr(self, f"_{name}"), parsed, ctx)
        try:
            result = fut.result(timeout=self.timeout)
            if name in SUBSTANTIVE_TOOLS and not result.startswith(("ERROR", "No ")):
                ctx.substantive += 1
            return result
        except cf.TimeoutError:
            return f"ERROR: {name} timed out after {self.timeout}s"
        except PriceError as e:
            return f"ERROR: {e}"
        except Exception as e:  # tool bugs become observations too
            return f"ERROR: {name} failed: {type(e).__name__}: {e}"

    def catalog(self) -> str:
        """Compact 'what exists' summary for the agent prompt, so it rarely needs list_transcripts."""
        by: dict[str, list] = {}
        for r in queries.get_all_transcripts():
            by.setdefault(r["ticker"], []).append((r["year"], r["quarter"]))
        return "\n".join(f"{t}: " + ", ".join(f"{q} {y}" for y, q in sorted(v)) for t, v in sorted(by.items()))

    # ── tools ──
    def _list_transcripts(self, a: ListArgs, ctx: RunContext) -> str:
        rows = queries.get_all_transcripts()
        if a.ticker:
            rows = [r for r in rows if r["ticker"] == a.ticker.upper()]
        if a.year:
            rows = [r for r in rows if r["year"] == a.year]
        if not rows:
            return "No transcripts are loaded for that filter."
        return "\n".join(f"{r['ticker']} {r['quarter']} {r['year']} call_date={r['date'] or 'unknown'}"
                         for r in sorted(rows, key=lambda r: (r["ticker"], r["year"], r["quarter"])))

    def _across_calls(self, a: SearchArgs, keys: list, filters: dict) -> list[dict]:
        """Best passages from EACH call in `keys` (a year's worth of calls), oldest first, so an answer about
        change over time sees every period instead of whichever single call ranks highest."""
        key_of = lambda h: (h["ticker"], h["quarter"], h["year"])  # noqa: E731
        if len(keys) > 12:  # too many calls to query one by one: over-fetch once and filter
            allowed = set(keys)
            pool = self.retriever.search(a.query, k=40, mode="hybrid_rerank", filters=filters)
            pool = [h for h in pool if key_of(h) in allowed]
        else:
            pool = []
            for t, q, y in keys:
                pool += self.retriever.search(a.query, k=3, mode="hybrid", filters={**filters, "ticker": t,
                                                                                    "quarter": q, "year": y})
            rr = self.retriever.reranker
            try:
                if getattr(rr, "loading", False):
                    raise RuntimeError("reranker is still loading")
                scores = rr.score(a.query, [h["text"] for h in pool]) if pool else []
                pool = [dict(h, score=float(s)) for h, s in sorted(zip(pool, scores), key=lambda x: -x[1])]
            except Exception:  # noqa: BLE001  (reranker unavailable: keep hybrid order)
                pool.sort(key=lambda h: -h["score"])
        groups: dict = {}
        for h in pool:
            groups.setdefault(key_of(h), []).append(h)
        picked: list[dict] = []
        while len(picked) < a.k and any(groups.values()):  # round-robin: best hit of every call first
            for g in groups.values():
                if g and len(picked) < a.k:
                    picked.append(g.pop(0))
        order = {k: i for i, k in enumerate(keys)}
        return sorted(picked, key=lambda h: order.get(key_of(h), 99))

    def _search_transcripts(self, a: SearchArgs, ctx: RunContext) -> str:
        filters = {"ticker": a.ticker, "quarter": a.quarter, "year": a.year, "section": a.section}
        dates = {(r["ticker"], r["quarter"], r["year"]): r["date"] for r in queries.get_all_transcripts()}
        if a.call_year:  # calendar year of the call: search each call in that year, then merge
            keys = sorted((k for k, d in dates.items() if (d or "").startswith(str(a.call_year))
                           and (not a.ticker or k[0] == a.ticker.upper())
                           and (not a.quarter or k[1] == a.quarter) and (not a.year or k[2] == a.year)),
                          key=lambda k: dates[k])
            if not keys:
                return f"No transcripts have a call date in {a.call_year}."
            hits = self._across_calls(a, keys, filters)
        else:
            hits = self.retriever.search(a.query, k=a.k, mode="hybrid_rerank", filters=filters)
        if not hits:
            return "No matching passages found."
        out = []
        for h in hits:
            lab = ctx.register(h)
            who = f" {h['speaker']}" if h.get("speaker") else ""
            when = dates.get((h["ticker"], h["quarter"], h["year"]))
            head = (f"{h['ticker']} call {when} (fiscal {h['quarter']} {h['year']})" if when
                    else f"{h['ticker']} {h['quarter']} {h['year']}")
            out.append(f"[{lab}] {head}{who} ({h['section'] or 'n/a'}): {h['text']}")
        return "\n".join(out)

    def _sentiment_for(self, ticker: str, quarter: str, year: int, section: str | None) -> dict:
        t = queries.get_transcript_by_key(ticker.upper(), quarter.upper(), year)
        if not t:
            raise PriceError(f"No transcript for {ticker.upper()} {quarter.upper()} {year}")
        s = queries.get_sentiment(t["id"])
        if not s:
            raise PriceError(f"Sentiment not computed for {ticker.upper()} {quarter.upper()} {year}")
        agg = aggregate_by_section(s["sentence_scores"], section) if section else {
            k: s[k] for k in ("positive_pct", "neutral_pct", "negative_pct", "net_score")}
        if not agg:
            raise PriceError(f"No '{section}' sentences scored (older rows lack section data)")
        return {k: agg[k] for k in ("positive_pct", "neutral_pct", "negative_pct", "net_score")}

    def _get_sentiment(self, a: SentimentArgs, ctx: RunContext) -> str:
        r = self._sentiment_for(a.ticker, a.quarter, a.year, a.section)
        return (f"Sentiment {a.ticker.upper()} {a.quarter.upper()} {a.year}"
                f"{' ' + a.section if a.section else ''}: " + json.dumps(r))

    def _get_sentiment_trend(self, a: TrendArgs, ctx: RunContext) -> str:
        rows = [t for t in queries.get_all_transcripts() if t["ticker"] == a.ticker.upper()]
        if not rows:
            raise PriceError(f"No transcripts for {a.ticker.upper()}")
        out = []
        for t in sorted(rows, key=lambda r: (r["year"], r["quarter"])):
            s = queries.get_sentiment(t["id"])
            lab = (f"call {t['date']} (fiscal {t['quarter']} {t['year']})" if t.get("date")
                   else f"{t['quarter']} {t['year']}")
            out.append(f"{lab}: net_score={s['net_score']:+.3f}" if s else f"{lab}: not computed")
        return f"Net sentiment trend {a.ticker.upper()}:\n" + "\n".join(out)

    def _get_price_reaction(self, a: ReactionArgs, ctx: RunContext) -> str:
        call_date = a.call_date
        if not call_date:
            if not (a.quarter and a.year):
                raise PriceError("provide call_date, or quarter and year")
            t = queries.get_transcript_by_key(a.ticker.upper(), a.quarter.upper(), a.year)
            if not t or not t.get("date"):
                raise PriceError(f"call date unknown for {a.ticker.upper()} {a.quarter.upper()} {a.year}; "
                                 f"add it to data/manifest.csv")
            call_date = t["date"]
        r = self.prices.reaction(a.ticker, call_date, tuple(a.windows))
        parts = [f"{k}: {v['pct_change']:+.2f}%" if v else f"{k}: not available yet" for k, v in r["moves"].items()]
        return (f"Price reaction {r['ticker']} (call {r['call_date']}, baseline close {r['baseline_close']} "
                f"on {r['baseline_date']}): " + ", ".join(parts))

    def _get_price_history(self, a: HistoryArgs, ctx: RunContext) -> str:
        rows = self.prices.history(a.ticker, a.start, a.end)
        if not rows:
            raise PriceError("no price rows in range")
        closes = [c for _, c in rows]
        return (f"{a.ticker.upper()} closes {rows[0][0]}..{rows[-1][0]}: first={closes[0]:.2f}, "
                f"last={closes[-1]:.2f}, min={min(closes):.2f}, max={max(closes):.2f}, n={len(rows)}")

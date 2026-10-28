"""
src/summarization/summarizer.py

Map-reduce summarizer (R0.5, F3, F10, F12). v1 read only the first 1,200 words
(~19% of a call, none of the Q&A). This version:
  MAP    each ~700-word window of the call -> short notes + candidate verbatim quotes
  REDUCE notes -> SUMMARY / TAKEAWAYS / RISKS / OPPORTUNITIES / TONE / JUSTIFICATION
  VERIFY quotes are kept only if they are verbatim substrings of the transcript
The label is a *tone* read of the call, not an investment signal.
"""

from __future__ import annotations

import re

from src.llm.provider import LLMError, get_llm

NOT_ADVICE = "Tone signal describes the language used on the call. It is not investment advice."
WINDOW_WORDS = 700

MAP_PROMPT = """You are reading ONE part of an earnings call. Write notes for an analyst.
Respond in EXACTLY this format, no markdown:

NOTES:
- fact or statement 1 (include numbers exactly)
- fact or statement 2
- fact or statement 3
QUOTES:
- one short verbatim sentence copied exactly from the text
- (optional) another verbatim sentence"""

REDUCE_PROMPT = """You are a financial analyst. You are given notes from every part of an earnings call.
Respond in EXACTLY this format, no markdown:

SUMMARY:
(2-4 short sentences)

TAKEAWAYS:
- takeaway 1
- takeaway 2
- takeaway 3

RISKS:
- risk 1
- risk 2

OPPORTUNITIES:
- opportunity 1
- opportunity 2

TONE:
Positive, Neutral, or Negative (management's overall tone)

JUSTIFICATION:
(one sentence explaining the tone using the notes)"""

_TONE_TO_LABEL = {"positive": "Bullish", "negative": "Bearish", "neutral": "Neutral",
                  "bullish": "Bullish", "bearish": "Bearish"}


def _windows(text: str, size: int = WINDOW_WORDS) -> list[str]:
    paras, out, cur, n = [p for p in text.split("\n") if p.strip()], [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > size:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        out.append("\n".join(cur))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[“”\"']", "", s)).strip().lower()


def verify_quotes(candidates: list[str], transcript: str, limit: int = 3) -> list[str]:
    """Keep only quotes that appear verbatim (modulo whitespace/quote marks)."""
    hay, out = _norm(transcript), []
    for q in candidates:
        q = q.strip().strip('"“”')
        if len(q.split()) >= 5 and _norm(q) in hay and q not in out:
            out.append(q)
        if len(out) >= limit:
            break
    return out


def _bullets(block: str) -> list[str]:
    return [re.sub(r"^[-*•]\s*", "", l.strip()) for l in block.splitlines()
            if l.strip().startswith(("-", "*", "•")) and len(l.strip()) > 2]


def _sections(raw: str, names: list[str]) -> dict[str, str]:
    pat = re.compile(r"^\s*(" + "|".join(names) + r")\s*:?\s*(.*)$", re.I)
    out: dict[str, list[str]] = {}
    cur = None
    for line in raw.splitlines():
        m = pat.match(line)
        if m and (not m.group(2) or line.strip().upper().startswith(m.group(1).upper() + ":")):
            cur = m.group(1).upper()
            out.setdefault(cur, [])
            if m.group(2):
                out[cur].append(m.group(2))
            continue
        if cur:
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def summarize_transcript(text: str, model: str | None = None, llm=None, progress=None) -> dict:
    llm = llm or get_llm(model=model)
    model_name = getattr(llm, "model", model or "unknown")
    try:
        notes, quote_cands = [], []
        parts = _windows(text)
        for i, part in enumerate(parts, 1):
            if progress:
                progress(f"summarizing part {i}/{len(parts)}")
            raw = llm.chat([{"role": "system", "content": MAP_PROMPT}, {"role": "user", "content": part}],
                           temperature=0.1, max_tokens=300)
            s = _sections(raw, ["NOTES", "QUOTES"])
            notes += _bullets(s.get("NOTES", raw))
            quote_cands += _bullets(s.get("QUOTES", ""))
        joined = "\n".join(f"- {n}" for n in notes[:60])
        raw = llm.chat([{"role": "system", "content": REDUCE_PROMPT}, {"role": "user", "content": joined}],
                       temperature=0.1, max_tokens=500)
    except LLMError as e:
        return {"executive_summary": f"Summarization failed: {e}", "key_takeaways": [], "risks": [],
                "opportunities": [], "notable_quotes": [], "analyst_signal": "Neutral",
                "signal_justification": "", "_error": str(e), "_connection_error": e.kind == "connection",
                "_model_missing": e.kind == "model_missing"}

    s = _sections(raw, ["SUMMARY", "TAKEAWAYS", "RISKS", "OPPORTUNITIES", "TONE", "SIGNAL", "JUSTIFICATION"])
    tone_raw = (s.get("TONE") or s.get("SIGNAL") or "neutral").strip().split()[0].strip(".,").lower() if (
        s.get("TONE") or s.get("SIGNAL")) else "neutral"
    return {
        "executive_summary": re.sub(r"\s+", " ", s.get("SUMMARY", raw[:500])).strip(),
        "key_takeaways": _bullets(s.get("TAKEAWAYS", ""))[:5],
        "risks": _bullets(s.get("RISKS", ""))[:5],
        "opportunities": _bullets(s.get("OPPORTUNITIES", ""))[:5],
        "notable_quotes": verify_quotes(quote_cands, text),
        "analyst_signal": _TONE_TO_LABEL.get(tone_raw, "Neutral"),
        "signal_justification": re.sub(r"\s+", " ", s.get("JUSTIFICATION", "")).strip(),
        "model_used": model_name,
    }


def check_ollama_status(model: str | None = None) -> dict:
    from src.llm.provider import OllamaLLM
    return OllamaLLM(model).status()

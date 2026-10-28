"""
src/qa/rag.py

Plain RAG answerer (F2, S6). Used as baseline B0 (dense top-5) and B1 (hybrid +
rerank) and as the agent's synthesis step. Hard requirements:
  * every claim cites a retrieved chunk as [c1], [c2]...
  * if the context does not contain the answer, reply exactly INSUFFICIENT_EVIDENCE
"""

from __future__ import annotations

import re

from src.llm.provider import LLMError

ABSTAIN = "INSUFFICIENT_EVIDENCE"
DISCLAIMER = "This reports what was said on earnings calls. It is not investment advice."

SYSTEM_PROMPT = f"""You answer questions about company earnings calls using ONLY the numbered context passages.
Rules:
- Cite every claim with the passage label in square brackets, like [c1] or [c2][c3].
- Quote numbers exactly as written in the passages.
- If the passages do not contain the answer, reply with exactly: {ABSTAIN}
- Do not use outside knowledge. Do not give investment advice.
- Be concise: 1 to 4 sentences."""


def format_context(hits: list[dict]) -> tuple[str, dict[str, str]]:
    """Returns (context_text, label->chunk_id map)."""
    labels, blocks = {}, []
    for i, h in enumerate(hits, 1):
        label = f"c{i}"
        labels[label] = h["chunk_id"]
        who = f" | {h['speaker']}" if h.get("speaker") else ""
        sec = f" | {'Q&A' if h.get('section') == 'qa' else 'prepared' if h.get('section') else ''}".rstrip(" |")
        blocks.append(f"[{label}] {h['ticker']} {h['quarter']} {h['year']}{who}{sec}\n{h['text']}")
    return "\n\n".join(blocks), labels


def extract_citations(answer: str, labels: dict[str, str]) -> list[str]:
    seen, out = set(), []
    for lab in re.findall(r"\[(c\d+)\]", answer):
        if lab in labels and lab not in seen:
            seen.add(lab)
            out.append(labels[lab])
    return out


def is_abstention(answer: str) -> bool:
    return ABSTAIN in answer.upper().replace(" ", "_") or answer.strip() == ""


def answer_question(question: str, retriever, llm, *, mode: str = "hybrid_rerank", k: int = 5,
                    filters: dict | None = None, hits: list[dict] | None = None) -> dict:
    hits = hits if hits is not None else retriever.search(question, k=k, mode=mode, filters=filters)
    if not hits:
        return {"answer": ABSTAIN, "abstained": True, "citations": [], "hits": [], "disclaimer": DISCLAIMER}
    context, labels = format_context(hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {question}"}]
    try:
        answer = llm.chat(messages, temperature=0.0, max_tokens=300).strip()
    except LLMError as e:
        return {"answer": f"LLM unavailable: {e}", "abstained": True, "citations": [], "hits": hits,
                "error": e.kind, "disclaimer": DISCLAIMER}
    abstained = is_abstention(answer)
    return {"answer": ABSTAIN if abstained else answer, "abstained": abstained,
            "citations": [] if abstained else extract_citations(answer, labels),
            "hits": hits, "labels": labels, "disclaimer": DISCLAIMER}

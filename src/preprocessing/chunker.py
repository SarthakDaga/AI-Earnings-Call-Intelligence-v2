"""
src/preprocessing/chunker.py

Chunking v2 (R2.2): 150-200 word windows split on TURN boundaries, each chunk
carrying speaker / role / section, with a context header that is prepended
only for embedding and reranking (stored text stays clean).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.turns import _BOILER_PARA_RE, ParsedTranscript, Turn

SHORT_TURN_WORDS = 20  # turns shorter than this merge into the next turn of the same section


@dataclass
class Chunk:
    text: str
    speaker: str
    role: str
    section: str
    index: int = 0


def context_header(ticker: str, quarter: str, year: int, chunk: Chunk) -> str:
    role = {"executive": "management", "analyst": "analyst", "ir": "IR", "operator": "operator"}.get(chunk.role, "")
    who = f"{chunk.speaker} ({role})" if role else chunk.speaker
    sec = "Q&A" if chunk.section == "qa" else "Prepared remarks"
    return f"{ticker} | {quarter} {year} | {who} | {sec}"


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        out.append(words[start:end])
        if end == len(words):
            break
        start += step
    return out


def chunk_turns(parsed: ParsedTranscript, chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: tuple[Turn, str] | None = None  # short turn waiting to merge forward

    def emit(turn: Turn, text: str) -> None:
        for w in _windows(text.split(), chunk_size, overlap):
            if w:
                chunks.append(Chunk(" ".join(w), turn.speaker, turn.role, turn.section, len(chunks)))

    for t in parsed.turns:
        if t.role == "operator":
            continue
        paras = [p for p in t.paragraphs if not _BOILER_PARA_RE.search(p)]
        text = " ".join(paras).strip()
        if not text:
            continue
        if pending is not None:
            ptxt = pending[1]
            if pending[0].section == t.section:
                text = f"{ptxt} {text}"
            else:
                emit(*pending)
            pending = None
        if len(text.split()) < SHORT_TURN_WORDS:
            pending = (t, text)
            continue
        emit(t, text)
    if pending is not None:
        emit(*pending)
    for i, c in enumerate(chunks):
        c.index = i
    return chunks

"""
src/ingestion/parser.py

v1 cleaner. KEPT UNCHANGED so the A0 baseline stays reproducible (tag it in git).
New code should use src/ingestion/turns.py (parse_transcript / to_clean_text).
"""

import re

SPEAKER_LABEL_PATTERN = re.compile(r"^[A-Z][A-Za-z\s\.\-]+:\s*$", re.MULTILINE)
INLINE_SPEAKER_PATTERN = re.compile(r"^[A-Z][A-Za-z\s\.\-]{2,40}:\s+", re.MULTILINE)

FILLER_PHRASES = [
    r"please go ahead[.,]?",
    r"your (next )?question (please|comes from)[.,]?",
    r"(the )?next question (is )?from[.,]?",
    r"thank you[.,]?\s*(and\s+)?",
    r"(we'll|we will) now (take|open|begin)[^.]*questions[.,]?",
    r"this concludes (today's|the) (conference call|question[^.]*)[.,]?",
    r"ladies and gentlemen[.,]?",
    r"good (morning|afternoon|evening)[.,]?",
    r"(before|at the) (we begin|beginning)[^.]*safe harbor[^.]*\.",
    r"forward[\-\s]looking statements?[^.]*\.",
    r"this call (is|may be) recorded[.,]?",
    r"please (note|be advised)[^.]*\.",
]
FILLER_PATTERN = re.compile("|".join(FILLER_PHRASES), re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(r"[\[\(]\d{2}:\d{2}(:\d{2})?[\]\)]")
SLIDE_PATTERN = re.compile(r"(slide|page)\s+\d+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\n{3,}")


def clean_transcript(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = TIMESTAMP_PATTERN.sub("", text)
    text = SLIDE_PATTERN.sub("", text)
    text = SPEAKER_LABEL_PATTERN.sub("", text)
    text = INLINE_SPEAKER_PATTERN.sub("", text)
    text = FILLER_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def extract_paragraphs(cleaned_text: str) -> list[str]:
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n")]
    return [p for p in paragraphs if len(p) > 40]


def get_text_stats(raw_text: str, cleaned_text: str) -> dict:
    return {
        "raw_chars": len(raw_text),
        "cleaned_chars": len(cleaned_text),
        "reduction_pct": round((1 - len(cleaned_text) / max(len(raw_text), 1)) * 100, 1),
        "raw_words": len(raw_text.split()),
        "cleaned_words": len(cleaned_text.split()),
    }

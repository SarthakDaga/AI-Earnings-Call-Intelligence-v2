"""
scripts/diagnose_format.py

Print the *structure* of a transcript file (not its content) so label formats can be debugged:
the first lines (cut to 60 chars), short stand-alone lines that look like speaker labels, and what the
parser made of the file.

    python scripts/diagnose_format.py data/transcripts/AAPL_Q1_2025.txt
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.turns import parse_transcript  # noqa: E402


def main(path: str) -> None:
    raw = Path(path).read_text(encoding="utf-8")
    lines = raw.replace("\r\n", "\n").split("\n")
    print(f"{path}: {len(lines)} lines, {len(raw.split())} words")
    print("\n-- first 10 non-blank lines (cut to 60 chars) --")
    shown = 0
    for ln in lines:
        if ln.strip():
            print("  " + ln.strip()[:60])
            shown += 1
            if shown == 10:
                break
    short = Counter(ln.strip() for i, ln in enumerate(lines)
                    if ln.strip() and len(ln.split()) <= 8 and len(ln) <= 70 and (i == 0 or not lines[i - 1].strip()))
    print("\n-- most common short lines that follow a blank line (likely labels) --")
    for text, n in short.most_common(12):
        print(f"  x{n:<3d} {text[:60]}")
    p = parse_transcript(raw)
    print(f"\n-- parser result: {len(p.turns)} turns, Q&A found={p.qa_found}, analysts={len(p.analysts)} --")
    for t in p.turns[:8]:
        print(f"  [{t.role:8s}|{t.section:8s}] {t.speaker[:28]:28s} {t.word_count:5d} words")
    print("warnings:", p.warnings or "none")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
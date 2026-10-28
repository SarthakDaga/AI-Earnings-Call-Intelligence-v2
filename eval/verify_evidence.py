"""
eval/verify_evidence.py

Checks that every evidence span in the question set really occurs in its
transcript file (normalized), so a typo in a span can never silently look like
a retrieval failure.

    python -m eval.verify_evidence [--questions eval/questions.jsonl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import load_questions, norm_text  # noqa: E402
from src import config  # noqa: E402


def verify(questions: list[dict], transcripts_dir: Path) -> list[str]:
    errors, cache = [], {}
    for q in questions:
        for e in q["evidence"]:
            name = e.get("transcript", "")
            path = transcripts_dir / f"{name}.txt"
            if not path.exists():
                errors.append(f"{q['id']}: transcript file {path.name} not found")
                continue
            if name not in cache:
                cache[name] = norm_text(path.read_text(encoding="utf-8"))
            if norm_text(e["span"]) not in cache[name]:
                errors.append(f"{q['id']}: span not found in {name}: {e['span'][:70]!r}")
        if q["answerable"] and q["category"] not in ("price_linked",) and not q["evidence"]:
            errors.append(f"{q['id']}: answerable question has no evidence")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions")
    ap.add_argument("--transcripts", default=str(config.TRANSCRIPTS_DIR))
    a = ap.parse_args()
    qs = load_questions(a.questions)
    errs = verify(qs, Path(a.transcripts))
    print(f"{len(qs)} questions checked, {len(errs)} problem(s)")
    for e in errs:
        print("  -", e)
    cats = {}
    for q in qs:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    print("by category:", cats, "| dev/test:", sum(q["split"] == "dev" for q in qs), "/", sum(q["split"] == "test" for q in qs))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())

"""
scripts/validate_corpus.py

Sanity-check the transcript corpus BEFORE trusting any metric built on it:
filename/manifest agreement, in-file header vs manifest (F11), Q&A detection,
analyst turns with missing questions (F16), duplicate periods.

    python scripts/validate_corpus.py [--strict]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.ingestion.loader import load_all_transcripts, load_manifest  # noqa: E402
from src.ingestion.turns import parse_transcript, validate_header  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 on any warning")
    strict = ap.parse_args().strict
    manifest = load_manifest()
    transcripts = load_all_transcripts(manifest_path=config.MANIFEST_PATH)
    problems = 0
    seen = set()
    print(f"\n{'transcript':16s} {'turns':>5s} {'words':>6s} {'qa':>3s} {'analysts':>8s}  notes")
    for t in transcripts:
        key = (t["ticker"], t["quarter"], t["year"])
        p = parse_transcript(t["raw_text"])
        notes = list(p.warnings) + validate_header(p.header, fiscal_quarter=t["quarter"], fiscal_year=t["year"],
                                                    call_date=t["date"])
        if key in seen:
            notes.append("duplicate period")
        seen.add(key)
        if key not in manifest:
            notes.append("not in manifest.csv")
        elif not manifest[key]["call_date"]:
            notes.append("manifest has no call_date")
        problems += len(notes)
        print(f"{t['ticker']}_{t['quarter']}_{t['year']:<8} {len(p.turns):5d} {len(t['raw_text'].split()):6d} "
              f"{'yes' if p.qa_found else 'NO':>3s} {len(p.analysts):8d}  {'; '.join(notes) or 'ok'}")
    for key in manifest:
        if key not in seen:
            print(f"manifest row without a transcript file: {key}")
            problems += 1
    print(f"\n{len(transcripts)} transcripts, {problems} issue(s).")
    return 1 if (strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())

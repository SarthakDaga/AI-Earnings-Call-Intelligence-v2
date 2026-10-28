"""
src/ingestion/loader.py

Loads transcript .txt files named TICKER_Q#_YYYY.txt. Period comes from the
filename; call date and source URL come from data/manifest.csv (R0.3), because
in-file headers are unreliable (F11).

manifest.csv columns: ticker,fiscal_quarter,fiscal_year,call_date,source_url
"""

import csv
import re
from pathlib import Path
from typing import Optional

from src import config

TICKER_TO_COMPANY = {
    "NVDA": "NVIDIA", "MSFT": "Microsoft", "AAPL": "Apple", "GOOGL": "Alphabet",
    "GOOG": "Alphabet", "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
}

TRANSCRIPTS_DIR = config.TRANSCRIPTS_DIR
MANIFEST_PATH = config.MANIFEST_PATH


def parse_filename(filename: str) -> Optional[dict]:
    stem = Path(filename).stem
    match = re.match(r"^([A-Z]+)_(Q[1-4])_(\d{4})$", stem)
    if not match:
        print(f"[Loader] WARNING: Could not parse filename '{filename}'. "
              f"Expected format: TICKER_Q#_YYYY.txt")
        return None
    ticker, quarter, year = match.groups()
    return {"ticker": ticker, "company": TICKER_TO_COMPANY.get(ticker, ticker),
            "quarter": quarter, "year": int(year)}


def load_manifest(path: Path | None = None) -> dict[tuple[str, str, int], dict]:
    """Return {(ticker, quarter, year): row}. Missing manifest => empty dict."""
    path = Path(path or MANIFEST_PATH)
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["ticker"].strip().upper(), row["fiscal_quarter"].strip().upper(),
                       int(row["fiscal_year"]))
            except (KeyError, ValueError):
                continue
            out[key] = {"call_date": (row.get("call_date") or "").strip() or None,
                        "source_url": (row.get("source_url") or "").strip() or None}
    return out


def load_transcript_file(filepath: Path, manifest: dict | None = None) -> Optional[dict]:
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[Loader] File not found: {filepath}")
        return None
    metadata = parse_filename(filepath.name)
    if metadata is None:
        return None
    try:
        raw_text = filepath.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[Loader] Failed to read {filepath.name}: {e}")
        return None
    if len(raw_text) < 100:
        print(f"[Loader] WARNING: {filepath.name} seems too short ({len(raw_text)} chars).")

    manifest = load_manifest() if manifest is None else manifest
    entry = manifest.get((metadata["ticker"], metadata["quarter"], metadata["year"]), {})
    return {**metadata, "raw_text": raw_text, "source": "manual upload",
            "date": entry.get("call_date"), "source_url": entry.get("source_url"),
            "filepath": str(filepath)}


def load_all_transcripts(directory: Path | None = None, manifest_path: Path | None = None) -> list[dict]:
    directory = Path(directory or TRANSCRIPTS_DIR)
    if not directory.exists():
        print(f"[Loader] Transcripts directory not found: {directory}")
        return []
    files = sorted(directory.glob("*.txt"))
    if not files:
        print(f"[Loader] No .txt files found in {directory}")
        return []
    manifest = load_manifest(manifest_path)
    transcripts = []
    for f in files:
        r = load_transcript_file(f, manifest)
        if r:
            transcripts.append(r)
            print(f"[Loader] Loaded: {f.name} -> {r['company']} {r['quarter']} {r['year']} "
                  f"(call date: {r['date'] or 'MISSING'})")
    print(f"[Loader] Total loaded: {len(transcripts)} transcript(s)")
    return transcripts

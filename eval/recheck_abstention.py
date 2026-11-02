"""Read-only: list answers in the latest answers run that look like an abstention (misspelled INSUFFICIENT_EVIDENCE,
etc.) but were NOT counted as abstained. Changes nothing.

    python -m eval.recheck_abstention
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

LOOKS_LIKE = re.compile(r"INSUFF\w*[ _-]*EVID\w*", re.I)


def main() -> None:
    files = sorted((config.EVAL_DIR / "results").glob("*-answers.json"))
    if not files:
        sys.exit("no answers results found")
    res = json.loads(files[-1].read_text(encoding="utf-8"))
    print(f"run {res['run_id']}")
    found = 0
    for key, block in res["rows"].items():
        for r in block["per_question"]:
            ans = r.get("answer") or ""
            if LOOKS_LIKE.search(ans) and not r["abstained"]:
                found += 1
                print(f"{key} {r['id']} [{r['category']}] answerable={r['answerable']} counted as answered: "
                      f"{ans[:90]!r}")
    print(f"\n{found} answer(s) look like an abstention but were counted as answered.")


if __name__ == "__main__":
    main()

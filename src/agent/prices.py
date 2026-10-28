"""
src/agent/prices.py

Free price data via yfinance (an unofficial scraper that may rate-limit cloud
IPs). Lookup order: SQLite cache -> yfinance -> optional snapshot CSV
(data/prices_snapshot.csv with columns ticker,date,close). Every successful
fetch is cached. Failures raise PriceError with a message the agent can relay.

Reaction convention: earnings calls for these companies happen after the
market closes, so the baseline is the last close ON OR BEFORE the call date and
the reaction is the move over the next N trading days.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from src import config
from src.database import queries


class PriceError(RuntimeError):
    pass


def _d(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def fetch_yfinance(ticker: str, start: str, end: str) -> list[tuple[str, float]]:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise PriceError("yfinance is not installed") from e
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    except Exception as e:
        raise PriceError(f"yfinance request failed: {e}") from e
    if df is None or df.empty:
        raise PriceError(f"yfinance returned no data for {ticker} {start}..{end} (rate limit or bad range)")
    return [(idx.strftime("%Y-%m-%d"), float(row["Close"])) for idx, row in df.iterrows()]


def load_snapshot(ticker: str, start: str, end: str, path: Path | None = None) -> list[tuple[str, float]]:
    path = Path(path or config.PRICE_SNAPSHOT_PATH)
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["ticker"].upper() == ticker.upper() and start <= r["date"] <= end:
                rows.append((r["date"], float(r["close"])))
    return sorted(rows)


class PriceProvider:
    def __init__(self, fetcher=None, snapshot_path: Path | None = None):
        self.fetcher = fetcher or fetch_yfinance
        self.snapshot_path = snapshot_path

    def history(self, ticker: str, start: str, end: str) -> list[tuple[str, float]]:
        ticker = ticker.upper()
        cached = queries.get_prices(ticker, start, end)
        if cached and queries.price_fetch_logged(ticker, start, end):
            return cached
        try:
            rows = self.fetcher(ticker, start, (_d(end) + timedelta(days=1)).isoformat())
            queries.upsert_prices(ticker, rows)
            queries.log_price_fetch(ticker, start, end)
            return [(d, c) for d, c in rows if start <= d <= end]
        except PriceError as e:
            if cached:
                return cached
            snap = load_snapshot(ticker, start, end, self.snapshot_path)
            if snap:
                queries.upsert_prices(ticker, snap)
                return snap
            raise PriceError(f"No price data available for {ticker}: {e}") from e

    def reaction(self, ticker: str, call_date: str, windows: tuple[int, ...] = (1, 5)) -> dict:
        cd = _d(call_date)
        start = (cd - timedelta(days=10)).isoformat()
        end = (cd + timedelta(days=max(windows) * 3 + 10)).isoformat()
        rows = self.history(ticker, start, end)
        base_idx = max((i for i, (d, _) in enumerate(rows) if d <= cd.isoformat()), default=None)
        if base_idx is None:
            raise PriceError(f"No close on or before {call_date} for {ticker}")
        base_date, base_close = rows[base_idx]
        out = {"ticker": ticker.upper(), "call_date": call_date, "baseline_date": base_date,
               "baseline_close": round(base_close, 2), "moves": {}}
        for n in windows:
            if base_idx + n >= len(rows):
                out["moves"][f"{n}d"] = None  # window not yet available
                continue
            d, c = rows[base_idx + n]
            out["moves"][f"{n}d"] = {"date": d, "close": round(c, 2),
                                     "pct_change": round((c / base_close - 1) * 100, 2)}
        return out

"""
src/config.py

Single env-driven configuration module (R0.6). Everything that was previously
hard-coded (paths, model names, chunk settings, LLM provider) lives here.

Everything defaults to a fully free, local stack:
  * LLM        -> Ollama running on your machine
  * Embeddings -> sentence-transformers (open weights, CPU)
  * Reranker   -> cross-encoder (open weights, CPU)
  * Prices     -> yfinance (free) with a SQLite cache and optional snapshot CSV
Optional free-tier hosted LLM endpoints (Groq, Google AI Studio, OpenRouter
":free" models) can be enabled purely through environment variables.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load settings from a .env file in the project root (real environment variables win over the file)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


# ── Paths ────────────────────────────────────────────────────
DATA_DIR = Path(_env("EARNINGS_DATA_DIR", str(ROOT / "data")))
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
DB_PATH = DATA_DIR / "earnings.db"
CHROMA_DIR = DATA_DIR / "chroma_db"
PRICE_SNAPSHOT_PATH = DATA_DIR / "prices_snapshot.csv"
EVAL_DIR = ROOT / "eval"

# ── Models ───────────────────────────────────────────────────
EMBED_MODEL = _env("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANKER_SMALL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BASE = "BAAI/bge-reranker-base"
RERANKER_MODEL = _env("RERANKER_MODEL", RERANKER_SMALL)
FINBERT_MODEL = "ProsusAI/finbert"

# ── LLM provider (all free options) ─────────────────────────
# ollama        -> local, free, private (default)
# openai_compat -> any OpenAI-compatible endpoint, e.g. a free tier of
#                  Groq / Google AI Studio / OpenRouter ":free" models
# fake          -> deterministic stub for tests
LLM_PROVIDER = _env("LLM_PROVIDER", "ollama")
LLM_MODEL = _env("LLM_MODEL", "llama3.2:3b")
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
LLM_TIMEOUT = _int("LLM_TIMEOUT", 300)

# Judge: a *different model family* than the generator (llama vs qwen by
# default), still free and local. Override to use a free hosted tier.
JUDGE_PROVIDER = _env("JUDGE_PROVIDER", LLM_PROVIDER)
JUDGE_MODEL = _env("JUDGE_MODEL", "qwen2.5:7b")
JUDGE_BASE_URL = _env("JUDGE_BASE_URL", OPENAI_BASE_URL)
JUDGE_API_KEY = _env("JUDGE_API_KEY", OPENAI_API_KEY)

# ── Serving ──────────────────────────────────────────────────
ADMIN_API_KEY = _env("ADMIN_API_KEY", "")  # empty => write endpoints disabled
ENABLE_UPLOADS = _bool("ENABLE_UPLOADS", False)
RATE_LIMIT_PER_MIN = _int("RATE_LIMIT_PER_MIN", 30)
AGENT_MAX_STEPS = _int("AGENT_MAX_STEPS", 6)
TOOL_TIMEOUT_S = _int("TOOL_TIMEOUT_S", 10)
CORS_ORIGINS = [o for o in _env("CORS_ORIGINS", "*").split(",") if o]
PRELOAD_MODELS = _bool("PRELOAD_MODELS", True)  # warm models in a background thread at API start
RETRIEVAL_CONFIG = _env("RETRIEVAL_CONFIG", "v2")


@dataclass(frozen=True)
class ChunkConfig:
    """
    Everything that changes the stored vectors. The hash of these fields
    names the Chroma collection (R0.2) so experiments never share stale vectors.
    """

    name: str
    strategy: str          # "words" (v1) | "turns" (v2)
    chunk_size: int
    overlap: int
    embed_model: str
    context_header: bool = False
    parser: str = "v1"     # "v1" = clean_transcript, "v2" = turn parser

    @property
    def hash(self) -> str:
        payload = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha1(payload).hexdigest()[:8]

    @property
    def collection(self) -> str:
        return f"chunks_{self.name}_{self.hash}"


CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    # A0 baseline: exactly what v1 did
    "v1": ChunkConfig("v1", "words", 400, 50, "all-MiniLM-L6-v2", False, "v1"),
    # Chunking v2: turn-aware, 180 words, 40 overlap, context header
    "v2": ChunkConfig("v2", "turns", 180, 40, "all-MiniLM-L6-v2", True, "v2"),
    # Optional embedding ablation (accepts longer input)
    "v2bge": ChunkConfig("v2bge", "turns", 180, 40, "BAAI/bge-small-en-v1.5", True, "v2"),
}


def get_chunk_config(name: str | None = None) -> ChunkConfig:
    name = name or RETRIEVAL_CONFIG
    if name not in CHUNK_CONFIGS:
        raise KeyError(f"Unknown chunk config '{name}'. Options: {list(CHUNK_CONFIGS)}")
    return CHUNK_CONFIGS[name]

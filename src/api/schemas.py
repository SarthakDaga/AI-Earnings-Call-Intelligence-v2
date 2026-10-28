"""Pydantic request/response models (OpenAPI docs at /docs)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class Filters(BaseModel):
    ticker: Optional[str] = None
    quarter: Optional[str] = Field(None, pattern=r"^[Qq][1-4]$")
    year: Optional[int] = Field(None, ge=2000, le=2100)
    section: Optional[Literal["prepared", "qa"]] = None
    role: Optional[Literal["executive", "analyst", "ir"]] = Field(
        None, description="Experimental: roles are inferred, source labels are noisy")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: Mode = "hybrid_rerank"
    filters: Optional[Filters] = None


class Hit(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    quarter: str
    year: int
    section: str = ""
    speaker: str = ""
    role: str = ""
    score: float
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


class SearchResponse(BaseModel):
    mode: Mode
    latency_ms: float
    hits: list[Hit]


class AskRequest(SearchRequest):
    query: str = Field(min_length=2, max_length=500, alias="question")

    model_config = {"populate_by_name": True}


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[str]
    hits: list[Hit]
    disclaimer: str
    latency_ms: float


class AgentRequest(BaseModel):
    message: str = Field(min_length=2, max_length=800)
    thread_id: Optional[str] = Field(None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")


class TranscriptOut(BaseModel):
    id: int
    company: str
    ticker: str
    quarter: str
    year: int
    date: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None


class SentimentOut(BaseModel):
    transcript_id: int
    section: Optional[str] = None
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    net_score: float
    top_positive_sentences: list[dict] = []
    top_negative_sentences: list[dict] = []
    sentence_scores: Optional[list[dict]] = None


class SummaryOut(BaseModel):
    status: Literal["ready", "running", "failed", "missing"]
    summary: Optional[dict] = None
    detail: Optional[str] = None
    note: str = "Tone signal describes the language used on the call. It is not investment advice."


class UploadRequest(BaseModel):
    filename: str = Field(pattern=r"^[A-Z]+_Q[1-4]_\d{4}\.txt$")
    text: str = Field(min_length=100)
    call_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class HealthOut(BaseModel):
    status: str
    chunks: int
    transcripts: int
    embedder_loaded: bool
    reranker_loaded: bool
    bm25_ready: bool
    llm_provider: str
    llm_model: str
    retrieval_config: str

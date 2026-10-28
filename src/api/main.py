"""
src/api/main.py

FastAPI app factory. Models load once via the lifespan handler. FinBERT is NOT
loaded at serve time: sentiment is precomputed offline (F7).

    uvicorn src.api.main:app --port 7860
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import config
from src.api.deps import RateLimiter, Services
from src.api.routers import agent, search, transcripts
from src.api.schemas import HealthOut
from src.database import queries
from src.database.schema import init_db

log = logging.getLogger("api")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

EXPENSIVE = ("/v1/ask", "/v1/agent/chat", "/summary")


def build_default_services() -> Services:
    from src.agent import Agent, ToolBox
    from src.llm import get_llm
    from src.retrieval import Retriever
    retriever = Retriever(config.get_chunk_config()).ensure_ready()
    llm = get_llm()
    return Services(retriever=retriever, llm=llm, agent=Agent(ToolBox(retriever), llm))


def _warm_up(r) -> None:
    """Load models + open the vector store off the request path, so the first search is fast.
    Each step is independent: a failure (e.g. reranker not downloaded) is logged, never fatal."""
    from src.retrieval import dense
    steps = (
        ("vector store", lambda: dense.get_collection(r.cfg, r.chroma_path)),
        ("embedder", lambda: r.embedder.encode_query("warm up")),
        ("reranker", lambda: r.reranker.score("warm up", ["warm up"])),
    )
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            fn()
            log.info(f"[warm-up] {name} ready in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            log.info(f"[warm-up] {name} FAILED: {type(e).__name__}: {str(e)[:150]}")
    log.info("[warm-up] done")


def _err(status: int, code: str, message: str, request: Request | None = None) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) if request else None
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": rid}})


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        app.state.services = services or build_default_services()
        if config.PRELOAD_MODELS and services is None:
            threading.Thread(target=_warm_up, args=(app.state.services.retriever,), daemon=True).start()
        yield

    app = FastAPI(title="Earnings Call Intelligence API", version="2.0.0", lifespan=lifespan,
                  description="Hybrid search, cited RAG answers and a LangGraph agent over earnings call "
                              "transcripts. Reports what was said; not investment advice.")
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        ip = request.client.host if request.client else "unknown"
        path = request.url.path
        if path.startswith("/v1/") and request.method != "OPTIONS":
            ok, retry = limiter.check(ip, "expensive" if any(x in path for x in EXPENSIVE) else "default")
            if not ok:
                resp = _err(429, "rate_limited", f"Too many requests. Retry in {retry}s.", request)
                resp.headers["Retry-After"] = str(retry)
                resp.headers["X-Request-ID"] = rid
                return resp
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled")
            response = _err(500, "internal_error", "Internal server error", request)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        log.info(json.dumps({"request_id": rid, "method": request.method, "path": path,
                             "status": response.status_code, "latency_ms": ms, "ip": ip}))
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(request: Request, exc: HTTPException):
        return _err(exc.status_code, f"http_{exc.status_code}", str(exc.detail), request)

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors())
        return _err(422, "validation_error", msg, request)

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health(request: Request):
        svc: Services = request.app.state.services
        st = svc.retriever.status()
        return HealthOut(status="ok", chunks=st["chunks"], transcripts=len(queries.get_all_transcripts()),
                         embedder_loaded=st["embedder_loaded"], reranker_loaded=st["reranker_loaded"],
                         bm25_ready=st["bm25_ready"], llm_provider=getattr(svc.llm, "provider", "?"),
                         llm_model=getattr(svc.llm, "model", "?"), retrieval_config=st["config"])

    app.include_router(transcripts.router)
    app.include_router(search.router)
    app.include_router(agent.router)
    return app


app = create_app()  # services are built lazily in the lifespan handler

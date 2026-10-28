import time

from fastapi import APIRouter, Depends

from src.api.deps import Services, get_services
from src.api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from src.qa.rag import answer_question

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _filters(req) -> dict:
    return req.filters.model_dump(exclude_none=True) if req.filters else {}


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: Services = Depends(get_services)):
    hits, ms = svc.retriever.timed_search(req.query, k=req.k, mode=req.mode, filters=_filters(req))
    return SearchResponse(mode=req.mode, latency_ms=round(ms, 1), hits=hits)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, svc: Services = Depends(get_services)):
    t0 = time.perf_counter()
    r = answer_question(req.query, svc.retriever, svc.llm, mode=req.mode, k=req.k, filters=_filters(req))
    return AskResponse(answer=r["answer"], abstained=r["abstained"], citations=r["citations"],
                       hits=r["hits"], disclaimer=r["disclaimer"],
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))

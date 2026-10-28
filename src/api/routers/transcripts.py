import threading

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Response

from src import config
from src.api.deps import Services, check_admin, get_services, require_admin
from src.api.schemas import SentimentOut, SummaryOut, TranscriptOut, UploadRequest
from src.database import queries
from src.ingestion.loader import parse_filename
from src.ingestion.turns import parse_transcript, to_clean_text
from src.sentiment.finbert import aggregate_by_section
from src.summarization.summarizer import summarize_transcript

router = APIRouter(prefix="/v1/transcripts", tags=["transcripts"])
_index_lock = threading.Lock()


@router.get("", response_model=list[TranscriptOut])
def list_transcripts():
    return queries.get_all_transcripts()


@router.get("/{transcript_id}/sentiment", response_model=SentimentOut)
def get_sentiment(transcript_id: int, section: str | None = None, include_sentences: bool = True):
    if section not in (None, "prepared", "qa"):
        raise HTTPException(422, "section must be 'prepared' or 'qa'")
    if not queries.get_transcript_by_id(transcript_id):
        raise HTTPException(404, "Transcript not found")
    s = queries.get_sentiment(transcript_id)
    if not s:
        raise HTTPException(404, "Sentiment not computed yet. Run scripts/run_pipeline.py "
                                 "(FinBERT is not loaded at serve time).")
    out = {"transcript_id": transcript_id, "section": section, **{
        k: s[k] for k in ("positive_pct", "neutral_pct", "negative_pct", "net_score",
                          "top_positive_sentences", "top_negative_sentences")}}
    if section:
        agg = aggregate_by_section(s["sentence_scores"], section)
        if not agg:
            raise HTTPException(404, f"No '{section}' sentences scored for this transcript")
        out.update({k: agg[k] for k in ("positive_pct", "neutral_pct", "negative_pct", "net_score",
                                        "top_positive_sentences", "top_negative_sentences")})
    if include_sentences:
        out["sentence_scores"] = [x for x in s["sentence_scores"] if not section or x.get("section") == section]
    return out


def _run_summary(svc: Services, transcript_id: int, text: str):
    try:
        r = summarize_transcript(text, llm=svc.llm)
        if r.get("_error"):
            svc.summary_jobs[transcript_id] = f"failed: {r['executive_summary']}"
            return
        queries.insert_summary(transcript_id, r["executive_summary"], r["key_takeaways"], r["risks"],
                               r["opportunities"], r["notable_quotes"], r["analyst_signal"],
                               r["signal_justification"], r.get("model_used", ""))
        svc.summary_jobs.pop(transcript_id, None)
    except Exception as e:
        svc.summary_jobs[transcript_id] = f"failed: {e}"


@router.get("/{transcript_id}/summary", response_model=SummaryOut)
def get_summary(transcript_id: int, svc: Services = Depends(get_services)):
    cached = queries.get_summary(transcript_id)
    if cached:
        return SummaryOut(status="ready", summary=cached)
    job = svc.summary_jobs.get(transcript_id)
    if job == "running":
        return SummaryOut(status="running")
    if job:
        return SummaryOut(status="failed", detail=job)
    return SummaryOut(status="missing")


@router.post("/{transcript_id}/summary", response_model=SummaryOut)
def post_summary(transcript_id: int, background: BackgroundTasks, response: Response, force: bool = False,
                 svc: Services = Depends(get_services), x_api_key: str | None = Header(default=None)):
    """Fetch the cached summary (public). Generating one spends LLM time, so it needs X-API-Key
    and runs as a background job (poll GET .../summary)."""
    t = queries.get_transcript_by_id(transcript_id)
    if not t:
        raise HTTPException(404, "Transcript not found")
    cached = queries.get_summary(transcript_id)
    if cached and not force:
        return SummaryOut(status="ready", summary=cached)
    check_admin(x_api_key)
    if svc.summary_jobs.get(transcript_id) == "running":
        response.status_code = 202
        return SummaryOut(status="running")
    if force:
        queries.delete_summary(transcript_id)
    svc.summary_jobs[transcript_id] = "running"
    background.add_task(_run_summary, svc, transcript_id, t["processed_text"] or t["raw_text"])
    response.status_code = 202
    return SummaryOut(status="running")


def _ingest_and_index(svc: Services, req: UploadRequest):
    from src.retrieval.index import build_index
    meta = parse_filename(req.filename)
    parsed = parse_transcript(req.text)
    tid = queries.insert_transcript(meta["company"], meta["ticker"], meta["quarter"], meta["year"],
                                    req.text, to_clean_text(parsed), date=req.call_date, source="upload")
    with _index_lock:
        t = queries.get_transcript_by_id(tid)
        build_index(svc.retriever.cfg, svc.retriever.embedder, [t], force=True,
                    chroma_path=svc.retriever.chroma_path, progress=lambda *_: None)
        svc.retriever.reload()


@router.post("", status_code=202, dependencies=[Depends(require_admin)])
def upload(req: UploadRequest, background: BackgroundTasks, svc: Services = Depends(get_services)):
    if not config.ENABLE_UPLOADS:
        raise HTTPException(403, "Uploads are disabled on this deployment")
    background.add_task(_ingest_and_index, svc, req)
    return {"status": "accepted", "filename": req.filename,
            "note": "Chunks are indexed in the background. Sentiment is computed offline by scripts/run_pipeline.py."}

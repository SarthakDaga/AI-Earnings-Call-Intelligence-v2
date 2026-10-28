import json

import pytest
from fastapi.testclient import TestClient

from src import config
from src.agent import Agent, ToolBox
from src.agent.prices import PriceProvider
from src.api.deps import Services
from src.api.main import create_app
from src.database import queries
from src.llm import FakeLLM

SUMMARY_SCRIPT = lambda m: ("NOTES:\n- margin 52%\nQUOTES:\n- none" if "ONE part" in m[0]["content"] else  # noqa: E731
                            "SUMMARY:\nGood quarter.\n\nTAKEAWAYS:\n- a\n\nRISKS:\n- r\n\nOPPORTUNITIES:\n- o\n\n"
                            "TONE:\nNeutral\n\nJUSTIFICATION:\nBalanced.")


@pytest.fixture
def client(retriever, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 0)
    llm = FakeLLM(lambda m: "Servo supply was the bottleneck [c1]." if "Context passages" in m[-1]["content"]
                  else SUMMARY_SCRIPT(m))
    svc = Services(retriever=retriever, llm=llm,
                   agent=Agent(ToolBox(retriever, PriceProvider(fetcher=lambda *a: [])), llm))
    with TestClient(create_app(svc)) as c:
        c.svc = svc
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["chunks"] > 0 and r["transcripts"] == 2 and r["bm25_ready"]
    assert r["llm_provider"] == "fake"


def test_openapi_docs_available(client):
    spec = client.get("/openapi.json").json()
    for path in ("/health", "/v1/transcripts", "/v1/search", "/v1/ask", "/v1/agent/chat"):
        assert path in spec["paths"]
    assert client.get("/docs").status_code == 200


def test_request_id_header_roundtrip(client):
    assert client.get("/health", headers={"x-request-id": "abc123"}).headers["x-request-id"] == "abc123"
    assert client.get("/health").headers["x-request-id"]


def test_list_transcripts(client):
    rows = client.get("/v1/transcripts").json()
    assert {(r["ticker"], r["quarter"]) for r in rows} == {("ACME", "Q1"), ("ACME", "Q2")}


def test_sentiment_endpoint(client):
    tid = queries.get_all_transcripts()[0]["id"]
    assert client.get(f"/v1/transcripts/{tid}/sentiment").status_code == 404  # not computed
    sc = [{"text": "a", "label": "positive", "score": .9, "positive": .9, "negative": .05, "neutral": .05, "section": "prepared"},
          {"text": "b", "label": "negative", "score": .9, "positive": .05, "negative": .9, "neutral": .05, "section": "qa"}]
    queries.insert_sentiment(tid, 50, 0, 50, 0.0, [], [], sc)
    full = client.get(f"/v1/transcripts/{tid}/sentiment").json()
    assert full["net_score"] == 0.0 and len(full["sentence_scores"]) == 2
    qa = client.get(f"/v1/transcripts/{tid}/sentiment", params={"section": "qa"}).json()
    assert qa["net_score"] < 0 and len(qa["sentence_scores"]) == 1
    assert client.get(f"/v1/transcripts/{tid}/sentiment", params={"section": "zzz"}).status_code == 422
    assert client.get("/v1/transcripts/9999/sentiment").status_code == 404


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_search_modes(client, mode):
    r = client.post("/v1/search", json={"query": "servo motor supply", "k": 3, "mode": mode}).json()
    assert r["mode"] == mode and r["hits"] and r["latency_ms"] >= 0
    assert "servo" in " ".join(h["text"].lower() for h in r["hits"])


def test_search_filters(client):
    r = client.post("/v1/search", json={"query": "revenue", "filters": {"quarter": "Q2", "section": "qa"}}).json()
    assert r["hits"] and all(h["quarter"] == "Q2" and h["section"] == "qa" for h in r["hits"])


def test_validation_error_uses_error_model(client):
    r = client.post("/v1/search", json={"query": "x", "mode": "bogus"})
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error" and body["request_id"]


def test_ask_returns_cited_answer(client):
    r = client.post("/v1/ask", json={"question": "What was the supply bottleneck?", "mode": "hybrid"}).json()
    assert not r["abstained"] and r["citations"] and "not investment advice" in r["disclaimer"]


def test_ask_abstains(client):
    client.svc.llm.script = ["INSUFFICIENT_EVIDENCE"]
    r = client.post("/v1/ask", json={"question": "What is the CEO's favourite colour?"}).json()
    assert r["abstained"] and r["citations"] == []


def test_agent_sse_stream(client):
    with client.stream("POST", "/v1/agent/chat", json={"message": "What about servo supply?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "route" and "tool_call" in kinds and "answer" in kinds and kinds[-1] == "done"


def test_agent_thread_id_validated(client):
    assert client.post("/v1/agent/chat", json={"message": "hello there", "thread_id": "../x"}).status_code == 422


def test_summary_requires_key_to_generate(client, monkeypatch):
    tid = queries.get_all_transcripts()[0]["id"]
    assert client.get(f"/v1/transcripts/{tid}/summary").json()["status"] == "missing"
    monkeypatch.setattr(config, "ADMIN_API_KEY", "")
    assert client.post(f"/v1/transcripts/{tid}/summary").status_code == 403  # disabled
    monkeypatch.setattr(config, "ADMIN_API_KEY", "secret")
    assert client.post(f"/v1/transcripts/{tid}/summary").status_code == 401
    r = client.post(f"/v1/transcripts/{tid}/summary", headers={"X-API-Key": "secret"})
    assert r.status_code == 202
    ready = client.get(f"/v1/transcripts/{tid}/summary").json()  # TestClient runs background tasks inline
    assert ready["status"] == "ready" and ready["summary"]["executive_summary"] == "Good quarter."
    assert "not investment advice" in ready["note"]
    assert client.post(f"/v1/transcripts/{tid}/summary").json()["status"] == "ready"  # public once cached


def test_upload_disabled_by_default(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "secret")
    body = {"filename": "ACME_Q3_2025.txt", "text": "Operator: hello " * 20}
    assert client.post("/v1/transcripts", json=body, headers={"X-API-Key": "secret"}).status_code == 403
    monkeypatch.setattr(config, "ENABLE_UPLOADS", True)
    assert client.post("/v1/transcripts", json=body).status_code == 401
    assert client.post("/v1/transcripts", json=body, headers={"X-API-Key": "secret"}).status_code == 202
    assert len(client.get("/v1/transcripts").json()) == 3


def test_upload_filename_validated(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "secret")
    r = client.post("/v1/transcripts", json={"filename": "bad.txt", "text": "x" * 200}, headers={"X-API-Key": "secret"})
    assert r.status_code == 422


def test_rate_limit(retriever, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 4)
    llm = FakeLLM(["INSUFFICIENT_EVIDENCE"])
    svc = Services(retriever=retriever, llm=llm, agent=Agent(ToolBox(retriever), llm))
    with TestClient(create_app(svc)) as c:
        codes = [c.get("/v1/transcripts").status_code for _ in range(6)]
        assert codes[:4] == [200] * 4 and codes[4] == 429
        assert c.get("/health").status_code == 200  # health is never limited
        r = c.get("/v1/transcripts")
        assert r.headers["retry-after"] and r.json()["error"]["code"] == "rate_limited"


def test_expensive_paths_have_smaller_budget(retriever, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 9)  # expensive budget = 3
    svc = Services(retriever=retriever, llm=FakeLLM(["INSUFFICIENT_EVIDENCE"]),
                   agent=Agent(ToolBox(retriever), FakeLLM(["x"])))
    with TestClient(create_app(svc)) as c:
        codes = [c.post("/v1/ask", json={"question": "anything at all"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]

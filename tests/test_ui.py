"""Executes every Streamlit page headlessly against a live (in-process) API server."""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from src import config
from src.agent import Agent, ToolBox
from src.agent.prices import PriceProvider
from src.api.deps import Services
from src.api.main import create_app
from src.database import queries
from src.llm import FakeLLM

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def live_api(retriever, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MIN", 0)
    llm = FakeLLM(lambda m: "Servo supply was the main bottleneck [c1].")
    svc = Services(retriever=retriever, llm=llm, agent=Agent(ToolBox(retriever, PriceProvider(fetcher=lambda *a: [])), llm))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    monkeypatch.setenv("API_URL", f"http://127.0.0.1:{port}")
    yield port
    server.should_exit = True
    th.join(timeout=5)


@pytest.mark.parametrize("page", ["Home", "Sentiment", "Summary", "Search", "Ask the analyst"])
def test_every_page_renders(live_api, page):
    tid = queries.get_all_transcripts()[0]["id"]
    queries.insert_sentiment(tid, 50, 40, 10, 0.4, [{"text": "great", "score": .9}], [{"text": "bad", "score": .8}],
                             [{"text": "a", "label": "positive", "score": .9, "positive": .9, "negative": .05,
                               "neutral": .05, "section": "prepared"}] * 6)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_search_page_returns_hits_and_compare(live_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Search").run()
    at.text_input[0].set_value("servo motor supply").run()
    assert not at.exception and any("servo" in m.value.lower() for m in at.markdown)
    at.checkbox[0].check().run()
    assert not at.exception


def test_api_down_gives_friendly_error(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_agent_chat_streams_trace_and_cites(live_api):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Ask the analyst").run()
    at.chat_input[0].set_value("What did they say about servo supply?").run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "bottleneck" in text and "ACME Q" in text  # answer + citation card rendered


def test_dollar_amounts_are_not_rendered_as_latex(live_api):
    """Real run: '$22.6 billion ... $30.8 billion' became a LaTeX formula and lost its dollar signs.
    Every card that shows transcript/model text must write '$' as an HTML entity."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Search").run()
    at.text_input[0].set_value("revenue").run()
    cards = [m.value for m in at.markdown if 'class="card"' in m.value]
    assert cards
    assert all("$" not in c for c in cards)       # raw dollar signs never reach st.markdown unescaped
    assert any("&#36;" in c for c in cards)       # ...and the amounts are still shown (as an HTML entity)

import json

import pytest

from src.agent import Agent, PriceError, PriceProvider, ToolBox, classify, verify_answer
from src.agent.tools import RunContext
from src.database import queries
from src.llm import FakeLLM
from src.qa.rag import ABSTAIN


def fake_fetch(ticker, start, end):
    """Business-day closes growing 1% per day: +1d is +1.00%, +5d is +5.10%."""
    from datetime import date, timedelta
    d, px, rows = date.fromisoformat(start), 100.0, []
    while d < date.fromisoformat(end):
        if d.weekday() < 5:
            rows.append((d.isoformat(), px))
            px *= 1.01
        d += timedelta(days=1)
    return rows


@pytest.fixture
def toolbox(retriever, loaded):
    for t in queries.get_all_transcripts():
        queries.insert_sentiment(t["id"], 40.0, 50.0, 10.0, 0.30 if t["quarter"] == "Q1" else 0.1, [], [],
                                 [{"text": "x", "label": "positive", "score": .9, "positive": .9,
                                   "negative": .05, "neutral": .05, "section": "qa"}])
    queries.insert_transcript("Acme", "ACME", "Q1", 2025, "raw " * 50, "x", date="2025-04-30")
    return ToolBox(retriever, PriceProvider(fetcher=fake_fetch, snapshot_path=loaded / "none.csv"))


def scripted(actions, synth):
    """Agent actions are served in order; synthesis returns `synth`."""
    it = iter(actions)

    def fn(messages):
        if "research agent" in messages[0]["content"]:
            try:
                return json.dumps(next(it))
            except StopIteration:
                return json.dumps({"action": "final"})
        return synth if isinstance(synth, str) else synth(messages)
    return fn


def test_routing():
    assert classify("What was the Q1 revenue guidance?") == "rag"
    assert classify("How did the stock move after the call?") == "agent"
    assert classify("Compare sentiment across quarters") == "agent"


def test_rag_route_cites_and_skips_tool_loop(toolbox):
    llm = FakeLLM(lambda m: "Servo supply was the main bottleneck [c1].")
    r = Agent(toolbox, llm).run("What was the bottleneck for servo motor supply?")
    assert r["route"] == "rag" and r["steps"] == 1 and not r["abstained"]
    assert r["citations"] and r["citations"][0]["label"] == "c1"


def test_full_multi_step_trace(toolbox):
    actions = [
        {"action": "list_transcripts", "args": {}},
        {"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME", "quarter": "Q1", "year": 2025}},
        {"action": "get_sentiment", "args": {"ticker": "ACME", "quarter": "Q1", "year": 2025}},
        {"action": "get_price_reaction", "args": {"ticker": "ACME", "quarter": "Q1", "year": 2025}},
        {"action": "final"},
    ]
    synth = "Management said servo supply was tight [c1]. Net sentiment was 0.3 and the stock moved +5.10% in 5 days."
    r = Agent(toolbox, FakeLLM(scripted(actions, synth))).run("How did tone on supply change and how did the stock move?")
    assert r["tools_used"] == ["list_transcripts", "search_transcripts", "get_sentiment", "get_price_reaction"]
    assert not r["abstained"] and r["citations"]
    assert any(e["type"] == "verify" and e["ok"] for e in r["trace"])
    pr = [e for e in r["trace"] if e["type"] == "tool_result" and "Price reaction" in e["result"]][0]
    assert "+1.00%" in pr["result"] and "+5.10%" in pr["result"]


def test_step_cap_is_enforced(toolbox):
    n = {"i": 0}

    def forever(messages):
        if "research agent" in messages[0]["content"]:
            n["i"] += 1
            extra = [{}, {"quarter": "Q1"}, {"quarter": "Q2"}][n["i"] % 3]  # same filters >2 times counts as repeating
            return json.dumps({"action": "search_transcripts", "args": {"query": f"servo supply {n['i']}", **extra}})
        return "Supply was tight [c1]."
    r = Agent(toolbox, FakeLLM(forever), max_steps=6).run("Compare how supply changed across quarters")
    assert r["steps"] == 6 and len(r["tools_used"]) == 6
    assert any(e["type"] == "limit" for e in r["trace"])


def test_verifier_rejects_invented_numbers_then_abstains(toolbox):
    actions = [{"action": "search_transcripts", "args": {"query": "revenue guidance"}}, {"action": "final"}]
    r = Agent(toolbox, FakeLLM(scripted(actions, "Revenue will hit $99 billion next year [c1]."))).run(
        "Compare revenue guidance across quarters")
    assert r["abstained"] and r["answer"] == ABSTAIN and r["citations"] == []
    assert sum(1 for e in r["trace"] if e["type"] == "verify") == 2  # draft + one retry


def test_verifier_retry_can_recover(toolbox):
    drafts = iter(["Guidance was $99 billion [c1].", "Second quarter guidance was $4.5 billion [c1]."])
    actions = [{"action": "search_transcripts", "args": {"query": "second quarter revenue guidance"}}, {"action": "final"}]
    r = Agent(toolbox, FakeLLM(scripted(actions, lambda m: next(drafts)))).run("Compare guidance across quarters")
    assert not r["abstained"] and "$4.5 billion" in r["answer"]


def test_bad_args_and_unknown_tools_become_observations(toolbox):
    ctx = RunContext()
    assert "invalid arguments" in toolbox.call("get_sentiment", {"ticker": "ACME", "quarter": "Q9", "year": 2025}, ctx)
    assert "unknown tool" in toolbox.call("rm_rf", {}, ctx)
    assert "ERROR" in toolbox.call("get_sentiment", {"ticker": "ZZZ", "quarter": "Q1", "year": 2025}, ctx)


def test_tool_timeout(toolbox):
    import time
    toolbox.timeout = 1
    toolbox._list_transcripts = lambda a, c: time.sleep(3)
    assert "timed out" in toolbox.call("list_transcripts", {}, RunContext())


def test_followup_uses_thread_history(toolbox):
    llm = FakeLLM(lambda m: "Supply was tight [c1].")
    agent = Agent(toolbox, llm)
    agent.run("What did they say about servo supply?", thread_id="t1")
    state = agent.graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert state["history"] and state["history"][0]["q"].startswith("What did they say")


def test_no_evidence_searches_once_then_abstains(toolbox):
    """Model says 'final' with nothing gathered: the agent searches with the question, and the synthesizer
    (which only sees irrelevant passages) abstains."""
    r = Agent(toolbox, FakeLLM(scripted([{"action": "final"}], ABSTAIN))).run("Compare the weather across quarters")
    assert r["abstained"] and r["tools_used"] == ["search_transcripts"]


def test_agent_recovers_when_model_misuses_list_transcripts(toolbox):
    """Real llama3.2:3b trace: it called list_transcripts five times with search arguments and never searched."""
    misuse = [{"action": "list_transcripts", "args": {"query": "servo supply", "ticker": "ACME", "k": "10"}},
              {"action": "list_transcripts", "args": {"year": "2025", "section": "prepared"}},
              {"action": "list_transcripts", "args": {"quarter": "Q1"}}]
    r = Agent(toolbox, FakeLLM(scripted(misuse, "Servo supply was tight [c1]."))).run(
        "How did servo supply change across quarters?")
    assert not r["abstained"] and r["citations"]
    assert any("only accepts ticker and year" in e for e in r["evidence"])   # the model was told what to do instead
    kinds = [e["type"] for e in r["trace"]]
    assert "limit" in kinds                                                   # fallback announced in the trace


def test_rag_route_filters_to_the_company_in_the_question(toolbox):
    r = Agent(toolbox, FakeLLM(lambda m: "Servo supply was tight [c1].")).run("What did Acme say about servo supply?")
    call = next(e for e in r["trace"] if e["type"] == "tool_call")
    assert call["args"].get("ticker") == "ACME"


def test_sanitize_args_fixes_small_model_mistakes():
    from src.agent.tools import sanitize_args
    a = sanitize_args("get_price_reaction", {"ticker": "NVIDIA", "quarter": "Q3 2025", "windows": "[1, 5]",
                                             "call_date": "[]", "junk": 1})
    assert a == {"ticker": "NVDA", "quarter": "Q3", "year": 2025, "windows": [1, 5]}
    s = sanitize_args("search_transcripts", {"query": "margins", "ticker": "all", "quarter": "[]", "year": "FY2025",
                                             "section": "Q&A", "k": "10"})
    assert s == {"query": "margins", "year": 2025, "section": "qa", "k": 8}
    assert sanitize_args("get_sentiment", {"ticker": "aapl", "quarter": "3", "year": "2024"}) == \
        {"ticker": "AAPL", "quarter": "Q3", "year": 2024}


def test_infer_ticker():
    from src.agent.tools import infer_ticker
    known = {"NVDA", "MSFT", "AAPL"}
    assert infer_ticker("How did NVIDIA's demand change?", known) == "NVDA"
    assert infer_ticker("What did msft say about Azure?", known) == "MSFT"
    assert infer_ticker("Compare Apple and NVIDIA", known) is None     # two companies: no single filter
    assert infer_ticker("What about supply constraints?", known) is None
    assert infer_ticker("Tell me about Amazon", known) is None          # not in the loaded corpus


def test_list_transcripts_filters(toolbox):
    out = toolbox.call("list_transcripts", {"ticker": "acme"}, RunContext())
    assert out.startswith("ACME") and "AAPL" not in out
    assert toolbox.call("list_transcripts", {"query": "x"}, RunContext()).startswith("ERROR")


def test_stream_yields_events_and_done(toolbox):
    llm = FakeLLM(lambda m: "Servo supply was tight [c1].")
    kinds = [e["type"] for e in Agent(toolbox, llm).stream("What about servo supply?")]
    assert kinds[0] == "route" and "answer" in kinds and kinds[-1] == "done"


def test_verify_answer_unit():
    ev = "Revenue was $4.5 billion, up 18%."
    assert verify_answer("Revenue was $4.5 billion [c1].", ev, "q") == []
    assert verify_answer("Revenue was $5 billion.", ev, "q")
    assert verify_answer('He said "this quote is invented entirely".', ev, "q")
    assert verify_answer("In Q3 it grew.", ev, "q") == []          # 'Q3' is not a number claim
    assert verify_answer("Over 2025 it grew 18%.", ev, "What happened in 2025?") == []


def test_price_cache_snapshot_and_errors(loaded):
    calls = {"n": 0}

    def counting(t, s, e):
        calls["n"] += 1
        return fake_fetch(t, s, e)
    p = PriceProvider(fetcher=counting, snapshot_path=loaded / "none.csv")
    p.reaction("ACME", "2025-04-30")
    p.reaction("ACME", "2025-04-30")
    assert calls["n"] == 1  # second call served from SQLite cache

    snap = loaded / "snap.csv"
    snap.write_text("ticker,date,close\nZZZ,2025-04-29,10\nZZZ,2025-04-30,11\nZZZ,2025-05-01,12.1\n")

    def failing(t, s, e):
        raise PriceError("rate limited")
    r = PriceProvider(fetcher=failing, snapshot_path=snap).reaction("ZZZ", "2025-04-30", (1,))
    assert r["moves"]["1d"]["pct_change"] == 10.0
    with pytest.raises(PriceError):
        PriceProvider(fetcher=failing, snapshot_path=loaded / "none.csv").history("QQQ", "2025-01-01", "2025-01-10")


def test_verifier_ignores_digits_inside_product_names():
    """Real run: 'GB200' was read as the number '00', and a good NVIDIA answer was thrown away."""
    ev = "[c1] NVDA Q4 2025: Blackwell ramped."
    assert verify_answer("Demand for GB200, H100 and H20 systems grew [c1].", ev, "How did demand change?") == []
    assert verify_answer("Revenue grew 45% [c1].", "revenue grew 40%", "q")           # real mismatches still caught


def test_repeating_model_stops_early_and_still_answers(toolbox):
    """Real run: after one search and one trend call the 3B model repeated itself until the step cap."""
    same = {"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME"}}
    r = Agent(toolbox, FakeLLM(scripted([same] * 6, "Servo supply was tight [c1]."))).run(
        "How did servo supply change across quarters?")
    assert not r["abstained"] and r["citations"]
    assert r["steps"] < 6                                   # did not burn the whole budget repeating
    assert any(e["type"] == "limit" and "repeating" in e["message"] for e in r["trace"])


def test_infer_call_year():
    from src.agent.tools import infer_call_year
    assert infer_call_year("How did NVIDIA's demand change across 2025?") == 2025
    assert infer_call_year("What did they say in 2024 about supply?") == 2024
    assert infer_call_year("How did Q3 2025 compare?") is None          # fiscal label: leave alone
    assert infer_call_year("fiscal 2025 guidance") is None
    assert infer_call_year("What about supply constraints?") is None


def test_call_year_filters_by_call_date_and_evidence_shows_dates(toolbox):
    ctx = RunContext()
    out = toolbox.call("search_transcripts", {"query": "servo motor supply", "ticker": "ACME", "call_year": 2025}, ctx)
    assert out.startswith("[c1]") and "call 2025-04-30" in out
    assert "Q2 2025" not in out                                          # Q2 has no 2025 call date in the fixtures
    assert toolbox.call("search_transcripts", {"query": "servo", "call_year": 2019}, RunContext()) == \
        "No transcripts have a call date in 2019."


def test_bare_year_in_question_means_calendar_year(toolbox):
    """Real run: 'across 2025' was answered from fiscal-2025 calls (calendar 2024) and labelled 'Q1 2025'."""
    searched = {"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME", "year": 2025}}
    r = Agent(toolbox, FakeLLM(scripted([searched], "Servo supply was tight [c1]."))).run(
        "How did servo supply change across 2025?")
    call = next(e for e in r["trace"] if e["type"] == "tool_call")
    assert call["args"]["call_year"] == 2025 and "year" not in call["args"]


def test_call_year_search_draws_from_every_call_in_that_year(toolbox):
    """Real run: 'across 2025' was answered from one call. Each call in the year must contribute, oldest first."""
    from src.database.schema import get_connection
    conn = get_connection()
    conn.execute("UPDATE transcripts SET date='2025-07-31' WHERE quarter='Q2'")
    conn.commit()
    conn.close()
    out = toolbox.call("search_transcripts", {"query": "servo motor supply", "ticker": "ACME", "call_year": 2025,
                                              "k": 2}, RunContext())
    lines = [ln for ln in out.split("\n") if ln.startswith("[c")]
    assert len(lines) == 2
    assert "ACME call 2025-04-30 (fiscal Q1 2025)" in lines[0] and "ACME call 2025-07-31 (fiscal Q2 2025)" in lines[1]


def test_calendar_year_question_rejects_fiscal_labels_in_answer():
    ev = "[c1] NVDA call 2025-11-19 (fiscal Q3 2026): data center revenue was $51 billion."
    bad = verify_answer("In Q3 2025 data center revenue was $51 billion [c1].", ev, "How did demand change across 2025?")
    assert any("fiscal labels" in p for p in bad)
    ok = verify_answer("On the 2025-11-19 call data center revenue was $51 billion [c1].", ev,
                       "How did demand change across 2025?")
    assert ok == []
    # a user who asks in fiscal labels may get fiscal labels back
    assert verify_answer("Q3 2025 revenue was $51 billion.", ev, "How did Q3 2025 compare?") == []


def test_same_price_call_with_different_windows_is_a_duplicate(toolbox):
    first = {"action": "get_price_reaction", "args": {"ticker": "ACME", "quarter": "Q1", "year": 2025}}
    again = {"action": "get_price_reaction", "args": {"ticker": "ACME", "quarter": "Q1", "year": 2025,
                                                      "windows": "[1, 5]"}}
    r = Agent(toolbox, FakeLLM(scripted([first, again, again], "The stock moved +5.10% in 5 days."))).run(
        "How did the stock move after the call?")
    assert r["tools_used"].count("get_price_reaction") == 1
    assert any(e["type"] == "limit" and "repeating" in e["message"] for e in r["trace"])


def test_price_tools_are_skipped_when_question_is_not_about_the_stock(toolbox):
    actions = [{"action": "get_price_reaction", "args": {"ticker": "ACME", "quarter": "Q1", "year": 2025}},
               {"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME"}},
               {"action": "final"}]
    r = Agent(toolbox, FakeLLM(scripted(actions, "Servo supply was tight [c1]."))).run(
        "How did servo supply change across 2025?")
    assert "get_price_reaction" not in r["tools_used"] and "search_transcripts" in r["tools_used"]
    assert not r["abstained"]


def test_fiscal_label_only_problem_keeps_answer_with_legend(toolbox):
    """Real run: the 3B model kept writing 'Q3 2025'; the whole answer was discarded. A labelling problem must
    not cost the user a correct answer: keep it and add a call-date legend."""
    actions = [{"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME"}},
               {"action": "final"}]
    r = Agent(toolbox, FakeLLM(scripted(actions, "In Q1 2025 servo supply was tight [c1]."))).run(
        "How did servo supply change across 2025?")
    assert not r["abstained"] and "Labels such as" in r["answer"] and "call 2025-04-30" in r["answer"]
    assert sum(1 for e in r["trace"] if e["type"] == "verify") == 2   # draft + one retry, then kept
    assert r["citations"]


def test_retry_feedback_lists_allowed_call_dates(toolbox):
    seen = []

    def synth(m):
        seen.append(m[-1]["content"])
        return "In Q1 2025 servo supply was tight [c1]."
    actions = [{"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME"}},
               {"action": "final"}]
    Agent(toolbox, FakeLLM(scripted(actions, synth))).run("How did servo supply change across 2025?")
    assert any("Call dates of your sources" in x and "2025-04-30" in x for x in seen)


def test_sentiment_trend_is_date_first(toolbox):
    out = toolbox.call("get_sentiment_trend", {"ticker": "ACME"}, RunContext())
    assert "call 2025-04-30 (fiscal Q1 2025): net_score=" in out


def test_date_must_match_the_passage_it_is_cited_to():
    """Real run (mistral): 'On the 2025-08-27 call ... [c6]' where c6 was the 2025-02-26 call, marked verified."""
    evd = {"c1": {"text": "Revenue was $4.5 billion.", "ticker": "ACME", "quarter": "Q1", "year": 2025}}
    dates = {("ACME", "Q1", 2025): "2025-04-30"}
    ev = "[c1] ACME call 2025-04-30 (fiscal Q1 2025): Revenue was $4.5 billion. Also 2025-08-27."
    bad = verify_answer("On the 2025-08-27 call revenue was $4.5 billion [c1].", ev, "q", evd, dates)
    assert any("does not match" in p for p in bad)
    assert verify_answer("On the 2025-04-30 call revenue was $4.5 billion [c1].", ev, "q", evd, dates) == []


def test_number_cited_to_wrong_passage_is_a_soft_problem():
    evd = {"c1": {"text": "Revenue was $4.5 billion.", "ticker": "A", "quarter": "Q1", "year": 2025},
           "c2": {"text": "Margin was 52 percent.", "ticker": "A", "quarter": "Q1", "year": 2025}}
    ev = "[c1] Revenue was $4.5 billion.\n[c2] Margin was 52 percent."
    p = verify_answer("Revenue was $4.5 billion [c2].", ev, "q", evd, {})
    assert p and all(x.startswith("citation mismatch") for x in p)
    assert verify_answer("Revenue was $4.5 billion [c1]. Margin was 52 percent [c2].", ev, "q", evd, {}) == []


def test_wrong_call_date_abstains_after_one_retry(toolbox):
    actions = [{"action": "search_transcripts", "args": {"query": "servo supply", "ticker": "ACME"}},
               {"action": "final"}]
    r = Agent(toolbox, FakeLLM(scripted(actions, "On the 2025-04-04 call servo supply was tight [c1]."))).run(
        "How did servo supply change across 2025?")
    assert r["abstained"]
    v = [e for e in r["trace"] if e["type"] == "verify"]
    assert len(v) == 2 and any("does not match" in x for x in v[0]["problems"])


def test_reworded_searches_with_same_filters_stop_after_two(toolbox):
    acts = [{"action": "search_transcripts", "args": {"query": f"servo supply {i}", "ticker": "ACME"}}
            for i in range(5)]
    r = Agent(toolbox, FakeLLM(scripted(acts, "Servo supply was tight [c1]."))).run(
        "How did servo supply change across 2025?")
    assert r["tools_used"].count("search_transcripts") == 2 and not r["abstained"]
    assert any(e["type"] == "limit" and "repeating" in e["message"] for e in r["trace"])


def test_infer_fiscal_period():
    from src.agent.tools import infer_fiscal_period as f
    assert f("What did NVIDIA say about supply in Q3 2025?") == ("Q3", 2025)
    assert f("fiscal 2025 Q3 guidance") == ("Q3", 2025)
    assert f("Q4 FY26 margins") == ("Q4", 2026)
    assert f("Compare Q1 2025 and Q2 2025") is None          # two periods: no single filter
    assert f("How did demand change across 2025?") is None   # calendar year, handled by call_year


def test_fiscal_label_in_question_filters_the_search(toolbox):
    r = Agent(toolbox, FakeLLM(lambda m: "Servo supply was tight [c1].")).run(
        "What did Acme say about servo supply in Q1 2025?")
    call = next(e for e in r["trace"] if e["type"] == "tool_call")
    assert call["args"]["quarter"] == "Q1" and call["args"]["year"] == 2025
    assert all(c["quarter"] == "Q1" for c in r["citations"])


def test_period_that_is_not_loaded_abstains(toolbox):
    """Trap questions: 'Q4 2025' is not in the corpus, so the answer must be an abstention, not another quarter."""
    r = Agent(toolbox, FakeLLM(lambda m: "Servo supply was tight [c1].")).run(
        "What did Acme say about servo supply in Q4 2025?")
    assert r["abstained"] and r["citations"] == []

"""
app/streamlit_app.py  -  thin client for the Earnings Intelligence API.

    uvicorn src.api.main:app --port 7860      # terminal 1
    streamlit run app/streamlit_app.py        # terminal 2

Set API_URL if the API is elsewhere. ADMIN_API_KEY (optional) enables the
"Generate summary" button against a locked-down API.
"""

import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:  # the UI runs in its own process, so it reads the same .env as the API (ADMIN_API_KEY, API_URL)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

API_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")
NOT_ADVICE = "Reports what was said on earnings calls. Not investment advice."

st.set_page_config(page_title="Earnings Intelligence", page_icon="📊", layout="wide")
st.markdown("""<style>
.stApp{background:#0d1117;color:#e6edf3}[data-testid=stSidebar]{background:#161b22;border-right:1px solid #30363d}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.1rem;margin:.5rem 0}
.meta{font-family:monospace;font-size:.72rem;color:#8b949e;margin-bottom:.4rem}
.badge{display:inline-block;border:1px solid #d29922;color:#d29922;padding:.2rem .8rem;border-radius:2rem;font-family:monospace;font-size:.82rem}
.badge.pos{border-color:#3fb950;color:#3fb950}.badge.neg{border-color:#f85149;color:#f85149}
.step{border-left:3px solid #388bfd;padding:.3rem .8rem;margin:.3rem 0;font-size:.85rem;color:#c9d1d9}
.step.err{border-color:#f85149}.step.ok{border-color:#3fb950}
.cite{border-left:3px solid #8b949e;padding:.3rem .8rem;margin:.3rem 0;font-size:.82rem;color:#8b949e}
</style>""", unsafe_allow_html=True)


# ── API helpers ──
def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kw.pop("timeout", 180), **kw)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with: uvicorn src.api.main:app --port 7860")
        st.stop()
    except requests.Timeout:
        return None, ("The API did not answer in time. If you just started it, it is still loading models "
                      "(watch its terminal for '[warm-up] done'); try again in a minute.")
    if r.status_code >= 400:
        msg = r.json().get("error", {}).get("message", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        return None, f"{r.status_code}: {msg}"
    return r.json(), None


@st.cache_data(ttl=30)
def get_transcripts():
    data, err = api("GET", "/v1/transcripts")
    return data or []


def transcript_picker():
    ts = get_transcripts()
    if not ts:
        st.info("No transcripts loaded. Add files to data/transcripts and run scripts/run_pipeline.py.")
        st.stop()
    labels = {f"{t['ticker']} {t['quarter']} {t['year']}": t for t in ts}
    return labels[st.sidebar.selectbox("Transcript", list(labels))]


def safe(text) -> str:
    """Text going inside an HTML block: escape markup, and $ (Streamlit would render $...$ as a LaTeX formula)."""
    return html.escape(str(text), quote=False).replace("$", "&#36;")


def md(text) -> str:
    """Text going to st.markdown: a literal dollar sign must be escaped or \"$22.6 billion ... $30.8 billion\" becomes math."""
    return str(text).replace("$", "\\$")


def hit_card(h, show_ranks=True):
    who = f" · {h['speaker']}" if h.get("speaker") else ""
    sec = f" · {'Q&A' if h['section'] == 'qa' else 'prepared'}" if h.get("section") else ""
    ranks = f" · dense#{h['dense_rank'] or '-'} bm25#{h['bm25_rank'] or '-'}" if show_ranks else ""
    st.markdown(f'<div class="card"><div class="meta">{h["ticker"]} {h["quarter"]} {h["year"]}{who}{sec}'
                f' · score {h["score"]:.3f}{ranks}</div>{safe(h["text"])}</div>', unsafe_allow_html=True)


# ── sidebar ──
st.sidebar.markdown("## 📊 Earnings Intelligence")
page = st.sidebar.radio("Page", ["Home", "Sentiment", "Summary", "Search", "Ask the analyst"],
                        label_visibility="collapsed")
health, herr = api("GET", "/health")
if health:
    st.sidebar.caption(f"{health['transcripts']} transcripts · {health['chunks']} chunks · "
                       f"{health['llm_provider']}:{health['llm_model']}")
st.sidebar.caption(NOT_ADVICE)

# ── pages ──
if page == "Home":
    st.title("Earnings Call Intelligence")
    st.caption("Hybrid search, cited answers and a tool-using agent over earnings call transcripts. " + NOT_ADVICE)
    ts = get_transcripts()
    if ts:
        st.dataframe(pd.DataFrame(ts)[["company", "ticker", "quarter", "year", "date", "source"]],
                     width="stretch", hide_index=True)
    else:
        st.info("No transcripts yet. See README, Quickstart.")

elif page == "Sentiment":
    t = transcript_picker()
    st.title(f"Sentiment · {t['ticker']} {t['quarter']} {t['year']}")
    sec = st.radio("Section", ["all", "prepared", "qa"], horizontal=True,
                   format_func=lambda x: {"all": "Whole call", "prepared": "Prepared remarks", "qa": "Q&A"}[x])
    s, err = api("GET", f"/v1/transcripts/{t['id']}/sentiment", params={"section": None if sec == "all" else sec})
    if err:
        st.warning(err)
        st.stop()
    c = st.columns(4)
    for col, (lab, val) in zip(c, [("Positive", f"{s['positive_pct']}%"), ("Neutral", f"{s['neutral_pct']}%"),
                                   ("Negative", f"{s['negative_pct']}%"), ("Net score", f"{s['net_score']:+.3f}")]):
        col.metric(lab, val)
    scores = s.get("sentence_scores") or []
    if scores:
        net = pd.Series([x["positive"] - x["negative"] for x in scores])
        roll = net.rolling(max(5, len(net) // 30), center=True, min_periods=1).mean()
        fig = go.Figure(go.Scatter(y=roll, mode="lines", line=dict(color="#388bfd", width=2)))
        fig.add_hline(y=0, line_dash="dot", line_color="#30363d")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Sentiment arc (rolling mean)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"))
        st.plotly_chart(fig, width="stretch")
    l, r = st.columns(2)
    with l:
        st.subheader("Most positive")
        for x in s["top_positive_sentences"]:
            st.markdown(f'<div class="card">{safe(x["text"])}<div class="meta">confidence {x["score"]:.2f}</div></div>', unsafe_allow_html=True)
    with r:
        st.subheader("Most negative")
        for x in s["top_negative_sentences"]:
            st.markdown(f'<div class="card">{safe(x["text"])}<div class="meta">confidence {x["score"]:.2f}</div></div>', unsafe_allow_html=True)

elif page == "Summary":
    t = transcript_picker()
    st.title(f"Summary · {t['ticker']} {t['quarter']} {t['year']}")
    r, err = api("GET", f"/v1/transcripts/{t['id']}/summary")
    if r["status"] == "ready":
        sm = r["summary"]
        tone = sm["analyst_signal"]
        cls = {"Bullish": "pos", "Bearish": "neg"}.get(tone, "")
        st.markdown(f'<span class="badge {cls}">TONE: {tone.upper()}</span> &nbsp; {safe(sm["signal_justification"])}', unsafe_allow_html=True)
        st.caption(r["note"])
        st.markdown(f'<div class="card">{safe(sm["executive_summary"])}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.subheader("Key takeaways")
            for x in sm["key_takeaways"]:
                st.markdown(f"- {md(x)}")
            st.subheader("Opportunities")
            for x in sm["opportunities"]:
                st.markdown(f"- {md(x)}")
        with b:
            st.subheader("Risks")
            for x in sm["risks"]:
                st.markdown(f"- {md(x)}")
            st.subheader("Verified quotes")
            for q in sm["notable_quotes"] or ["(none verified verbatim)"]:
                st.markdown(f'<div class="cite">“{safe(q)}”</div>', unsafe_allow_html=True)
        st.caption(f"Generated by {sm['model_used']} (map-reduce over the whole call; quotes are checked against the transcript)")
    elif r["status"] == "running":
        st.info("Generating (map-reduce over the full call, can take several minutes on CPU). Refresh in a moment.")
        if st.button("Refresh"):
            st.rerun()
    else:
        if r["status"] == "failed":
            st.error(r["detail"])
        if st.button("Generate summary", type="primary"):
            _, err = api("POST", f"/v1/transcripts/{t['id']}/summary", headers={"X-API-Key": ADMIN_KEY} if ADMIN_KEY else {})
            if err:
                st.error(err)
                if err.startswith("403"):
                    st.info("Summaries are a write action, so the API keeps them off until you set ADMIN_API_KEY. "
                            "Add `ADMIN_API_KEY=any-long-string` to .env, then restart BOTH the API and this UI.")
            else:
                st.rerun()

elif page == "Search":
    st.title("Search")
    q = st.text_input("Query", placeholder='e.g. "supply constraints" or "gross margin guidance"')
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Mode", ["hybrid_rerank", "hybrid", "dense", "bm25"])
    k = c2.selectbox("Results", [3, 5, 8, 10], index=1)
    tk = c3.selectbox("Ticker", ["any"] + sorted({t["ticker"] for t in get_transcripts()}))
    sec = c4.selectbox("Section", ["any", "prepared", "qa"])
    compare = st.checkbox("Compare dense vs hybrid + rerank side by side")
    flt = {k_: v for k_, v in {"ticker": None if tk == "any" else tk, "section": None if sec == "any" else sec}.items() if v}
    if q:
        modes = ["dense", "hybrid_rerank"] if compare else [mode]
        cols = st.columns(len(modes))
        for col, m in zip(cols, modes):
            r, err = api("POST", "/v1/search", json={"query": q, "k": k, "mode": m, "filters": flt or None})
            with col:
                if err:
                    st.warning(err)
                    continue
                st.caption(f"**{m}** · {r['latency_ms']:.0f} ms")
                for h in r["hits"]:
                    hit_card(h)

else:  # Ask the analyst
    st.title("Ask the analyst")
    st.caption("A LangGraph agent chooses between transcript search, sentiment and price tools. "
               "Answers cite their sources and are checked against tool output. " + NOT_ADVICE)
    if "thread" not in st.session_state:
        st.session_state.thread, st.session_state.chat = None, []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(md(m["content"]))
    prompt = st.chat_input("e.g. How did the tone on supply change across quarters, and how did the stock react?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(md(prompt))
        with st.chat_message("assistant"):
            status = st.status("Working...", expanded=True)
            final, cites = None, []
            try:
                with requests.post(f"{API_URL}/v1/agent/chat", json={"message": prompt, "thread_id": st.session_state.thread},
                                   stream=True, timeout=600) as resp:
                    if resp.status_code != 200:
                        status.update(label="Error", state="error")
                        st.error(resp.text)
                    ev_type = None
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("event:"):
                            ev_type = line[6:].strip()
                        elif line and line.startswith("data:"):
                            e = json.loads(line[5:])
                            if e["type"] == "route":
                                status.write(f"🧭 route: **{e['route']}**")
                            elif e["type"] == "tool_call":
                                status.markdown(f'<div class="step">🔧 <b>{e["tool"]}</b> {json.dumps(e["args"])}</div>', unsafe_allow_html=True)
                            elif e["type"] == "tool_result":
                                cls = "err" if e.get("error") else "ok"
                                status.markdown(f'<div class="step {cls}">↳ {e["result"][:300]}</div>', unsafe_allow_html=True)
                            elif e["type"] == "verify":
                                status.write("✅ verified against evidence" if e["ok"] else f"⚠️ verification issue: {e.get('problems')}")
                            elif e["type"] == "limit":
                                status.write(f"ℹ️ {e['message']}")
                            elif e["type"] == "answer":
                                final, cites = e["answer"], e["citations"]
                            elif e["type"] == "done":
                                st.session_state.thread = e["thread_id"]
                            elif e["type"] == "error":
                                status.write(f"❌ {e['message']}")
            except requests.ConnectionError:
                st.error(f"Cannot reach the API at {API_URL}")
            status.update(label="Done", state="complete", expanded=False)
            if final:
                text = ("I don't have enough evidence in the loaded transcripts to answer that." if final == "INSUFFICIENT_EVIDENCE" else final)
                st.markdown(md(text))
                for c in cites:
                    st.markdown(f'<div class="cite">[{c["label"]}] {c["ticker"]} {c["quarter"]} {c["year"]} · {c["speaker"] or "n/a"}<br>{safe(c["snippet"])}…</div>', unsafe_allow_html=True)
                st.session_state.chat.append({"role": "assistant", "content": text})
    if st.button("New conversation"):
        st.session_state.thread, st.session_state.chat = None, []
        st.rerun()

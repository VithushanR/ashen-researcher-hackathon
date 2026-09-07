"""Streamlit front end for Ashen Researcher.

Four things on screen, in the order a judge needs them:

1. A question box.
2. A live reasoning trace -- each research round appearing as the agent
   finishes it, showing the query it wrote, what it found, and whether it
   decided that was enough.
3. The final answer, every claim clickable through to the source that supports it.
4. Conflict callouts, when sources disagreed and the agent had to choose.

The trace panel is the reason this interface exists. Any question-answering
system can print an answer; this one shows the judge the moment it decided it
did not have enough evidence and searched again. That is the whole claim of
sub-track 1C, made visible.

Run it (with the API already running on port 8000):

    streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# Allows `streamlit run src/ui/app.py` from the repo root without installing the
# package -- one less step in the README's clean-machine path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ui.render import (  # noqa: E402
    citation_location,
    confidence_style,
    conflict_claims,
    is_winning_claim,
    linkify_answer,
    status_badge,
    summarise_trace,
    tier_badge,
    verdict_badge,
)

API_BASE = os.getenv("ASHEN_API_BASE", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = int(os.getenv("ASHEN_UI_TIMEOUT", "300"))

SAMPLE_QUESTIONS = [
    ("Which war was won by the organization that included Isolde Mournvale "
     "as one of its members?", "Multi-hop: no single document holds the answer"),
    ("In which year was the Gauntlet of Sorrowfell actually forged?",
     "Disputed: two sources give different years"),
    ("Whose dominion encompasses the lair of the Gravemaw Wyrm?",
     "Unanswerable: the agent admits the gap"),
]

st.set_page_config(
    page_title="Ashen Researcher",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colours use alpha-blended backgrounds against currentColor text so the page
# stays legible in both Streamlit themes without a second palette.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1180px; }
      .ashen-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
                     margin-bottom: 0.1rem; }
      .ashen-sub { opacity: 0.65; font-size: 0.95rem; margin-bottom: 1.4rem; font-style: italic; }
      .ashen-card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 10px;
        padding: 0.85rem 1rem; margin-bottom: 0.6rem;
        background: rgba(128,128,128,0.06);
      }
      .ashen-step-head { display: flex; align-items: center; gap: 0.6rem;
                         margin-bottom: 0.45rem; flex-wrap: wrap; }
      .ashen-round { font-size: 0.7rem; font-weight: 700; opacity: 0.55;
                     letter-spacing: 0.09em; }
      .ashen-query { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: 0.9rem; }
      .ashen-found { opacity: 0.7; font-size: 0.85rem; margin-top: 0.35rem; }
      .ashen-missing { color: #b8791f; font-size: 0.86rem; margin-top: 0.4rem;
                       font-weight: 500; }
      .ashen-answer { font-size: 1.05rem; line-height: 1.8; }
      .ashen-conflict {
        border-left: 4px solid #9a6a1f; background: rgba(154,106,31,0.10);
        border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
      }
      .ashen-gap {
        border-left: 4px solid #a33a3a; background: rgba(163,58,58,0.10);
        border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
      }
      .ashen-meta { opacity: 0.6; font-size: 0.82rem; }
      .ashen-pill { border-radius: 999px; padding: 2px 10px; font-size: 0.72rem;
                    font-weight: 600; white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------

def fetch_health() -> dict[str, Any] | None:
    """Ask the backend what is actually wired. None means it is not running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def stream_answer(question: str, baseline: bool):
    """Yield SSE events from the backend as the research happens."""
    with requests.post(
        f"{API_BASE}/ask/stream",
        json={"question": question, "baseline": baseline},
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


@st.cache_data(show_spinner=False)
def fetch_source(filename: str) -> dict[str, Any] | None:
    """Fetch the archive file behind a citation. Cached -- files do not change."""
    try:
        response = requests.get(f"{API_BASE}/source", params={"filename": filename}, timeout=15)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        return None


def pill(label: str, colour: str) -> str:
    return (f'<span class="ashen-pill" style="background:{colour}1f;color:{colour};'
            f'border:1px solid {colour}55">{label}</span>')


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_step(step: dict[str, Any]) -> None:
    """One research round, as a card in the trace panel.

    Deliberately shows the *gap* as prominently as the find. "Still missing: which
    war the Ashen Order won" is what makes the next round make sense, and it is
    the sentence that distinguishes this from a one-shot retrieval system.
    """
    label, colour = verdict_badge(step.get("verdict"))
    body = (
        f'<div class="ashen-card">'
        f'<div class="ashen-step-head">'
        f'<span class="ashen-round">ROUND {step.get("step", "?")}</span>'
        f"{pill(label, colour)}"
        f"</div>"
        f'<div class="ashen-query">🔍 {step.get("query") or "—"}</div>'
    )
    if step.get("found"):
        body += f'<div class="ashen-found">{step["found"]}</div>'
    if step.get("missing"):
        body += f'<div class="ashen-missing">↳ Still missing: {step["missing"]}</div>'
    st.markdown(body + "</div>", unsafe_allow_html=True)


def render_conflicts(conflicts: list[dict[str, Any]]) -> None:
    """Conflict callouts: what disagreed, what won, and why.

    The flagship feature. The losing claim stays on screen next to the winner --
    never collapsed into a single value, because preserving and explaining the
    disagreement is the differentiator, and hiding it is what a system that
    retrieved one source at random would do.
    """
    if not conflicts:
        return

    st.markdown("#### ⚖️ Sources disagreed")
    for conflict in conflicts:
        resolved = conflict.get("resolved_value")
        rows = ""
        for claim in conflict_claims(conflict):
            won = is_winning_claim(claim, resolved)
            marker, weight = ("✅", "600") if won else ("❌", "400")
            rows += (
                f'<div style="margin:0.4rem 0;font-weight:{weight}">{marker} '
                f'{claim.get("claim", "")}'
                f'<div class="ashen-meta" style="margin-left:1.6rem">'
                f'{claim.get("source", "unknown source")}</div></div>'
            )
        resolution = conflict.get("resolution") or (
            "Not resolved — both claims are reported as they stand."
        )
        st.markdown(
            f'<div class="ashen-conflict">'
            f'<div style="font-weight:700;margin-bottom:0.5rem">'
            f'{conflict.get("attribute") or "Conflicting claim"}</div>'
            f"{rows}"
            f'<div style="margin-top:0.7rem;font-size:0.9rem">'
            f"<b>Resolution:</b> {resolution}</div></div>",
            unsafe_allow_html=True,
        )


def render_gaps(unresolved: list[str]) -> None:
    """What the agent could not establish. Shown, never quietly dropped."""
    if not unresolved:
        return
    st.markdown("#### 🕳️ Not established by the archive")
    items = "".join(f"<div style='margin:0.3rem 0'>• {claim}</div>" for claim in unresolved)
    st.markdown(f'<div class="ashen-gap">{items}</div>', unsafe_allow_html=True)


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Numbered, expandable evidence -- the click target of the answer's markers."""
    if not citations:
        st.info("No citations were attached to this answer.")
        return

    st.markdown("#### 📎 Evidence")
    for index, citation in enumerate(citations, start=1):
        location = citation_location(citation)
        # Anchor target for the superscript links in the answer text.
        st.markdown(f'<div id="cite-{index}"></div>', unsafe_allow_html=True)
        with st.expander(f"[{index}]  {citation.get('claim', '')}  —  {location}"):
            badge = tier_badge(citation)
            if badge:
                st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**Source:** `{location}`")
            if citation.get("chunk_id"):
                st.markdown(
                    f"<span class='ashen-meta'>chunk <code>{citation['chunk_id']}</code></span>",
                    unsafe_allow_html=True,
                )
            if citation.get("text"):
                st.markdown("**Supporting passage**")
                st.info(citation["text"])

            key = f"src-{index}"
            if st.button("Open source file", key=f"btn-{key}"):
                st.session_state[key] = True
            if st.session_state.get(key):
                source = fetch_source(str(citation.get("filename", "")))
                if source is None:
                    st.warning("That file is not in the local archive folder. "
                               "Set ASHEN_ARCHIVE_ROOT if the archive lives elsewhere.")
                elif source.get("content"):
                    st.text_area("File contents", source["content"], height=280,
                                 key=f"text-{key}")
                else:
                    st.info(source.get("note", "Binary source — open it from the archive."))


def render_answer(payload: dict[str, Any]) -> None:
    """The final answer block: status, confidence, prose, conflicts, gaps, evidence."""
    status_label, status_colour = status_badge(payload.get("status", ""))
    confidence = int(payload.get("confidence", 0))
    conf_label, conf_colour = confidence_style(confidence)

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f'{pill(status_label, status_colour)}'
            f'<span class="ashen-meta" style="margin-left:0.8rem">'
            f'{payload.get("iterations_used", 0)} research rounds · '
            f'route: {payload.get("route") or "n/a"}</span>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div style="text-align:right">'
            f'<span style="color:{conf_colour};font-weight:700;font-size:1.4rem">{confidence}</span>'
            f'<span class="ashen-meta">/100 · {conf_label}</span></div>',
            unsafe_allow_html=True,
        )
    st.progress(min(max(confidence, 0), 100) / 100)

    st.markdown(
        f'<div class="ashen-answer">'
        f'{linkify_answer(payload.get("answer", ""), payload.get("citations", []))}</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    render_conflicts(payload.get("conflicts", []))
    render_gaps(payload.get("unresolved_claims", []))
    render_citations(payload.get("citations", []))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown('<div class="ashen-title">🕯️ Ashen Researcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ashen-sub">It does not stop when it finds something relevant. '
    "It stops when it has enough evidence.</div>",
    unsafe_allow_html=True,
)

health = fetch_health()

with st.sidebar:
    st.markdown("### System status")
    if health is None:
        st.error("Backend unreachable")
        st.caption(f"Expected at `{API_BASE}`. Start it with:")
        st.code("uvicorn src.api.main:app --port 8000", language="bash")
    elif health["pipeline"] == "real":
        st.success("Real pipeline wired")
    else:
        st.warning("Running on fixtures")
        st.caption(
            "Answers below are canned, not researched. Person B's `research()` "
            "and Person C's `compose_answer()` are not both importable yet."
        )

    if health:
        st.markdown(
            f"- Agent loop (B): {'✅' if health['research_available'] else '⏳ pending'}\n"
            f"- Composer (C): {'✅' if health['compose_available'] else '⏳ pending'}\n"
            f"- Trace: `{health['streaming']}`"
        )
        if health["streaming"] == "replayed":
            st.caption("`replayed` = the loop has no per-round callback yet, so the "
                       "trace is shown after the run rather than during it.")

    st.divider()
    st.markdown("### Compare")
    baseline = st.toggle(
        "Baseline RAG",
        value=False,
        help="One retrieval, no loop, no sufficiency check. The contrast that "
             "shows what the agent loop actually buys.",
    )

    st.divider()
    st.markdown("### Sample questions")
    for index, (sample, why) in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample-{index}", use_container_width=True):
            st.session_state["question"] = sample
        st.caption(why)

question = st.text_area(
    "Ask the archive",
    key="question",
    height=90,
    placeholder="Which war was won by the organization that included Isolde Mournvale?",
)
ask_clicked = st.button("Research", type="primary", disabled=health is None)

# Results live in session state so a rerun -- clicking "Open source file",
# expanding a citation -- re-renders the answer instead of losing it. Without
# this, any interaction after the answer arrives would blank the page.
st.session_state.setdefault("steps", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)

trace_area = st.container()

if ask_clicked and question.strip():
    st.session_state.update(steps=[], result=None, error=None)
    for key in [k for k in st.session_state if k.startswith("src-")]:
        st.session_state[key] = False

    if health and health["pipeline"] == "stub":
        st.warning("Fixture mode — this answer is canned, not researched.", icon="⚠️")

    with trace_area:
        st.markdown("### 🧠 Reasoning trace")

    with st.status("Researching…", expanded=True) as status_box:
        try:
            for event in stream_answer(question.strip(), baseline):
                kind = event.get("event")
                if kind == "step":
                    st.session_state["steps"].append(event["step"])
                    with trace_area:
                        render_step(event["step"])
                    status_box.update(
                        label=f"Round {event['step'].get('step', '?')} — "
                              f"{verdict_badge(event['step'].get('verdict'))[0].lower()}"
                    )
                elif kind == "answer":
                    st.session_state["result"] = event["payload"]
                elif kind == "error":
                    st.session_state["error"] = event.get("message", "Unknown error")
        except requests.RequestException as exc:
            st.session_state["error"] = f"Could not reach the backend: {exc}"

        if st.session_state["error"]:
            status_box.update(label="Research failed", state="error")
        else:
            status_box.update(label=summarise_trace(st.session_state["steps"]), state="complete")

elif st.session_state["steps"]:
    # A rerun with no new question: redraw the previous trace from session state.
    with trace_area:
        st.markdown("### 🧠 Reasoning trace")
        for step in st.session_state["steps"]:
            render_step(step)
        st.caption(summarise_trace(st.session_state["steps"]))

if st.session_state["error"]:
    st.error(st.session_state["error"])
elif st.session_state["result"]:
    st.divider()
    render_answer(st.session_state["result"])
    with st.expander("Raw response (JSON)"):
        st.json(st.session_state["result"])

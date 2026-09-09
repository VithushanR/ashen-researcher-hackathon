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

Layout is a single centered conversation column (question in, everything else
below it) rather than a wide dashboard -- the trace is a quiet, secondary
"process" detail; the answer is the main event.

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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Results live in session state so a rerun -- clicking "Open source file",
# expanding a citation -- re-renders the conversation instead of losing it.
st.session_state.setdefault("steps", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("error", None)
st.session_state.setdefault("asked", None)

# Palette: strict 60/30/10. 60% a warm cream/ivory canvas, 30% a warm taupe
# for elevated surfaces (cards/panels/inputs), 10% a single burnished-copper
# accent reserved for the primary CTA, active states, and the one number in
# the whole page that matters most (confidence). Every other colour on screen
# is a tint of the warm-brown text or taupe surface, never a fourth hue --
# that restraint is what reads as premium rather than decorated. Defined once
# as CSS custom properties; render.py's badge palettes are hand-kept in the
# same warm family since that file intentionally has no Streamlit/CSS access
# of its own.
#
# Hex values here are mirrored in .streamlit/config.toml for Streamlit's native
# widgets (buttons, inputs, slider, sidebar chrome) -- keep the two in sync.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

      :root {
        --ashen-bg: #f7f0e3;            /* 60% dominant -- warm cream/ivory canvas */
        --ashen-surface: #ece0cb;       /* 30% secondary -- warm taupe panel */
        --ashen-surface-2: #e3d5ba;     /* nested surface, one step deeper */
        --ashen-accent: #c1652e;        /* 10% accent -- burnished copper/terracotta */
        --ashen-accent-soft: rgba(193, 101, 46, 0.12);
        --ashen-accent-border: rgba(193, 101, 46, 0.45);
        --ashen-text: #3a2e22;          /* dark warm brown, not pure black */
        --ashen-text-muted: #7a6a56;    /* muted warm brown-grey */
        --ashen-text-faint: #9c8b74;
        --ashen-border: rgba(58, 46, 34, 0.12);
        --ashen-serif: 'Fraunces', Georgia, serif;
        --ashen-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }

      html, body, [class*="css"] { font-family: var(--ashen-sans); }
      .stApp { background: var(--ashen-bg); color: var(--ashen-text); }

      /* A centered conversation column, ChatGPT-width, not a wide dashboard. */
      .block-container { padding-bottom: 4rem; max-width: 720px; }

      section[data-testid="stSidebar"] {
        background: var(--ashen-surface); border-right: 1px solid var(--ashen-border);
      }
      section[data-testid="stSidebar"] .block-container { padding-top: 2rem; max-width: none; }

      /* ---- Type scale ---------------------------------------------------- */
      h1, h2, h3, h4, .ashen-title { font-family: var(--ashen-serif); font-weight: 600;
                                      color: var(--ashen-text); letter-spacing: -0.01em; }
      .stMarkdown h3, .stMarkdown h4 {
        font-family: var(--ashen-serif); font-weight: 600; margin-top: 0.4rem;
      }
      .ashen-title { font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.15rem; }
      .ashen-sub { color: var(--ashen-text-muted); font-family: var(--ashen-serif);
                   font-style: italic; font-size: 1.05rem; margin-bottom: 1.6rem; }

      /* Small uppercase label for secondary section headers -- Process,
         Sources disagreed, Evidence -- kept deliberately quiet next to the
         serif answer prose so nothing competes with the main event. */
      .ashen-eyebrow { font-family: var(--ashen-sans); font-size: 0.72rem; font-weight: 700;
                        letter-spacing: 0.12em; text-transform: uppercase;
                        color: var(--ashen-text-faint); margin: 1.6rem 0 0.7rem; }

      /* The user's own question, echoed above the answer like a chat turn --
         understated, never competing with the assistant's response below it. */
      .ashen-user-turn { color: var(--ashen-text-muted); font-size: 0.95rem;
                          line-height: 1.5; margin: 1.8rem 0 0.9rem;
                          padding-left: 0.8rem; border-left: 2px solid var(--ashen-border); }

      /* ---- Question input: the singular focal point ---------------------- */
      div[data-testid="stTextArea"] textarea {
        background: var(--ashen-surface); border: 1px solid var(--ashen-border);
        border-radius: 12px; color: var(--ashen-text); font-family: var(--ashen-serif);
        font-size: 1.15rem; padding: 1rem 1.1rem; line-height: 1.5;
      }
      div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--ashen-accent-border);
        box-shadow: 0 0 0 3px var(--ashen-accent-soft);
      }

      /* Primary CTA is the one place, besides confidence, the accent fills a
         surface rather than just tinting one -- its rarity is the point. */
      div[data-testid="stButton"] button[kind="primary"] {
        background: var(--ashen-accent); border: none; color: #fbf3e6;
        font-weight: 700; font-family: var(--ashen-sans); border-radius: 8px;
        padding: 0.6rem 1.8rem; letter-spacing: 0.01em;
      }
      div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #a8531f; box-shadow: 0 4px 14px rgba(193, 101, 46, 0.25);
      }
      div[data-testid="stButton"] button[kind="primary"]:disabled {
        background: var(--ashen-surface-2); color: var(--ashen-text-faint);
      }
      div[data-testid="stButton"] button:not([kind="primary"]) {
        background: var(--ashen-surface-2); border: 1px solid var(--ashen-border);
        color: var(--ashen-text); border-radius: 8px; text-align: left;
      }

      /* Sidebar sample-question buttons read as a clean list, not a button grid. */
      section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: transparent; border: none; padding: 0.25rem 0;
        color: var(--ashen-text); font-size: 0.88rem; font-weight: 500;
      }
      section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        color: var(--ashen-accent);
      }
      section[data-testid="stSidebar"] .ashen-eyebrow { margin-top: 0.2rem; }

      /* ---- Trace / process panel: compact, secondary, deliberately quiet - */
      .ashen-card {
        border: 1px solid var(--ashen-border); border-radius: 8px;
        padding: 0.6rem 0.85rem; margin-bottom: 0.4rem;
        background: var(--ashen-surface); font-size: 0.85rem;
      }
      .ashen-step-head { display: flex; align-items: center; gap: 0.55rem;
                         margin-bottom: 0.3rem; flex-wrap: wrap; }
      .ashen-round { font-size: 0.64rem; font-weight: 700; color: var(--ashen-text-faint);
                     letter-spacing: 0.1em; }
      .ashen-query { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                     font-size: 0.8rem; color: var(--ashen-text-muted); }
      .ashen-found { color: var(--ashen-text-muted); font-size: 0.78rem; margin-top: 0.3rem; }
      /* Not the accent colour on purpose -- accent is reserved for the CTA,
         active states, and confidence; this is a secondary trace detail. */
      .ashen-missing { color: #8a6a3f; font-size: 0.78rem; margin-top: 0.35rem;
                       font-weight: 500; }
      div[data-testid="stExpander"] summary { font-family: var(--ashen-sans); }

      /* ---- Answer: the main event ----------------------------------------- */
      div.st-key-answer-card {
        background: var(--ashen-surface); border: 1px solid var(--ashen-border);
        border-radius: 16px; padding: 1.7rem 2rem 1.4rem; margin: 0.3rem 0 0.4rem;
      }
      .ashen-answer-meta { display: flex; align-items: center; justify-content: space-between;
                            flex-wrap: wrap; gap: 0.6rem; }
      .ashen-answer { font-family: var(--ashen-serif); font-size: 1.25rem; line-height: 1.8;
                      color: var(--ashen-text); margin-top: 1rem; }
      .ashen-confidence-value { color: var(--ashen-accent); font-weight: 700; font-size: 1.5rem;
                                 font-family: var(--ashen-serif); }
      div[data-testid="stProgress"] div[role="progressbar"] > div { background: var(--ashen-accent); }
      div[data-testid="stProgress"] div[role="progressbar"] { background: var(--ashen-surface-2); }

      /* Conflict callout: the single most important thing to notice on screen,
         but calm -- a bordered card with a thin accent edge, not an alarm. */
      .ashen-conflict {
        border: 1px solid var(--ashen-accent-border); border-left: 3px solid var(--ashen-accent);
        background: var(--ashen-accent-soft); border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.9rem;
      }
      .ashen-gap {
        border-left: 3px solid var(--ashen-text-faint); background: var(--ashen-surface-2);
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.9rem;
      }
      .ashen-meta { color: var(--ashen-text-faint); font-size: 0.8rem; }
      .ashen-claim-won { font-weight: 600; }
      .ashen-claim-won .ashen-mark { color: #7fa06a; font-weight: 700; margin-right: 0.4rem; }
      .ashen-claim-lost { color: var(--ashen-text-muted); font-weight: 400; }
      .ashen-claim-lost .ashen-mark { color: var(--ashen-text-faint); margin-right: 0.4rem; }

      /* Restrained pill/badge -- colour comes from render.py's muted palette,
         this just enforces the small, quiet, pill shape across all of them. */
      .ashen-pill { border-radius: 999px; padding: 2px 11px; font-size: 0.7rem;
                    font-weight: 600; white-space: nowrap; font-family: var(--ashen-sans); }

      /* Citation chips: understated, not default blue hyperlinks. */
      .ashen-cite-chip {
        display: inline-block; background: var(--ashen-surface-2); color: var(--ashen-text-muted);
        border: 1px solid var(--ashen-border); border-radius: 5px; padding: 0 6px;
        font-weight: 700; font-size: 0.68rem; font-family: var(--ashen-sans);
        text-decoration: none; vertical-align: super;
      }
      .ashen-cite-chip:hover { color: var(--ashen-accent); border-color: var(--ashen-accent-border); }

      div[data-testid="stExpander"] {
        background: var(--ashen-surface); border: 1px solid var(--ashen-border); border-radius: 10px;
      }
      hr { border-color: var(--ashen-border) !important; }
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
    """One research round, as a card in the process panel.

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
        f'<div class="ashen-query">{step.get("query") or "—"}</div>'
    )
    if step.get("found"):
        body += f'<div class="ashen-found">{step["found"]}</div>'
    if step.get("missing"):
        body += f'<div class="ashen-missing">Still missing: {step["missing"]}</div>'
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

    st.markdown('<div class="ashen-eyebrow">Sources disagreed</div>', unsafe_allow_html=True)
    for conflict in conflicts:
        resolved = conflict.get("resolved_value")
        rows = ""
        for claim in conflict_claims(conflict):
            won = is_winning_claim(claim, resolved)
            css_class = "ashen-claim-won" if won else "ashen-claim-lost"
            mark = "&#10003;" if won else "&#8211;"  # a plain check mark / en dash, not emoji
            rows += (
                f'<div class="{css_class}" style="margin:0.4rem 0">'
                f'<span class="ashen-mark">{mark}</span>{claim.get("claim", "")}'
                f'<div class="ashen-meta" style="margin-left:1.4rem">'
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
    st.markdown('<div class="ashen-eyebrow">Not established by the archive</div>',
                unsafe_allow_html=True)
    items = "".join(f"<div style='margin:0.3rem 0'>{claim}</div>" for claim in unresolved)
    st.markdown(f'<div class="ashen-gap">{items}</div>', unsafe_allow_html=True)


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Numbered, expandable evidence -- the click target of the answer's markers."""
    if not citations:
        st.info("No citations were attached to this answer.")
        return

    st.markdown('<div class="ashen-eyebrow">Evidence</div>', unsafe_allow_html=True)
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


def render_composition_failed(payload: dict[str, Any]) -> None:
    """Research succeeded; composition didn't. Calm, not alarming: this is a
    known model-consistency limitation (see pipeline.py's _degraded_payload),
    not a broken app -- so it gets its own quiet card, the real evidence
    found, and no citation-linked prose there was never a validated answer for.
    """
    status_label, status_colour = status_badge(payload.get("status", ""))
    with st.container(border=False, key="answer-card"):
        st.markdown(
            f'<div class="ashen-answer-meta">'
            f'<span>{pill(status_label, status_colour)}'
            f'<span class="ashen-meta" style="margin-left:0.8rem">'
            f'{payload.get("iterations_used", 0)} research rounds · '
            f'route: {payload.get("route") or "n/a"}</span></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ashen-answer" style="font-size:1.05rem">'
            "Found relevant evidence but couldn't produce a fully verified "
            "answer this time — please try again.</div>",
            unsafe_allow_html=True,
        )

    evidence = payload.get("evidence", [])
    if evidence:
        st.markdown('<div class="ashen-eyebrow">Evidence found</div>', unsafe_allow_html=True)
        for item in evidence:
            location = citation_location(item)
            text = (item.get("text") or "")[:300]
            st.markdown(
                f'<div class="ashen-card">'
                f'<div class="ashen-meta">{location}</div>'
                f'<div style="margin-top:0.3rem">{text}</div></div>',
                unsafe_allow_html=True,
            )
    render_gaps(payload.get("unresolved_claims", []))


def render_answer(payload: dict[str, Any]) -> None:
    """The final answer block: status, confidence, prose, conflicts, gaps, evidence."""
    if payload.get("status") == "composition_failed":
        render_composition_failed(payload)
        return
    status_label, status_colour = status_badge(payload.get("status", ""))
    confidence = int(payload.get("confidence", 0))
    conf_label, _conf_colour = confidence_style(confidence)

    # A real bordered container (not a raw HTML div) so the whole answer --
    # the meta row, the native progress bar, and the prose -- renders as one
    # elevated card. The meta row is one flex line, not Streamlit columns, so
    # it reads as a single conversational block rather than a dashboard split.
    with st.container(border=False, key="answer-card"):
        st.markdown(
            f'<div class="ashen-answer-meta">'
            f'<span>{pill(status_label, status_colour)}'
            f'<span class="ashen-meta" style="margin-left:0.8rem">'
            f'{payload.get("iterations_used", 0)} research rounds · '
            f'route: {payload.get("route") or "n/a"}</span></span>'
            f'<span><span class="ashen-confidence-value">{confidence}</span>'
            f'<span class="ashen-meta">/100 · {conf_label}</span></span></div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0), 100) / 100)

        st.markdown(
            f'<div class="ashen-answer">'
            f'{linkify_answer(payload.get("answer", ""), payload.get("citations", []))}</div>',
            unsafe_allow_html=True,
        )

    render_conflicts(payload.get("conflicts", []))
    render_gaps(payload.get("unresolved_claims", []))
    render_citations(payload.get("citations", []))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.markdown('<div class="ashen-title">Ashen Researcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ashen-sub">It does not stop when it finds something relevant. '
    "It stops when it has enough evidence.</div>",
    unsafe_allow_html=True,
)

health = fetch_health()

with st.sidebar:
    st.markdown('<div class="ashen-eyebrow">Sample questions</div>', unsafe_allow_html=True)
    for index, (sample, why) in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample-{index}", use_container_width=True):
            st.session_state["question"] = sample
        st.caption(why)

    st.divider()
    baseline = st.toggle(
        "Baseline RAG",
        value=False,
        help="One retrieval, no loop, no sufficiency check. The contrast that "
             "shows what the agent loop actually buys.",
    )

    st.divider()
    with st.expander("System status", expanded=False):
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
                f"- Agent loop (B): {'Ready' if health['research_available'] else 'Pending'}\n"
                f"- Composer (C): {'Ready' if health['compose_available'] else 'Pending'}\n"
                f"- Trace: `{health['streaming']}`"
            )
            if health["streaming"] == "replayed":
                st.caption("`replayed` = the loop has no per-round callback yet, so the "
                           "trace is shown after the run rather than during it.")

question = st.text_area(
    "Ask the archive",
    key="question",
    height=90,
    placeholder="Which war was won by the organization that included Isolde Mournvale?",
    label_visibility="collapsed",
)
ask_clicked = st.button("Research", type="primary", disabled=health is None)
if health is None:
    # A disabled button gives zero visual feedback on its own -- Streamlit
    # renders it identically to an enabled one apart from the dimmed style,
    # so without this a dead backend looks exactly like "the app is broken"
    # rather than "the button is correctly refusing to fire." This used to
    # be self-evident when the health check sat at the top of the sidebar;
    # moving it into a collapsed "System status" expander for a cleaner look
    # silently took this feedback away, so it's restored here, inline, where
    # it's actually visible at the moment it matters.
    st.caption(f"Research is disabled: no backend reachable at `{API_BASE}`. "
               "See System status in the sidebar.")

trace_area = st.container()

if ask_clicked and question.strip():
    st.session_state.update(steps=[], result=None, error=None, asked=question.strip())
    for key in [k for k in st.session_state if k.startswith("src-")]:
        st.session_state[key] = False

    st.markdown(f'<div class="ashen-user-turn">{question.strip()}</div>', unsafe_allow_html=True)

    if health and health["pipeline"] == "stub":
        st.warning("Fixture mode — this answer is canned, not researched.")

    with trace_area:
        st.markdown('<div class="ashen-eyebrow">Process</div>', unsafe_allow_html=True)

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

elif st.session_state["steps"] or st.session_state["result"] or st.session_state["error"]:
    # A rerun with no new question: redraw the previous turn from session
    # state. The trace is now a settled, secondary detail, so it collapses
    # into a quiet expander instead of standing open the way it does live.
    if st.session_state["asked"]:
        st.markdown(f'<div class="ashen-user-turn">{st.session_state["asked"]}</div>',
                    unsafe_allow_html=True)
    if st.session_state["steps"]:
        with trace_area:
            with st.expander(f"Process — {summarise_trace(st.session_state['steps'])}",
                              expanded=False):
                for step in st.session_state["steps"]:
                    render_step(step)

if st.session_state["error"]:
    st.error(st.session_state["error"])
elif st.session_state["result"]:
    render_answer(st.session_state["result"])
    with st.expander("Raw response (JSON)"):
        st.json(st.session_state["result"])

# The hero (empty, nothing asked yet) state sits lower on the page, like an
# empty ChatGPT conversation; once there's any activity the header compacts
# so the conversation feed gets the room instead. Computed here, at the very
# end of the script, rather than up front: a CSS <style> tag applies to the
# whole page regardless of where in the markup it appears, but session_state
# only reflects "did this run produce a result/error/step" once the
# ask_clicked handling above has actually run -- checking it before that
# point would still see last run's (pre-submit) state and stay in hero mode
# for one extra run every time a question is freshly submitted.
has_activity = bool(
    st.session_state["result"] or st.session_state["error"] or st.session_state["steps"]
)
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: {"3rem" if has_activity else "14vh"}; }}
      .ashen-title {{ font-size: {"1.7rem" if has_activity else "2.7rem"}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

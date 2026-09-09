"""Pure rendering helpers for the Streamlit UI.

No Streamlit import anywhere in this file. Everything here is a plain function
from data to a string, so the interesting logic -- turning inline
``[filename, p.47]`` markers into clickable citations, mapping source folders to
reliability tiers -- can be unit tested in milliseconds without booting a web
server. ``app.py`` stays a thin layout file that calls into here.

Anything returning HTML escapes its inputs. The answer text is model-generated
and the citation metadata comes from documents we did not write, so both are
untrusted: ``app.py`` renders these strings with ``unsafe_allow_html=True``, and
an unescaped ``<script>`` in an answer would execute.
"""

from __future__ import annotations

import html
import re
from typing import Any

# The reliability ladder from spec section 2.7. Colour is ordered by authority
# -- muted sage for the official record down to muted terracotta for an
# unverified letter -- so a judge reads trustworthiness at a glance without
# knowing what "T4" means. Desaturated on purpose: these sit on a warm
# charcoal/gold page (src/ui/app.py), and a bright rainbow badge would break
# the 60/30/10 discipline the rest of the UI holds to.
TIER_STYLES: dict[str, tuple[str, str]] = {
    "T1_authoritative": ("T1 · Authoritative", "#7fa06a"),
    "T2_curated": ("T2 · Curated", "#7b93ab"),
    "T3_narrative": ("T3 · Narrative", "#c9a15a"),
    "T4_unverified": ("T4 · Unverified", "#b06b5c"),
}

# Folder-to-tier mapping, used when a citation carries only source_type.
# Person A tags every chunk with a tier at ingestion, but a citation that has
# travelled through answer synthesis may have lost it.
SOURCE_TYPE_TIERS: dict[str, str] = {
    "codex": "T1_authoritative",
    "annals": "T1_authoritative",
    "wiki": "T2_curated",
    "novel": "T3_narrative",
    "chronicle": "T3_narrative",
    "chronicles": "T3_narrative",
    "ephemera": "T4_unverified",
}

STATUS_STYLES: dict[str, tuple[str, str]] = {
    "partial_validation_limited": ("Partial — validation limited", "#b06b5c"),
    "complete": ("Complete", "#7fa06a"),
    "complete_with_conflict": ("Conflict resolved", "#c9a15a"),
    "partial_gap_stated": ("Partial — gap stated", "#b06b5c"),
    # Research succeeded; composition didn't. Neutral, not alarmed -- this is
    # a known model-consistency limitation, not a red "error" state.
    "composition_failed": ("Answer unavailable", "#8a8377"),
}

VERDICT_STYLES: dict[str, tuple[str, str]] = {
    "sufficient": ("Enough evidence", "#7fa06a"),
    "insufficient": ("Not enough yet", "#c9a15a"),
    "conflict_detected": ("Conflict detected", "#9b83a8"),
    "capped_unresolved": ("Iteration cap reached", "#b06b5c"),
    "baseline_no_check": ("No sufficiency check", "#8a8377"),
}

NEUTRAL = "#8a8377"

# An inline citation marker: [filename.pdf, p.47] or [isolde_mournvale.md].
_MARKER = re.compile(r"\[([^\[\]]+?)\]")
# The page number inside one, in any of the forms a composer might write.
_PAGE = re.compile(r"\b(?:p\.?|pg\.?|page)\s*(\d+)", re.IGNORECASE)


def tier_for(citation: dict[str, Any]) -> str:
    """Best available reliability tier for a citation, or "" if unknown.

    Prefers the explicit ``reliability`` field Person A attaches at ingestion.
    Falls back to mapping ``source_type`` through the folder ladder, so the
    badge still renders when only the folder survived synthesis.
    """
    reliability = citation.get("reliability")
    if reliability in TIER_STYLES:
        return reliability
    return SOURCE_TYPE_TIERS.get(str(citation.get("source_type", "")).lower().strip(), "")


def tier_badge(citation: dict[str, Any]) -> str:
    """HTML pill showing the source tier. Empty string when the tier is unknown.

    Returning "" rather than an "Unknown" badge is deliberate: a citation whose
    provenance we cannot establish should look plain, not carry a label that
    implies we assessed it.
    """
    tier = tier_for(citation)
    if not tier:
        return ""
    label, colour = TIER_STYLES[tier]
    return (
        f'<span style="background:{colour}1a;color:{colour};border:1px solid {colour}55;'
        f"border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:600;"
        f'white-space:nowrap">{html.escape(label)}</span>'
    )


def citation_location(citation: dict[str, Any]) -> str:
    """Human-readable source location, following the docs/contracts.md priority.

    1. page, when it is not None
    2. section, when there is no page
    3. filename alone
    """
    filename = str(citation.get("filename") or "unknown source")
    page = citation.get("page")
    section = citation.get("section")
    if page is not None:
        return f"{filename}, p.{page}"
    if section:
        return f"{filename} — {section}"
    return filename


def _marker_page(marker_text: str) -> int | None:
    """Extract a page number from inside a citation marker, if it has one."""
    match = _PAGE.search(marker_text)
    return int(match.group(1)) if match else None


def _citation_index(marker_text: str, citations: list[dict[str, Any]]) -> int | None:
    """Find which citation an inline marker refers to.

    Matches on filename *and* page where both are available, because a codex
    volume is routinely cited at several different pages in one answer. Falls
    back to filename alone. Returns None when nothing matches, which is what
    keeps an unrecognised marker visible instead of being swallowed.
    """
    filename = marker_text.split(",")[0].strip().lower()
    if not filename:
        return None
    page = _marker_page(marker_text)

    fallback: int | None = None
    for index, citation in enumerate(citations, start=1):
        if str(citation.get("filename", "")).lower() != filename:
            continue
        if page is not None and citation.get("page") == page:
            return index  # exact filename + page
        if fallback is None:
            fallback = index
    return fallback


def linkify_answer(answer: str, citations: list[dict[str, Any]]) -> str:
    """Turn inline ``[filename, p.47]`` markers into numbered superscript links.

    The composer writes citations as bracketed filenames inside the prose. Left
    as-is they are noise in the middle of a sentence; renumbered as superscript
    links they become an index the reader can click straight through to the
    evidence, which is what makes a judge able to verify a claim live.

    A marker whose filename we cannot match is left **exactly as written**. It is
    never deleted: a citation we failed to parse is still information the reader
    should see, and silently dropping it would make an unsupported sentence look
    supported.

    The answer is HTML-escaped, because it is model-generated text about
    documents we did not write and ``app.py`` renders it as raw HTML.
    """
    if not answer:
        return ""
    if not citations:
        return html.escape(answer).replace("\n", "<br>")

    def link(number: int, title: str) -> str:
        # An understated chip, not a default blue hyperlink -- consistent with
        # the rest of the palette (src/ui/app.py's .ashen-cite-chip class).
        return (
            f'<a href="#cite-{number}" title="{html.escape(title)}" '
            f'class="ashen-cite-chip">{number}</a>'
        )

    parts: list[str] = []
    cursor = 0
    for match in _MARKER.finditer(answer):
        parts.append(html.escape(answer[cursor : match.start()]))
        number = _citation_index(match.group(1), citations)
        parts.append(html.escape(match.group(0)) if number is None else link(number, match.group(1)))
        cursor = match.end()
    parts.append(html.escape(answer[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def confidence_style(confidence: int) -> tuple[str, str]:
    """Label and colour for the grounded-rubric score.

    Thresholds mirror how the three answer statuses actually behave: a run that
    cleared sufficiency lands high, a resolved conflict lands mid, a capped
    partial lands low. This is a display band, not a judgement -- the score
    itself comes from Person B's rubric, not from here.
    """
    if confidence >= 80:
        return "Well supported", "#7fa06a"
    if confidence >= 55:
        return "Supported with caveats", "#c9a15a"
    return "Weakly supported", "#b06b5c"


def status_badge(status: str) -> tuple[str, str]:
    """Label and colour for an answer status. Unknown statuses render neutral."""
    return STATUS_STYLES.get(status, (str(status).replace("_", " ").title() or "Unknown", NEUTRAL))


def verdict_badge(verdict: str | None) -> tuple[str, str]:
    """Label and colour for one round's sufficiency verdict.

    VERDICT_STYLES covers every verdict Person B's merged loop emits. The
    fallback stays anyway: an unrecognised value is title-cased and rendered
    grey rather than dropped, so the trace panel keeps working if a new verdict
    is added upstream before it is styled here.
    """
    if not verdict:
        return ("Searching", NEUTRAL)
    return VERDICT_STYLES.get(verdict, (str(verdict).replace("_", " ").title(), NEUTRAL))


def conflict_claims(conflict: dict[str, Any]) -> list[dict[str, Any]]:
    """Competing claims, read from either conflict shape.

    ``docs/contracts.md`` uses a ``claims`` list; the fixtures PDF used pairwise
    ``claim_a`` / ``claim_b``. The API normalises on the way in, but the UI
    handles both so it still renders a hand-written fixture pasted in during a
    rehearsal, or a teammate's output that has not migrated yet.
    """
    if conflict.get("claims"):
        return list(conflict["claims"])
    claims: list[dict[str, Any]] = []
    for suffix in ("a", "b", "c"):
        claim = conflict.get(f"claim_{suffix}")
        if claim:
            claims.append({"claim": claim, "source": conflict.get(f"source_{suffix}", "unknown")})
    return claims


def is_winning_claim(claim: dict[str, Any], resolved_value: Any) -> bool:
    """Whether this claim is the one the resolution policy chose.

    Substring match on the resolved value, because Person B records the value
    ("341 AS") while the claim is prose around it ("Forged in 341 AS"). An
    unresolved conflict has no resolved value, and then no claim is marked --
    which is correct: the UI must not imply a winner where there is none.
    """
    if not resolved_value:
        return False
    return str(resolved_value).strip().lower() in str(claim.get("claim", "")).lower()


def summarise_trace(trace: list[dict[str, Any]]) -> str:
    """One line describing how the run went. Sits above the trace panel.

    Phrased around *why it stopped*, not just how many rounds it took -- the
    whole claim of this system is that it stops on sufficient evidence rather
    than on the first relevant hit.
    """
    if not trace:
        return "No research rounds recorded."
    rounds = len(trace)
    plural = "round" if rounds == 1 else "rounds"
    final = str(trace[-1].get("verdict") or "").lower()
    if final == "sufficient":
        return f"Stopped after {rounds} {plural} — enough evidence."
    if final == "capped_unresolved":
        return f"Stopped after {rounds} {plural} — iteration cap reached with a gap remaining."
    if final == "insufficient":
        # A finished trace whose last round is still "insufficient" is a run
        # that ran out of road: the loop only exits without a "sufficient"
        # verdict when it hit the iteration cap, went two rounds without new
        # evidence, or would have repeated a query.
        #
        # This branch exists because of what Person B's loop actually emits.
        # TraceStep documents "capped_unresolved" as a verdict, but the verdict
        # is copied straight from check_sufficiency(), which only ever returns
        # sufficient / insufficient / conflict_detected -- so the terminal step
        # of a capped run says "insufficient" and nothing ever says otherwise.
        # Without this, the honest-partial run -- the case sub-track 1C is
        # actually about -- fell through to the generic "N rounds recorded."
        #
        # Deriving it from the trace rather than waiting on a verdict rename
        # keeps the summary truthful either way: if Person B does start
        # emitting "capped_unresolved", the branch above takes over and this
        # one goes quiet. Safe because summarise_trace is only ever called on a
        # finished trace, never on a stream still in progress.
        return f"Stopped after {rounds} {plural} — ended with a gap still open."
    if final == "conflict_detected":
        return f"{rounds} {plural} — sources disagreed."
    if final == "baseline_no_check":
        return "Baseline: one retrieval, no sufficiency check."
    return f"{rounds} {plural} recorded."

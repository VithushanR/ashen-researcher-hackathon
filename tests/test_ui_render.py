"""Tests for the UI rendering helpers.

These run without Streamlit, a browser, or a server -- which is the reason
``render.py`` has no Streamlit import. The whole file executes in milliseconds,
so the display logic is cheap to test and stays tested.

Run from the repo root:

    pytest tests/test_ui_render.py -v
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.ui.render import (
    citation_location,
    confidence_style,
    conflict_claims,
    is_winning_claim,
    linkify_answer,
    status_badge,
    summarise_trace,
    tier_badge,
    tier_for,
    verdict_badge,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fake_api_response.json"

CITATIONS = [
    {"claim": "membership", "filename": "isolde_mournvale.md", "page": None},
    {"claim": "victory", "filename": "annals_of_the_ashen_era_vol2.pdf", "page": 47},
]


def cite_numbers(html: str) -> list[str]:
    """The citation numbers linked in a rendered answer, in order of appearance."""
    return re.findall(r'href="#cite-(\d+)"', html)


# ---------------------------------------------------------------------------
# linkify_answer
# ---------------------------------------------------------------------------

def test_multiple_citations_are_numbered_in_citation_order() -> None:
    rendered = linkify_answer(
        "Isolde is a member [isolde_mournvale.md], and the order won in 358 AS "
        "[annals_of_the_ashen_era_vol2.pdf, p.47].",
        CITATIONS,
    )
    assert cite_numbers(rendered) == ["1", "2"]
    assert ">1</sup>" in rendered and ">2</sup>" in rendered
    assert "[isolde_mournvale.md]" not in rendered, "the marker should be replaced by the link"


def test_a_repeated_citation_reuses_its_number() -> None:
    """Numbers index the evidence list, so the same source is always the same number."""
    rendered = linkify_answer(
        "First [isolde_mournvale.md] and again [isolde_mournvale.md].", CITATIONS
    )
    assert cite_numbers(rendered) == ["1", "1"]


def test_an_unmatched_marker_is_left_exactly_as_written() -> None:
    """Never delete a citation we failed to parse.

    Dropping it would make an unsupported sentence look supported, which is the
    precise failure this system exists to avoid. It stays visible but unlinked.
    """
    rendered = linkify_answer("A claim [unknown_source.pdf, p.9] here.", CITATIONS)
    assert "[unknown_source.pdf, p.9]" in rendered
    assert cite_numbers(rendered) == []


def test_markers_match_on_filename_and_page() -> None:
    """One document is routinely cited at several pages in a single answer.

    Matching on filename alone would point every marker at the first citation
    for that file, silently mis-attributing the claim to the wrong page.
    """
    citations = [
        {"claim": "campaigns", "filename": "annals.pdf", "page": 47},
        {"claim": "forging", "filename": "annals.pdf", "page": 112},
    ]
    rendered = linkify_answer("Forged [annals.pdf, p.112], won [annals.pdf, p.47].", citations)
    assert cite_numbers(rendered) == ["2", "1"]


def test_a_marker_without_a_page_falls_back_to_the_filename() -> None:
    citations = [
        {"claim": "a", "filename": "annals.pdf", "page": 47},
        {"claim": "b", "filename": "annals.pdf", "page": 112},
    ]
    assert cite_numbers(linkify_answer("Vague [annals.pdf] reference.", citations)) == ["1"]


@pytest.mark.parametrize(
    "payload",
    [
        '<script>alert("xss")</script>',
        "<img src=x onerror=alert(1)>",
        '<a href="http://evil.example">click</a>',
        "<b>bold</b> and <i>italic</i>",
    ],
)
def test_html_in_the_answer_is_escaped(payload: str) -> None:
    """The answer is model-generated and app.py renders it with unsafe_allow_html.

    Without escaping, markup produced by the model -- or copied out of an archive
    document -- would execute in the judge's browser.
    """
    rendered = linkify_answer(f"Answer {payload} [isolde_mournvale.md].", CITATIONS)
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "onerror" not in rendered or "&lt;img" in rendered
    assert "&lt;" in rendered, "the payload should appear escaped, not stripped"


def test_escaping_does_not_break_our_own_links() -> None:
    rendered = linkify_answer("<b>Bold</b> claim [isolde_mournvale.md].", CITATIONS)
    assert "<b>" not in rendered
    assert 'href="#cite-1"' in rendered


def test_newlines_become_line_breaks() -> None:
    assert "<br>" in linkify_answer("line one\nline two", CITATIONS)


def test_no_citations_returns_escaped_text_with_no_links() -> None:
    assert linkify_answer("Plain [x.pdf] answer.", []) == "Plain [x.pdf] answer."
    assert "<script>" not in linkify_answer("<script>bad</script>", [])


def test_empty_answer_is_empty() -> None:
    assert linkify_answer("", CITATIONS) == ""


# ---------------------------------------------------------------------------
# tier_for / tier_badge
# ---------------------------------------------------------------------------

def test_explicit_reliability_is_preferred() -> None:
    assert tier_for({"reliability": "T1_authoritative", "source_type": "ephemera"}) == "T1_authoritative"


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        ("codex", "T1_authoritative"),
        ("wiki", "T2_curated"),
        ("novel", "T3_narrative"),
        ("chronicles", "T3_narrative"),
        ("ephemera", "T4_unverified"),
    ],
)
def test_tier_falls_back_to_source_type(source_type: str, expected: str) -> None:
    """A citation that has passed through synthesis may have lost its tier."""
    assert tier_for({"source_type": source_type}) == expected


def test_source_type_matching_ignores_case_and_padding() -> None:
    assert tier_for({"source_type": "  CODEX  "}) == "T1_authoritative"


def test_an_unknown_source_gets_no_badge() -> None:
    """Blank, not "Unknown" -- a badge would imply we assessed the provenance."""
    assert tier_for({"source_type": "mystery"}) == ""
    assert tier_badge({"source_type": "mystery"}) == ""
    assert tier_badge({}) == ""


def test_badge_colour_is_ordered_by_authority() -> None:
    """Green for the official record down to red for an unverified letter.

    Lets a judge read trust at a glance in the conflict callout, before reading
    a word of the resolution.
    """
    assert "#1f7a4d" in tier_badge({"source_type": "codex"})
    assert "#a33a3a" in tier_badge({"source_type": "ephemera"})


def test_badge_escapes_its_label() -> None:
    assert "<" not in tier_badge({"reliability": "T1_authoritative"}).replace("<span", "").replace(
        "</span", ""
    )


# ---------------------------------------------------------------------------
# citation_location
# ---------------------------------------------------------------------------

def test_page_takes_priority() -> None:
    assert citation_location(
        {"filename": "annals.pdf", "page": 47, "section": "Campaigns"}
    ) == "annals.pdf, p.47"


def test_section_is_used_when_there_is_no_page() -> None:
    assert citation_location(
        {"filename": "isolde.md", "page": None, "section": "Affiliations"}
    ) == "isolde.md — Affiliations"


def test_filename_alone_is_the_last_resort() -> None:
    assert citation_location({"filename": "letter.txt", "page": None, "section": None}) == "letter.txt"
    assert citation_location({"filename": "letter.txt"}) == "letter.txt"


def test_page_zero_is_a_page_number_not_a_missing_value() -> None:
    """`if page:` would drop page 0; the contract says `is not None`."""
    assert citation_location({"filename": "a.pdf", "page": 0}) == "a.pdf, p.0"


def test_a_missing_filename_does_not_crash() -> None:
    assert citation_location({}) == "unknown source"


# ---------------------------------------------------------------------------
# conflict_claims / is_winning_claim
# ---------------------------------------------------------------------------

def test_current_multi_claim_shape_is_read() -> None:
    conflict = {"claims": [{"claim": "312 AS", "source": "letter"},
                           {"claim": "341 AS", "source": "codex"}]}
    assert [c["claim"] for c in conflict_claims(conflict)] == ["312 AS", "341 AS"]


def test_legacy_pairwise_shape_is_read() -> None:
    """The fixtures PDF predates the schema change; a rehearsal fixture may use it."""
    conflict = {"claim_a": "312 AS", "source_a": "letter",
                "claim_b": "341 AS", "source_b": "codex"}
    claims = conflict_claims(conflict)
    assert [c["claim"] for c in claims] == ["312 AS", "341 AS"]
    assert [c["source"] for c in claims] == ["letter", "codex"]


def test_legacy_three_way_conflict_is_read() -> None:
    conflict = {"claim_a": "a", "source_a": "1", "claim_b": "b", "source_b": "2",
                "claim_c": "c", "source_c": "3"}
    assert len(conflict_claims(conflict)) == 3


def test_a_legacy_claim_without_a_source_still_renders() -> None:
    assert conflict_claims({"claim_a": "312 AS"})[0]["source"] == "unknown"


def test_an_empty_conflict_yields_no_claims() -> None:
    assert conflict_claims({}) == []


def test_the_winning_claim_is_identified() -> None:
    assert is_winning_claim({"claim": "Forged in 341 AS"}, "341 AS") is True
    assert is_winning_claim({"claim": "Forged in 312 AS"}, "341 AS") is False


def test_an_unresolved_conflict_has_no_winner() -> None:
    """The UI must not imply a winner where the agent did not pick one."""
    assert is_winning_claim({"claim": "Forged in 341 AS"}, None) is False
    assert is_winning_claim({"claim": "Forged in 341 AS"}, "") is False


# ---------------------------------------------------------------------------
# confidence / status / verdict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(100, "Well supported"), (92, "Well supported"), (80, "Well supported"),
     (79, "Supported with caveats"), (55, "Supported with caveats"),
     (54, "Weakly supported"), (41, "Weakly supported"), (0, "Weakly supported")],
)
def test_confidence_bands(confidence: int, expected: str) -> None:
    assert confidence_style(confidence)[0] == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [("complete", "Complete"),
     ("complete_with_conflict", "Conflict resolved"),
     ("partial_gap_stated", "Partial — gap stated")],
)
def test_known_statuses(status: str, expected: str) -> None:
    assert status_badge(status)[0] == expected


def test_an_unknown_status_degrades_readably() -> None:
    assert status_badge("some_new_status")[0] == "Some New Status"


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("sufficient", "Enough evidence"),
     ("insufficient", "Not enough yet"),
     ("conflict_detected", "Conflict detected"),
     ("capped_unresolved", "Iteration cap reached"),
     ("baseline_no_check", "No sufficiency check")],
)
def test_known_verdicts(verdict: str, expected: str) -> None:
    assert verdict_badge(verdict)[0] == expected


def test_an_unstyled_verdict_from_person_b_still_renders() -> None:
    """Person B has not fixed their verdict vocabulary; the panel must not break."""
    assert verdict_badge("needs_more_hops")[0] == "Needs More Hops"
    assert verdict_badge(None)[0] == "Searching"


# ---------------------------------------------------------------------------
# summarise_trace
# ---------------------------------------------------------------------------

def test_an_empty_trace_says_so() -> None:
    assert summarise_trace([]) == "No research rounds recorded."


def test_a_sufficient_run_reports_why_it_stopped() -> None:
    """The claim of the whole system: it stops on evidence, not on first hit."""
    summary = summarise_trace(
        [{"step": 1, "verdict": "insufficient"}, {"step": 2, "verdict": "sufficient"}]
    )
    assert summary == "Stopped after 2 rounds — enough evidence."


def test_a_capped_run_admits_the_remaining_gap() -> None:
    summary = summarise_trace([{"step": i, "verdict": "insufficient"} for i in range(1, 5)]
                              + [{"step": 5, "verdict": "capped_unresolved"}])
    assert summary == "Stopped after 5 rounds — iteration cap reached with a gap remaining."


def test_a_baseline_run_is_labelled_as_such() -> None:
    assert summarise_trace([{"step": 1, "verdict": "baseline_no_check"}]) == (
        "Baseline: one retrieval, no sufficiency check."
    )


def test_a_single_round_is_singular() -> None:
    assert "1 round " in summarise_trace([{"step": 1, "verdict": "sufficient"}])


def test_an_unknown_final_verdict_still_summarises() -> None:
    assert summarise_trace(
        [{"step": 1, "verdict": "x"}, {"step": 2, "verdict": "??"}]
    ) == "2 rounds recorded."


# ---------------------------------------------------------------------------
# Against the real fixtures
# ---------------------------------------------------------------------------

def fixture_cases() -> list[tuple[str, dict]]:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return [(name, case) for name, case in data.items() if not name.startswith("_")]


@pytest.mark.parametrize(("name", "case"), fixture_cases())
def test_every_fixture_marker_resolves_to_a_citation(name: str, case: dict) -> None:
    """An unlinked marker in our own fixtures means the demo shows raw brackets."""
    rendered = linkify_answer(case["answer"], case["citations"])
    markers = len(re.findall(r"\[[^\[\]]+\]", case["answer"]))
    assert len(cite_numbers(rendered)) == markers, f"{name}: unresolved citation marker"


@pytest.mark.parametrize(("name", "case"), fixture_cases())
def test_every_fixture_citation_has_a_tier_badge(name: str, case: dict) -> None:
    for citation in case["citations"]:
        assert tier_badge(citation), f"{name}: no tier for {citation['filename']}"


@pytest.mark.parametrize(("name", "case"), fixture_cases())
def test_every_fixture_conflict_has_exactly_one_winner(name: str, case: dict) -> None:
    for conflict in case["conflicts"]:
        claims = conflict_claims(conflict)
        winners = [c for c in claims if is_winning_claim(c, conflict.get("resolved_value"))]
        assert len(claims) >= 2, f"{name}: a conflict needs competing claims"
        assert len(winners) == 1, f"{name}: expected one winner, got {len(winners)}"

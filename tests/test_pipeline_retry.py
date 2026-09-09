"""_compose_with_retry / _research_with_retry: retry by ORIGIN, not message text.

Both wrap a strict structured-output validator (coverage_validation.py,
sufficiency.py) that occasionally rejects a real model response because one
of its own JSON fields doesn't logically cohere with another. Three distinct
exact messages from this same family have been observed across two model
providers, so the retry is scoped by which module actually raised the error
(via the traceback), not by matching accumulated exact strings.

Every error here is produced by the REAL function that raises it in
production, fed a crafted fake model response -- not a hand-built exception --
so a change to how these validators raise things is what these tests would
actually catch.
"""
import json
from unittest.mock import Mock

import pytest

import agent.conflict as conflict
import agent.sufficiency as sufficiency
from agent.state import Evidence, ResearchState
from src.answer.composer import compose_answer
from src.answer.coverage_validation import CitationCoverageError, validate_answer_coverage
from src.answer.synthesis import CitationClaim
from src.api import pipeline


def _real_error(fn) -> ValueError:
    with pytest.raises(ValueError) as excinfo:
        fn()
    return excinfo.value


def _mapped_claim_mismatch():
    validate_answer_coverage(
        "Some answer.",
        [CitationClaim(claim="A claim not present in the answer text at all.", chunk_id="e1")],
        validate_coverage=lambda prompt: {
            "coverage": "complete", "uncovered_claims": [], "reason": None,
            "presentation": [{"requirement": "Q", "status": "answered", "answer_excerpt": "Some answer.",
                "evidence_ids": [], "citation_claim_indices": [0], "gap_indices": [],
                "conflict_indices": [], "limitation_indices": []}],
        },
        unresolved_claims=[], question="Q", required_claims=["Q"],
        evidence=[{"chunk_id": "e1", "text": "x"}], ordinary_answer="Some answer.",
    )


def _omitted_with_excerpt():
    validate_answer_coverage(
        "Some answer.", [CitationClaim(claim="Some answer.", chunk_id="e1")],
        validate_coverage=lambda prompt: {
            "coverage": "complete", "uncovered_claims": [], "reason": None,
            "presentation": [{"requirement": "Q", "status": "omitted", "answer_excerpt": "should not be here",
                "evidence_ids": ["e1"], "citation_claim_indices": [], "gap_indices": [],
                "conflict_indices": [], "limitation_indices": []}],
        },
        unresolved_claims=[], question="Q", required_claims=["Q"],
        evidence=[{"chunk_id": "e1", "text": "x"}], ordinary_answer="Some answer.",
    )


def _incomplete_coverage():
    validate_answer_coverage(
        "Some answer.", [CitationClaim(claim="Some answer.", chunk_id="e1")],
        validate_coverage=lambda prompt: {
            "coverage": "incomplete", "uncovered_claims": ["x"], "reason": "missing something",
            "presentation": [{"requirement": "Q", "status": "answered", "answer_excerpt": "Some answer.",
                "evidence_ids": [], "citation_claim_indices": [0], "gap_indices": [],
                "conflict_indices": [], "limitation_indices": []}],
        },
        unresolved_claims=[], question="Q", required_claims=["Q"],
        evidence=[{"chunk_id": "e1", "text": "x"}], ordinary_answer="Some answer.",
    )


def _verdict_contradiction():
    state = ResearchState(question="Q", required_claims=["Q"])
    raw = json.dumps({
        "coverage": "yes", "agreement": "yes", "verdict": "insufficient", "missing_info": None,
        "requirements": [{"requirement": "Q", "status": "supported", "missing_info": None}],
    })
    original = sufficiency.call_llm
    sufficiency.call_llm = lambda prompt: raw
    try:
        sufficiency.check_sufficiency(state)
    finally:
        sufficiency.call_llm = original


def _conflict_malformed_output():
    original = conflict.call_llm
    conflict.call_llm = lambda prompt: "not valid json"
    try:
        conflict.detect_conflicts([
            Evidence(chunk_id="a", filename="a.md", source_type="wiki", reliability="T2_curated",
                     text="x", content_type="text", page=None, section=None),
            Evidence(chunk_id="b", filename="b.md", source_type="wiki", reliability="T2_curated",
                     text="y", content_type="text", page=None, section=None),
        ])
    finally:
        conflict.call_llm = original


@pytest.mark.parametrize("trigger", [_mapped_claim_mismatch, _omitted_with_excerpt],
                          ids=["mapped-claim-mismatch", "omitted-uncertain-pydantic"])
def test_compose_retries_any_coverage_validation_self_consistency_error(trigger):
    error = _real_error(trigger)
    compose = Mock(side_effect=[error, "SUCCESS"])
    assert pipeline._compose_with_retry(compose, state=object(), question="Q") == "SUCCESS"
    assert compose.call_count == 2


def test_research_retries_sufficiency_self_consistency_error():
    error = _real_error(_verdict_contradiction)
    research = Mock(side_effect=[error, "SUCCESS_STATE"])
    assert pipeline._research_with_retry(research, question="Q") == "SUCCESS_STATE"
    assert research.call_count == 2


def test_compose_does_not_retry_composers_own_bug():
    error = _real_error(lambda: compose_answer(
        ResearchState(question="Q"), synthesize=Mock(), validate_coverage=Mock()))
    compose = Mock(side_effect=[error, "SHOULD_NOT_BE_CALLED"])
    with pytest.raises(ValueError):
        pipeline._compose_with_retry(compose, state=object(), question="Q")
    assert compose.call_count == 1


def test_compose_does_not_retry_citation_coverage_error():
    """Coverage genuinely incomplete is not the same as the verdict self-
    contradicting -- composer.py already gives this its own repair-with-
    feedback attempt, so blindly retrying here would blur that distinction."""
    error = _real_error(_incomplete_coverage)
    assert isinstance(error, CitationCoverageError)
    compose = Mock(side_effect=[error, "SHOULD_NOT_BE_CALLED"])
    with pytest.raises(CitationCoverageError):
        pipeline._compose_with_retry(compose, state=object(), question="Q")
    assert compose.call_count == 1


def test_compose_does_not_retry_transport_errors():
    compose = Mock(side_effect=[RuntimeError("LLM call failed after 4 attempts"), "SHOULD_NOT_BE_CALLED"])
    with pytest.raises(RuntimeError):
        pipeline._compose_with_retry(compose, state=object(), question="Q")
    assert compose.call_count == 1


def test_research_does_not_retry_conflicts_own_bug():
    """A malformed-output error from conflict.py is not from sufficiency.py --
    only the module actually named in this task is in scope."""
    error = _real_error(_conflict_malformed_output)
    research = Mock(side_effect=[error, "SHOULD_NOT_BE_CALLED"])
    with pytest.raises(ValueError):
        pipeline._research_with_retry(research, question="Q")
    assert research.call_count == 1

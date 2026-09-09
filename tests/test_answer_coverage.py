"""Runtime coverage gates with deterministic adapters and synthetic answers."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from src.answer.coverage_validation import CitationCoverageError, CoverageVerdict
from tests.answer_helpers import complete_coverage, supported_semantics
from tests.test_answer_composer import evidence, state
from tests.test_answer_conflicts import conflict, research
from src.answer.semantic_validation import CitationSupportError


EVENT = "Hesper was evasive during interrogation."


def output(answer=EVENT, claims=None):
    return {"answer": answer, "citation_claims": (
        [{"claim": EVENT, "chunk_id": "event"}] if claims is None else claims
    )}


def input_state(**overrides):
    return state(evidence("event", text=EVENT), **overrides)


def data(prompt):
    return json.loads(prompt.split("INPUT DATA:\n", 1)[1])


def test_complete_coverage_receives_answer_and_claims_before_semantics():
    calls = []

    def synthesize(prompt):
        calls.append("synthesis")
        return output()

    def coverage(prompt):
        calls.append("coverage")
        assert data(prompt) == {
            "question": input_state().question, "required_claims": [], "ordinary_answer": EVENT,
            "evidence": [{"chunk_id": "event", "text": EVENT}],
            "answer": EVENT, "citation_claims": output()["citation_claims"],
            "research_context": {"unresolved_claims": [], "conflicts": [],
                                 "no_evidence_returned": False},
        }
        return complete_coverage(prompt)

    def semantics(prompt):
        calls.append("semantics")
        return supported_semantics(prompt)

    result = compose_answer(input_state(), synthesize=synthesize,
                            validate_coverage=coverage, validate_semantics=semantics)
    assert calls == ["synthesis", "coverage", "semantics"]
    assert result.status == "complete"
    assert result.answer == EVENT


@pytest.mark.parametrize("answer", [
    EVENT + " Hesper had a guarded personality.",
    "Hesper was evasive during interrogation and had a guarded personality.",
])
def test_omitted_sentence_or_extra_clause_rejects_answer_without_rewriting(answer):
    synthesized = output(answer)
    before = deepcopy(synthesized)
    semantics = Mock()

    def coverage(prompt):
        payload = data(prompt)
        assert payload["answer"] == answer
        assert payload["citation_claims"] == output()["citation_claims"]
        assert "multiple ideas in a\nsingle sentence or conjunction" in prompt
        assert "simple word overlap does not" in prompt
        return {"coverage": "incomplete", "uncovered_claims": ["Hesper had a guarded personality."]}

    with pytest.raises(CitationCoverageError) as error:
        compose_answer(input_state(), synthesize=Mock(return_value=synthesized),
                       validate_coverage=coverage, validate_semantics=semantics)
    assert error.value.answer == answer
    assert error.value.verdict.uncovered_claims == ["Hesper had a guarded personality."]
    assert synthesized == before
    semantics.assert_not_called()


def test_zero_declared_claims_with_factual_answer_cannot_bypass_gate():
    # Partial synthesis permits zero claims structurally; coverage must still run.
    semantics = Mock()
    coverage = Mock(return_value={"coverage": "incomplete", "uncovered_claims": [EVENT]})
    with pytest.raises(CitationCoverageError):
        compose_answer(input_state(unresolved_claims=["Unknown date."]),
                       synthesize=Mock(return_value=output(claims=[])),
                       validate_coverage=coverage, validate_semantics=semantics)
    assert data(coverage.call_args.args[0])["citation_claims"] == []
    coverage.assert_called_once()
    semantics.assert_not_called()


def test_nonfactual_structural_wording_and_authoritative_gap_need_no_fake_claim():
    gap = "The date is not established by retrieved evidence."

    def coverage(prompt):
        payload = data(prompt)
        assert payload["answer"].startswith("Here is the available information.")
        assert gap in payload["answer"]
        assert payload["research_context"]["unresolved_claims"] == [gap]
        assert payload["citation_claims"] == []
        assert "purely non-factual uncertainty wording need no claim" in prompt
        assert "Hedging an actual factual assertion" in prompt
        return complete_coverage(prompt)

    result = compose_answer(
        input_state(unresolved_claims=[gap]),
        synthesize=Mock(return_value=output("Here is the available information.", [])),
        validate_coverage=coverage,
    )
    assert result.status == "partial_gap_stated"
    assert result.citations == []


def test_explicit_absence_requires_and_accepts_represented_absence_claim():
    claim = "The archive does not establish Hesper's canonical temperament."

    def coverage(prompt):
        payload = data(prompt)
        assert payload["answer"] == claim
        assert payload["citation_claims"][0]["claim"] == claim
        assert "needs a declared absence claim" in prompt
        return complete_coverage(prompt)

    result = compose_answer(
        state(evidence("absence", text="Hesper: No canonical temperament is established.")),
        synthesize=Mock(return_value=output(claim, [{"claim": claim, "chunk_id": "absence"}])),
        validate_coverage=coverage, validate_semantics=supported_semantics,
    )
    assert result.status == "complete"
    assert result.citations[0]["claim"] == claim


@pytest.mark.parametrize("verdict", [
    None, "complete", True, {}, {"coverage": True}, {"coverage": "yes"},
    {"coverage": "complete", "uncovered_claims": "none"},
    {"coverage": "complete", "reason": 123},
    {"coverage": "complete", "answer": "Replacement"},
    {"coverage": "complete", "uncovered_claims": ["Omitted fact"]},
    {"coverage": "incomplete", "uncovered_claims": [123]},
    CoverageVerdict.model_construct(coverage="invalid"),
])
def test_malformed_coverage_fails_before_semantic_validation(verdict):
    semantics = Mock()
    with pytest.raises(ValidationError):
        compose_answer(input_state(), synthesize=Mock(return_value=output()),
                       validate_coverage=Mock(return_value=verdict), validate_semantics=semantics)
    semantics.assert_not_called()


def test_uncertain_coverage_fails_safely():
    with pytest.raises(CitationCoverageError):
        compose_answer(input_state(), synthesize=Mock(return_value=output()),
                       validate_coverage=Mock(return_value={"coverage": "uncertain"}))


def test_missing_coverage_adapter_is_not_a_bypass():
    semantics = Mock()
    with pytest.raises(ValueError, match="coverage-validation adapter is required"):
        compose_answer(input_state(), synthesize=Mock(return_value=output()),
                       validate_semantics=semantics)
    semantics.assert_not_called()


def test_adapter_exception_propagates_without_answer():
    with pytest.raises(RuntimeError, match="Coverage unavailable"):
        compose_answer(input_state(), synthesize=Mock(return_value=output()),
                       validate_coverage=Mock(side_effect=RuntimeError("Coverage unavailable")))


def test_unknown_chunk_still_fails_after_coverage_before_semantics():
    coverage = Mock(side_effect=complete_coverage)
    semantics = Mock()
    with pytest.raises(ValueError, match="Unknown synthesis chunk_id: missing"):
        compose_answer(input_state(), synthesize=Mock(return_value=output(
            EVENT, [{"claim": EVENT, "chunk_id": "missing"}]
        )), validate_coverage=coverage, validate_semantics=semantics)
    coverage.assert_called_once()
    semantics.assert_not_called()


def test_coverage_pass_does_not_override_failed_support():
    coverage = Mock(side_effect=complete_coverage)
    with pytest.raises(CitationSupportError):
        compose_answer(input_state(), synthesize=Mock(return_value=output()),
                       validate_coverage=coverage, validate_semantics=Mock(return_value={
                           "support": "unsupported", "explicit_absence": "clear",
                       }))
    coverage.assert_called_once()


def test_no_evidence_gap_answer_still_requires_coverage():
    with pytest.raises(ValueError, match="coverage-validation adapter is required"):
        compose_answer(state(unresolved_claims=["Unknown date."]), synthesize=Mock())


def test_no_evidence_report_is_checked_with_authoritative_context():
    coverage = Mock(side_effect=complete_coverage)
    result = compose_answer(state(unresolved_claims=["Unknown date."]),
                            synthesize=Mock(), validate_coverage=coverage)
    payload = data(coverage.call_args.args[0])
    assert payload["answer"] == result.answer
    assert payload["research_context"]["no_evidence_returned"] is True
    assert payload["citation_claims"] == []


@pytest.mark.parametrize("resolved_value", ["green", None])
def test_conflict_report_coverage_preserves_upstream_decisions(resolved_value):
    supplied = conflict(resolved_value=resolved_value)
    coverage = Mock(side_effect=complete_coverage)
    synthesize = Mock(return_value={"answer": None, "citation_claims": []})
    semantics = Mock(side_effect=supported_semantics)
    result = compose_answer(research(supplied), synthesize=synthesize,
                            validate_coverage=coverage, validate_semantics=semantics)
    payload = data(coverage.call_args.args[0])
    assert payload["answer"] == result.answer
    assert payload["research_context"]["conflicts"] == result.conflicts
    assert payload["research_context"]["conflicts"][0]["resolved_value"] == resolved_value
    assert len(payload["citation_claims"]) == 2
    assert result.status == ("partial_gap_stated" if resolved_value is None else "complete_with_conflict")
    synthesize.assert_called_once()
    assert semantics.call_count == 2


def test_conflict_report_cannot_bypass_missing_or_failing_coverage():
    with pytest.raises(ValueError, match="coverage-validation adapter is required"):
        compose_answer(research(conflict()), synthesize=Mock(return_value={"answer": None, "citation_claims": []}))
    with pytest.raises(CitationCoverageError):
        compose_answer(research(conflict()), synthesize=Mock(return_value={"answer": None, "citation_claims": []}),
                       validate_coverage=Mock(return_value={"coverage": "uncertain"}))

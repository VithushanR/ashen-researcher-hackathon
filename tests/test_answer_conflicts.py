"""Synthetic conflicts test presentation, not Person B's resolution policy."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from src.answer.semantic_validation import CitationSupportError
from tests.answer_helpers import complete_coverage, supported_semantics
from tests.test_answer_composer import evidence, state


def conflict(resolved_value="green", resolution="Upstream supplied reasoning", count=2):
    return SimpleNamespace(
        attribute="test gate color",
        claims=[SimpleNamespace(claim=f"The gate is {color}.", source=f"Test source {i}",
                                chunk_id=f"test_{i}")
                for i, color in enumerate(["blue", "green", "red"][:count])],
        resolution=resolution, resolved_value=resolved_value,
    )


def research(*conflicts, **overrides):
    return state(
        *[evidence(f"test_{i}", filename=f"source_{i}.md", page=None,
                   section=f"Section {i}", source_type="wiki",
                   text=f"Synthetic source reports the gate is {color}.")
          for i, color in enumerate(["blue", "green", "red"])],
        conflicts=list(conflicts), **overrides,
    )


def test_resolved_conflict_uses_upstream_result_and_preserves_metadata():
    supplied = conflict()
    input_state = research(supplied)
    before = deepcopy(input_state)
    synthesize = Mock(side_effect=AssertionError("Must not ask an LLM to resolve conflicts"))

    result = compose_answer(input_state, validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.status == "complete_with_conflict"
    assert "Source conflict concerning test gate color" in result.answer
    assert "The research result prefers: green." in result.answer
    assert "Supplied resolution: Upstream supplied reasoning" in result.answer
    assert result.conflicts == [{
        "attribute": supplied.attribute,
        "claims": [vars(claim) for claim in supplied.claims],
        "resolution": supplied.resolution, "resolved_value": "green",
    }]
    assert result.question == input_state.question
    assert result.confidence == input_state.confidence
    assert result.iterations_used == input_state.iteration
    for i, citation in enumerate(result.citations):
        assert citation == {
            "claim": f'Test source {i} reports: "{supplied.claims[i].claim}"',
            "filename": f"source_{i}.md", "page": None,
            "section": f"Section {i}", "source_type": "wiki",
        }
        assert citation["claim"] in result.answer
    assert input_state == before
    synthesize.assert_not_called()


def test_unresolved_conflict_does_not_treat_resolution_text_as_a_winner():
    supplied = conflict(resolved_value=None, resolution="Tentative preference: green")
    synthesize = Mock(return_value={"answer": "Green wins"})

    result = compose_answer(research(supplied), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.status == "partial_gap_stated"
    assert "evidence remains contradictory" in result.answer
    assert "No preferred result has been established" in result.answer
    assert "The research result prefers:" not in result.answer
    assert "Tentative preference" not in result.answer
    assert result.conflicts[0]["resolved_value"] is None
    assert result.conflicts[0]["resolution"] == supplied.resolution
    assert result.conflicts[0]["claims"] == [vars(claim) for claim in supplied.claims]
    synthesize.assert_not_called()


@pytest.mark.parametrize("resolved_value", ["red", None])
def test_three_competing_claims_are_all_preserved(resolved_value):
    supplied = conflict(resolved_value=resolved_value, count=3)
    result = compose_answer(research(supplied), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock())
    assert len(result.conflicts[0]["claims"]) == 3
    assert len(result.citations) == 3
    for claim in supplied.claims:
        assert claim.claim in result.answer
        assert claim.source in result.answer
    assert result.conflicts[0]["claims"] == [vars(claim) for claim in supplied.claims]


def test_mixed_conflicts_remain_partial_and_preserve_both_decisions():
    result = compose_answer(
        research(conflict(), conflict(resolved_value=None)), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock()
    )
    assert result.status == "partial_gap_stated"
    assert "The research result prefers: green." in result.answer
    assert "evidence remains contradictory" in result.answer
    assert [item["resolved_value"] for item in result.conflicts] == ["green", None]


def test_resolved_value_without_explanation_does_not_invent_reason():
    result = compose_answer(research(conflict(resolution=None)), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock())
    assert result.status == "complete_with_conflict"
    assert result.conflicts[0]["resolution"] is None
    assert "Supplied resolution:" not in result.answer


def test_missing_chunk_id_keeps_claim_without_fabricating_citation():
    supplied = conflict()
    supplied.claims[1].chunk_id = None
    result = compose_answer(research(supplied), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock())
    assert len(result.citations) == 1
    assert result.conflicts[0]["claims"][1]["chunk_id"] is None
    assert supplied.claims[1].claim in result.answer
    assert "No evidence chunk reference was supplied" in result.answer


def test_unknown_conflict_chunk_id_is_rejected():
    supplied = conflict()
    supplied.claims[0].chunk_id = "invented"
    with pytest.raises(ValueError, match="Unknown conflict chunk_id: invented"):
        compose_answer(research(supplied), validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock())


def test_unresolved_conflict_preserves_reported_gaps():
    result = compose_answer(
        research(conflict(resolved_value=None), unresolved_claims=["Gate color is disputed."]),
        validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock(),
    )
    assert result.status == "partial_gap_stated"
    assert "Gate color is disputed." in result.answer


def test_upstream_result_is_not_overridden_by_source_tier():
    input_state = research(conflict(resolved_value="green"))
    input_state.evidence[0].reliability = "T1_authoritative"
    input_state.evidence[1].reliability = "T4_unverified"
    result = compose_answer(input_state, validate_coverage=complete_coverage, validate_semantics=supported_semantics, synthesize=Mock())
    assert "The research result prefers: green." in result.answer


# These adapters test orchestration and prompt contracts, not live model accuracy.


def compose_checked(input_state, validator):
    return compose_answer(input_state, synthesize=Mock(),
                          validate_coverage=complete_coverage, validate_semantics=validator)


@pytest.mark.parametrize("resolved_value", ["green", None])
def test_each_raw_competing_claim_checks_actual_passage_after_coverage(resolved_value):
    supplied = conflict(resolved_value=resolved_value, count=3)
    input_state = research(supplied)
    before = deepcopy(input_state)
    calls = []

    def coverage(prompt):
        calls.append("coverage")
        return {"coverage": "complete"}

    def semantic(prompt):
        payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
        index = len(calls) - 1
        assert payload["claim"] == supplied.claims[index].claim
        assert payload["referenced_evidence"] == {
            "chunk_id": supplied.claims[index].chunk_id,
            "text": input_state.evidence[index].text,
        }
        assert "Conflict attribution context:" in prompt
        assert "not which\nclaim is globally correct" in prompt
        assert "different competing source disagreeing does not" in prompt
        assert "genuine direct\nevidence disagreement needing Person B" not in prompt
        calls.append("semantic")
        return supported_semantics(prompt)

    result = compose_answer(input_state, synthesize=Mock(),
                            validate_coverage=coverage, validate_semantics=semantic)
    assert calls == ["coverage", "semantic", "semantic", "semantic"]
    assert result.conflicts[0]["resolved_value"] == resolved_value
    assert result.conflicts[0]["resolution"] == supplied.resolution
    assert result.status == ("partial_gap_stated" if resolved_value is None else "complete_with_conflict")
    assert input_state == before


@pytest.mark.parametrize("support", ["unsupported", "contradicted", "uncertain"])
def test_one_rejected_competing_claim_aborts_answer(support):
    input_state = research(conflict())
    input_state.evidence[1].text = "The castle gates were made of iron."
    before = deepcopy(input_state)
    validator = Mock(side_effect=[supported_semantics(""),
                                 {"support": support, "explicit_absence": "clear"}])
    with pytest.raises(CitationSupportError) as error:
        compose_checked(input_state, validator)
    assert validator.call_count == 2
    assert error.value.claim == input_state.conflicts[0].claims[1].claim
    payload = json.loads(validator.call_args.args[0].split("INPUT DATA:\n", 1)[1])
    assert payload["referenced_evidence"]["text"] == input_state.evidence[1].text
    assert input_state == before


@pytest.mark.parametrize("output", [None, {}, {"support": "supported"},
                                     {"support": "supported", "explicit_absence": "invalid"}])
def test_malformed_conflict_verdict_fails(output):
    with pytest.raises(ValidationError):
        compose_checked(research(conflict()), Mock(return_value=output))


def test_missing_conflict_semantic_adapter_fails():
    with pytest.raises(ValueError, match="semantic-validation adapter is required"):
        compose_checked(research(conflict()), None)


def test_conflict_semantic_exception_propagates():
    with pytest.raises(RuntimeError, match="unavailable"):
        compose_checked(research(conflict()), Mock(side_effect=RuntimeError("unavailable")))


@pytest.mark.parametrize("failure", ["unknown", "duplicate", "coverage"])
def test_conflict_prechecks_prevent_semantic_calls(failure):
    input_state = research(conflict())
    coverage = Mock(return_value={"coverage": "incomplete" if failure == "coverage" else "complete"})
    if failure == "unknown":
        input_state.conflicts[0].claims[-1].chunk_id = "invented"
    elif failure == "duplicate":
        input_state.evidence.append(input_state.evidence[0])
    semantic = Mock()
    with pytest.raises(ValueError):
        compose_answer(input_state, synthesize=Mock(), validate_coverage=coverage,
                       validate_semantics=semantic)
    semantic.assert_not_called()
    if failure != "coverage":
        coverage.assert_not_called()


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_references_have_no_citation_or_semantic_call(all_missing):
    supplied = conflict()
    supplied.claims[0].chunk_id = None
    if all_missing:
        supplied.claims[1].chunk_id = None
    validator = Mock(side_effect=supported_semantics)
    result = compose_checked(research(supplied), validator)
    assert len(result.citations) == validator.call_count == (0 if all_missing else 1)
    assert "No evidence chunk reference was supplied" in result.answer
    assert result.conflicts[0]["claims"] == [vars(c) for c in supplied.claims]


@pytest.mark.parametrize("claim,text,absence_status", [
    ("Hesper has a guarded temperament.", "Hesper was evasive during interrogation.", "violated"),
    ("Hesper's temperament does not exist.", "No canonical temperament is established.", "violated"),
    ("No canonical temperament is established.", "No canonical temperament is established.", "clear"),
    ("Hesper was evasive during interrogation.", "Hesper was evasive during interrogation.", "clear"),
])
def test_conflict_absence_compatibility(claim, text, absence_status):
    supplied = conflict()
    supplied.claims[0].claim = claim
    input_state = research(supplied)
    input_state.evidence[0].text = text
    absence = "No canonical temperament is established."
    input_state.evidence.append(evidence("absence", text="Hesper: " + absence, entities=["Hesper"]))

    def validator(prompt):
        payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
        assert any(item["chunk_id"] == "absence" for item in payload["evidence"])
        assert "Explicit absence still prohibits reconstructing" in prompt
        if payload["claim"] == claim:
            return {"support": "supported", "explicit_absence": absence_status,
                    "absence_findings": [{"chunk_id": "absence", "passage": absence}]}
        return supported_semantics(prompt)

    if absence_status == "violated":
        with pytest.raises(CitationSupportError):
            compose_checked(input_state, validator)
    else:
        assert compose_checked(input_state, validator).status == "complete_with_conflict"

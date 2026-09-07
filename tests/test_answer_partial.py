"""Partial-answer contract tests with synthetic evidence and injected synthesis."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from tests.answer_helpers import supported_semantics
from src.answer.synthesis import PartialSynthesisResult, SynthesisResult
from tests.test_answer_composer import evidence, state
from tests.test_answer_conflicts import conflict, research


SIGHTING = "The test creature was last sighted in Test Vale."
GAP = "The location of the test creature's lair is not established."


def partial_state(**overrides):
    values = dict(unresolved_claims=[GAP], confidence=41, iteration=5)
    values.update(overrides)
    return state(evidence(text=SIGHTING), **values)


def supported_result():
    return {"answer": SIGHTING,
            "citation_claims": [{"claim": SIGHTING, "chunk_id": "test_a"}]}


def test_partial_fact_and_gap_preserve_metadata_and_only_cite_supported_fact():
    input_state = partial_state()
    before = deepcopy(input_state)
    result = compose_answer(input_state, validate_semantics=supported_semantics, synthesize=Mock(return_value=supported_result()))

    assert result.status == "partial_gap_stated"
    assert result.question == input_state.question
    assert result.confidence == 41
    assert result.iterations_used == 5
    assert result.conflicts == []
    assert SIGHTING in result.answer
    assert "Unresolved information (not established by research):" in result.answer
    assert GAP in result.answer
    assert result.citations == [{
        "claim": SIGHTING, "filename": "synthetic.pdf", "page": 7,
        "section": "Test section", "source_type": "codex",
    }]
    assert input_state == before


def test_multiple_authoritative_gaps_are_kept_even_if_synthesis_omits_them():
    gaps = [GAP, "The creature's age is unknown."]
    result = compose_answer(partial_state(unresolved_claims=gaps),
                            validate_semantics=supported_semantics, synthesize=Mock(return_value=supported_result()))
    for gap in gaps:
        assert f"- {gap}" in result.answer
        assert all(citation["claim"] != gap for citation in result.citations)


def test_partial_prompt_separates_gaps_and_forbids_strengthening_evidence():
    synthesize = Mock(return_value=supported_result())
    result = compose_answer(partial_state(), validate_semantics=supported_semantics, synthesize=synthesize)
    prompt = synthesize.call_args.args[0]
    payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])

    assert payload["unresolved_claims"] == [GAP]
    assert payload["evidence"] == [{"chunk_id": "test_a", "text": SIGHTING}]
    assert "not evidence" in prompt
    assert "'last sighted in X' does not establish 'lair is in X'" in prompt
    assert "Only supported factual assertions belong in citation_claims" in prompt
    # Verifies faithful handling of injected synthesis, not semantic entailment.
    assert result.answer.startswith(SIGHTING)
    assert "lair is in Test Vale" not in result.answer
    assert result.citations[0]["claim"] == SIGHTING


def test_partial_status_is_not_inferred_from_high_confidence_or_low_iteration():
    result = compose_answer(partial_state(confidence=100, iteration=1),
                            validate_semantics=supported_semantics, synthesize=Mock(return_value=supported_result()))
    assert result.status == "partial_gap_stated"
    assert result.confidence == 100
    assert result.iterations_used == 1


def test_no_evidence_returns_gaps_without_synthesis_or_citations():
    synthesize = Mock()
    result = compose_answer(state(unresolved_claims=[GAP]), validate_semantics=supported_semantics, synthesize=synthesize)
    assert result.status == "partial_gap_stated"
    assert result.citations == []
    assert GAP in result.answer
    synthesize.assert_not_called()


def test_no_supported_fact_does_not_require_fabricated_citation():
    synthesize = Mock(return_value={
        "answer": "The retrieved material does not establish an answer.", "citation_claims": [],
    })
    result = compose_answer(partial_state(), validate_semantics=supported_semantics, synthesize=synthesize)
    assert result.citations == []
    assert GAP in result.answer
    assert result.status == "partial_gap_stated"


def test_unknown_chunk_in_partial_synthesis_is_rejected():
    output = supported_result()
    output["citation_claims"][0]["chunk_id"] = "invented"
    with pytest.raises(ValueError, match="Unknown synthesis chunk_id: invented"):
        compose_answer(partial_state(), validate_semantics=supported_semantics, synthesize=Mock(return_value=output))


def test_partial_synthesis_still_rejects_generated_citation_metadata():
    output = supported_result()
    output["citation_claims"][0]["filename"] = "invented.pdf"
    with pytest.raises(ValidationError):
        compose_answer(partial_state(), validate_semantics=supported_semantics, synthesize=Mock(return_value=output))


@pytest.mark.parametrize("model", [SynthesisResult, PartialSynthesisResult])
def test_partial_accepts_validated_synthesis_boundary_result(model):
    result = compose_answer(partial_state(), validate_semantics=supported_semantics, synthesize=Mock(return_value=model(**supported_result())))
    assert result.status == "partial_gap_stated"
    assert result.citations[0]["claim"] == SIGHTING


def test_resolved_conflict_with_remaining_gap_is_partial_without_changing_resolution():
    synthesize = Mock()
    result = compose_answer(research(conflict(), unresolved_claims=[GAP]), validate_semantics=supported_semantics, synthesize=synthesize)
    assert result.status == "partial_gap_stated"
    assert result.conflicts[0]["resolved_value"] == "green"
    assert "The research result prefers: green." in result.answer
    assert GAP in result.answer
    assert all(citation["claim"] != GAP for citation in result.citations)
    synthesize.assert_not_called()

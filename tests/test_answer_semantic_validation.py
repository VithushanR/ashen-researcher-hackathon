"""Deterministic adapter verdicts exercise real fail-closed runtime validation."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from tests.answer_helpers import complete_coverage
from src.answer.semantic_validation import CitationSupportError, SemanticVerdict, validate_citation_claim
from tests.test_answer_composer import evidence, state


def synthesis(claim, chunk_id="event"):
    return {"answer": claim, "citation_claims": [{"claim": claim, "chunk_id": chunk_id}]}


def verdict(support="supported", explicit_absence="clear", **extra):
    return {"support": support, "explicit_absence": explicit_absence, **extra}


def data(prompt):
    return json.loads(prompt.split("INPUT DATA:\n", 1)[1])


def test_direct_support_checks_real_text_and_preserves_answer_and_metadata():
    claim = "Hesper was evasive during interrogation."
    item = evidence("event", text=claim, entities=["Hesper"])
    input_state = state(item)
    before = deepcopy(input_state)

    def validator(prompt):
        payload = data(prompt)
        assert payload["claim"] == claim
        assert payload["referenced_evidence"] == {"chunk_id": "event", "text": claim}
        assert payload["evidence"] == [{
            "chunk_id": "event", "document_id": None, "section": item.section,
            "entities": ["Hesper"], "text": claim,
        }]
        return verdict()

    result = compose_answer(input_state, validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis(claim)),
                            validate_semantics=validator)

    assert result.answer == claim
    assert result.status == "complete"
    assert result.citations[0]["claim"] == claim
    assert result.citations[0]["filename"] == item.filename
    assert input_state == before


@pytest.mark.parametrize("support", ["unsupported", "contradicted", "uncertain"])
def test_real_chunk_does_not_allow_failed_semantic_check(support):
    claim = "Hesper has a guarded temperament."
    validator = Mock(return_value=verdict(support, reason="Passage only records an event."))
    with pytest.raises(CitationSupportError) as error:
        validate_citation_claim(claim, "event", {
            "event": evidence("event", text="Hesper was evasive during interrogation.")},
            validate_semantics=validator)
    assert error.value.claim == claim
    assert error.value.chunk_id == "event"
    assert error.value.verdict.support == support
    validator.assert_called_once()


def test_unrelated_real_chunk_cannot_be_rescued_by_other_evidence():
    claim = "Hesper rescued a traveler."
    unrelated = evidence("event", text="The test gate is blue.")
    actual_support = evidence("other", text=claim)

    def validator(prompt):
        payload = data(prompt)
        assert payload["referenced_evidence"]["text"] == unrelated.text
        assert "Do not use a\ndifferent chunk to rescue support" in prompt
        return verdict("unsupported", reason="The cited passage is unrelated.")

    with pytest.raises(CitationSupportError):
        validate_citation_claim(claim, "event", {"event": unrelated, "other": actual_support},
                                validate_semantics=validator)


@pytest.mark.parametrize("attribute,absence,adjacent,inferred", [
    ("temperament", "No canonical temperament is established.",
     "Hesper was evasive during interrogation.", "Hesper has a guarded temperament."),
    ("physical appearance", "No canonical physical features are established.",
     "Hesper worked as a smith.", "Hesper is muscular."),
])
def test_cross_chunk_absence_blocks_inference_even_if_local_support_is_positive(
    attribute, absence, adjacent, inferred
):
    disclaimer = evidence("absence", text=f"Hesper profile. {absence}", entities=["Hesper"],
                          section=attribute)
    event = evidence("event", text=adjacent, entities=["Hesper"])

    def validator(prompt):
        payload = data(prompt)
        assert payload["referenced_evidence"] == {"chunk_id": "event", "text": adjacent}
        assert payload["claim"] == inferred
        assert payload["evidence"][0]["text"] == disclaimer.text
        assert "Inspect ALL evidence" in prompt
        assert "Bind the\nentity, attribute, and scope" in prompt
        return verdict(
            explicit_absence="violated", reason=f"Inferred {attribute} despite explicit absence.",
            absence_findings=[{"chunk_id": "absence", "passage": absence}],
        )

    with pytest.raises(CitationSupportError) as error:
        validate_citation_claim(inferred, "event", {"absence": disclaimer, "event": event},
                                validate_semantics=validator)
    assert error.value.verdict.support == "supported"
    assert error.value.verdict.explicit_absence == "violated"
    assert error.value.verdict.absence_findings[0].passage == absence


@pytest.mark.parametrize("claim,chunk_id", [
    ("The archive does not establish Hesper's canonical temperament.", "absence"),
    ("Hesper was evasive during interrogation.", "event"),
])
def test_absence_and_direct_behavior_are_allowed_without_inferred_traits(claim, chunk_id):
    absence = "No canonical temperament is established."
    validator = Mock(return_value=verdict(
        absence_findings=[{"chunk_id": "absence", "passage": absence}],
    ))
    result = compose_answer(
        state(evidence("absence", text=f"Hesper profile. {absence}"),
              evidence("event", text="Hesper was evasive during interrogation.")),
        validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis(claim, chunk_id)), validate_semantics=validator,
    )
    assert result.answer == claim
    assert result.status == "complete"
    assert result.citations[0]["claim"] == claim


def test_other_entity_is_not_automatically_blocked_by_disclaimer():
    claim = "Orin has red hair."

    def validator(prompt):
        assert "Do not transfer\nan absence to another entity or attribute" in prompt
        assert data(prompt)["referenced_evidence"]["text"] == claim
        return verdict()

    result = compose_answer(
        state(evidence("absence", text="Hesper: No canonical physical features are established."),
              evidence("event", text=claim)),
        validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis(claim)), validate_semantics=validator,
    )
    assert result.answer == claim


def test_uncertain_absence_scope_fails_safely():
    with pytest.raises(CitationSupportError):
        validate_citation_claim("Claim", "event", {"event": evidence("event")},
                                validate_semantics=Mock(return_value=verdict(explicit_absence="uncertain")))


def test_unknown_chunk_rejected_before_any_semantic_calls():
    output = {"answer": "Two claims.", "citation_claims": [
        {"claim": "First", "chunk_id": "event"}, {"claim": "Second", "chunk_id": "invented"},
    ]}
    validator = Mock()
    with pytest.raises(ValueError, match="Unknown synthesis chunk_id: invented"):
        compose_answer(state(evidence("event")), validate_coverage=complete_coverage, synthesize=Mock(return_value=output),
                       validate_semantics=validator)
    validator.assert_not_called()


@pytest.mark.parametrize("output", [
    None, True, "supported", {}, {"support": "supported"},
    {"explicit_absence": "clear"}, verdict("yes"), verdict(explicit_absence=True),
    verdict(reason=42), verdict(answer="Rewritten answer"),
    verdict(absence_findings="none"), verdict(absence_findings=[{"chunk_id": "event"}]),
    SemanticVerdict.model_construct(support="invalid", explicit_absence="clear"),
])
def test_malformed_validator_result_rejected(output):
    with pytest.raises(ValidationError):
        compose_answer(state(evidence("event")), validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis("Claim")),
                       validate_semantics=Mock(return_value=output))


@pytest.mark.parametrize("output", [
    verdict(explicit_absence="violated"),
    verdict(absence_findings=[{"chunk_id": "invented", "passage": "Absent"}]),
    verdict(absence_findings=[{"chunk_id": "event", "passage": "Invented quotation"}]),
])
def test_missing_or_fabricated_absence_findings_fail_safely(output):
    with pytest.raises(ValueError):
        compose_answer(state(evidence("event")), validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis("Claim")),
                       validate_semantics=Mock(return_value=output))


def test_no_validator_wiring_fails_instead_of_skipping_check():
    with pytest.raises(ValueError, match="semantic-validation adapter is required"):
        compose_answer(state(evidence("event")), validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis("Claim")))


def test_validator_failure_propagates_without_answer():
    with pytest.raises(RuntimeError, match="Validator unavailable"):
        compose_answer(state(evidence("event")), validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis("Claim")),
                       validate_semantics=Mock(side_effect=RuntimeError("Validator unavailable")))


def test_one_rejected_claim_is_withheld_without_changing_validated_claim():
    output = {"answer": "Supported fact. Unsupported inference.", "citation_claims": [
        {"claim": "Supported fact.", "chunk_id": "event"},
        {"claim": "Unsupported inference.", "chunk_id": "event"},
    ]}
    before = deepcopy(output)
    validator = Mock(side_effect=[SemanticVerdict(**verdict()), verdict("unsupported")])
    result = compose_answer(state(evidence("event")), validate_coverage=complete_coverage,
                            synthesize=Mock(return_value=output), validate_semantics=validator)
    assert result.status == "partial_validation_limited"
    assert "Supported fact." in result.answer
    assert "Unsupported inference." not in result.answer
    assert [item["claim"] for item in result.citations] == ["Supported fact."]
    assert validator.call_count == 2
    assert output == before


def test_partial_supported_claim_is_also_semantically_checked():
    validator = Mock(return_value=verdict("unsupported"))
    result = compose_answer(state(evidence("event"), unresolved_claims=["Unknown date."]),
                            validate_coverage=complete_coverage, synthesize=Mock(return_value=synthesis("Unsupported claim")),
                            validate_semantics=validator)
    assert result.status == "partial_gap_stated"
    assert result.citations == []
    assert "Unknown date." in result.answer
    assert "Unsupported claim" not in result.answer
    validator.assert_called_once()

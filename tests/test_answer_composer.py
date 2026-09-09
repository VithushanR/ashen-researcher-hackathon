"""Synthetic contract-shaped inputs; no archive fixtures or API calls."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from tests.answer_helpers import complete_coverage
from tests.answer_helpers import supported_semantics
from src.answer.synthesis import CitationClaim, PartialSynthesisResult, SynthesisResult


def evidence(chunk_id="test_a", **overrides):
    fields = dict(
        chunk_id=chunk_id, document_id=None, filename="synthetic.pdf",
        source_type="codex", reliability="T1_authoritative", page=7,
        section="Test section", content_type="text",
        text="Synthetic record states the test gate is blue. More context.",
        entities=[],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def state(*items, **overrides):
    fields = dict(
        question="What color is the test gate?", route="simple", required_claims=[],
        iteration=2, search_history=["test gate"], evidence=list(items),
        discovered_entities=[], unresolved_claims=[], conflicts=[], confidence=87, trace=[],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def response(chunk_id="test_a"):
    return {
        "answer": "The test gate is blue.",
        "citation_claims": [{"claim": "The test gate is blue.", "chunk_id": chunk_id}],
    }


def test_single_evidence_success_and_minimal_prompt():
    item = evidence()
    research = state(item)
    synthesize = Mock(return_value=response())

    result = compose_answer(research, validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=synthesize)

    assert result.model_dump() == {
        "question": research.question, "answer": "The test gate is blue.",
        "status": "complete", "confidence": 87, "iterations_used": 2, "conflicts": [],
        "citations": [{"claim": "The test gate is blue.", "filename": item.filename,
                       "page": 7, "section": "Test section", "source_type": "codex"}],
    }
    synthesize.assert_called_once()
    prompt = synthesize.call_args.args[0]
    payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
    assert payload == {"question": research.question, "required_claims": [],
                       "unresolved_claims": [], "conflicts": [],
                       "evidence": [{"chunk_id": item.chunk_id, "text": item.text}]}
    assert result.citations[0]["claim"] != item.text
    assert research.required_claims == []
    assert not hasattr(research, "final_answer")


def test_multiple_chunks_resolve_by_id_not_evidence_order():
    first = evidence()
    second = evidence("test_b", filename="other.md", page=None, section="Colors",
                      source_type="wiki", text="The test wall is green.")
    output = {"answer": "The test wall is green. The test gate is blue.",
              "citation_claims": [
                  {"claim": "The test wall is green.", "chunk_id": "test_b"},
                  {"claim": "The test gate is blue.", "chunk_id": "test_a"},
              ]}

    result = compose_answer(state(first, second), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=output))

    assert result.citations == [
        {"claim": "The test wall is green.", "filename": "other.md", "page": None,
         "section": "Colors", "source_type": "wiki"},
        {"claim": "The test gate is blue.", "filename": "synthetic.pdf", "page": 7,
         "section": "Test section", "source_type": "codex"},
    ]


@pytest.mark.parametrize("section", ["Test section", None])
def test_section_and_filename_fallback_metadata(section):
    item = evidence(page=None, section=section)
    result = compose_answer(state(item), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=response()))
    assert result.citations[0] == {
        "claim": "The test gate is blue.", "filename": item.filename,
        "page": None, "section": section, "source_type": item.source_type,
    }


def test_unknown_chunk_is_rejected():
    with pytest.raises(ValueError, match="Unknown synthesis chunk_id: invented"):
        compose_answer(state(evidence()), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=response("invented")))


@pytest.mark.parametrize("field", ["filename", "page", "section", "source_type",
                                   "reliability", "document_id"])
def test_generated_metadata_is_rejected(field):
    output = response()
    output["citation_claims"][0][field] = "invented"
    with pytest.raises(ValidationError):
        compose_answer(state(evidence()), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=output))


@pytest.mark.parametrize("output", [
    {}, {"answer": "Uncited assertion", "citation_claims": []},
    {"answer": 42, "citation_claims": [{"claim": "Claim", "chunk_id": "test_a"}]},
    {"answer": "Answer", "citation_claims": [{"claim": "Claim"}]},
])
def test_malformed_synthesis_is_rejected(output):
    with pytest.raises(ValidationError):
        compose_answer(state(evidence()), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=output))


def test_validated_synthesis_result_is_accepted():
    result = compose_answer(
        state(evidence()), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(return_value=SynthesisResult(**response()))
    )
    assert result.answer == response()["answer"]


def assert_synthesis_rejected(output, *, partial=False):
    coverage = Mock()
    semantics = Mock()
    research = state(evidence(), unresolved_claims=["Unknown date."] if partial else [])
    with pytest.raises(ValidationError):
        compose_answer(research, synthesize=Mock(return_value=output),
                       validate_coverage=coverage, validate_semantics=semantics)
    coverage.assert_not_called()
    semantics.assert_not_called()


@pytest.mark.parametrize("as_instance", [False, True], ids=["dictionary", "instance"])
@pytest.mark.parametrize("empty_answer,empty_claims", [(True, True), (True, False), (False, True)])
def test_clean_constraints_apply_to_dicts_and_unchecked_instances(as_instance, empty_answer, empty_claims):
    output = response()
    if empty_answer:
        output["answer"] = ""
    if empty_claims:
        output["citation_claims"] = []
    if as_instance:
        output["citation_claims"] = [CitationClaim(**claim) for claim in output["citation_claims"]]
        output = SynthesisResult.model_construct(**output)
    assert_synthesis_rejected(output)


def test_mutated_synthesis_instance_is_revalidated():
    output = SynthesisResult(**response())
    output.answer = ""
    output.citation_claims.clear()
    assert_synthesis_rejected(output)


def test_partial_subclass_cannot_relax_clean_constraints():
    assert_synthesis_rejected(PartialSynthesisResult(answer="No supported facts.", citation_claims=[]))


def test_clean_compatible_subclass_is_accepted():
    output = PartialSynthesisResult(**response())
    result = compose_answer(state(evidence()), synthesize=Mock(return_value=output),
                            validate_coverage=complete_coverage, validate_semantics=supported_semantics)
    assert result.answer == output.answer
    assert result.status == "complete"
    assert result.citations[0]["claim"] == output.citation_claims[0].claim


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("field", ["claim", "chunk_id"])
@pytest.mark.parametrize("mutated", [False, True], ids=["unchecked", "mutated"])
def test_invalid_nested_claim_instance_is_revalidated(partial, field, mutated):
    fields = response()["citation_claims"][0]
    if mutated:
        claim = CitationClaim(**fields)
        setattr(claim, field, "")
    else:
        fields[field] = ""
        claim = CitationClaim.model_construct(**fields)
    output = response()
    output["citation_claims"] = [claim]
    assert_synthesis_rejected(output, partial=partial)


@pytest.mark.parametrize("as_instance", [False, True], ids=["dictionary", "instance"])
def test_partial_nonempty_answer_with_zero_claims_remains_valid(as_instance):
    output = {"answer": "No supported facts.", "citation_claims": []}
    if as_instance:
        output = PartialSynthesisResult(**output)
    semantics = Mock()
    result = compose_answer(
        state(evidence(), unresolved_claims=["Unknown date."]),
        synthesize=Mock(return_value=output), validate_coverage=complete_coverage,
        validate_semantics=semantics,
    )
    assert result.status == "partial_gap_stated"
    assert result.answer.startswith("No supported facts.")
    assert "Unknown date." in result.answer
    assert result.citations == []
    semantics.assert_not_called()


@pytest.mark.parametrize("as_instance", [False, True], ids=["dictionary", "instance"])
def test_partial_empty_answer_is_rejected(as_instance):
    output = {"answer": "", "citation_claims": []}
    if as_instance:
        output = PartialSynthesisResult.model_construct(**output)
    assert_synthesis_rejected(output, partial=True)


@pytest.mark.parametrize("research", [
    state(),
    state(evidence(), conflicts=[SimpleNamespace(
        attribute=None, claims=[], resolution=None, resolved_value=None)]),
    state(evidence(), evidence()),
])
def test_out_of_scope_or_ambiguous_state_rejected_before_synthesis(research):
    synthesize = Mock()
    with pytest.raises(ValueError):
        compose_answer(research, validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=synthesize)
    synthesize.assert_not_called()


def test_synthesis_failure_propagates_without_complete_answer():
    with pytest.raises(RuntimeError, match="Synthetic provider failure"):
        compose_answer(state(evidence()), validate_semantics=supported_semantics, validate_coverage=complete_coverage, synthesize=Mock(
            side_effect=RuntimeError("Synthetic provider failure")))

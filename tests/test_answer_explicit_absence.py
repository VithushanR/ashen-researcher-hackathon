"""Prompt/composition regressions; semantic rejection has separate runtime tests.

All examples are synthetic. Injected responses demonstrate the intended grounded
output. A supported-verdict double isolates these tests from semantic judgment;
no live LLM is tested here.
"""

import json
from unittest.mock import Mock

import pytest

from src.answer.composer import compose_answer
from tests.answer_helpers import supported_semantics
from tests.test_answer_composer import evidence, state


@pytest.mark.parametrize(
    "attribute,absence,adjacent_fact,answer,forbidden_inference,rule",
    [
        (
            "physical appearance", "No canonical physical features are established.",
            "Test Figure worked as a smith.",
            "Test Figure's canonical physical appearance is not established.",
            "Test Figure is muscular.",
            "not a description inferred\nfrom an occupation or activity",
        ),
        (
            "temperament", "No canonical temperament is established.",
            "Test Figure rescued a traveler.",
            "Test Figure's canonical temperament is not established.",
            "Test Figure is brave.",
            "not personality traits\ninferred from behavior or an event",
        ),
    ],
)
def test_explicit_absence_is_complete_and_adjacent_facts_do_not_replace_it(
    attribute, absence, adjacent_fact, answer, forbidden_inference, rule
):
    absence_item = evidence("absence", filename="synthetic_profile.md", page=None,
                            section=attribute, source_type="wiki",
                            text=f"Test Figure profile. {absence}")
    adjacent_item = evidence("adjacent", text=adjacent_fact)
    research = state(absence_item, adjacent_item,
                     question=f"What is Test Figure's {attribute}?")

    def synthesize(prompt):
        # Exercise the real prompt builder while replacing only the provider call.
        assert rule in prompt
        assert "A real chunk_id alone does not make either\ninference supported" in prompt
        assert "Do not reconstruct that attribute from adjacent facts" in prompt
        payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
        assert payload["evidence"] == [
            {"chunk_id": "absence", "text": absence_item.text},
            {"chunk_id": "adjacent", "text": adjacent_fact},
        ]
        return {"answer": answer, "citation_claims": [{"claim": answer, "chunk_id": "absence"}]}

    result = compose_answer(research, validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.answer == answer
    assert forbidden_inference not in result.answer
    assert result.status == "complete"
    assert result.confidence == research.confidence
    assert result.iterations_used == research.iteration
    assert result.citations == [{
        "claim": answer, "filename": "synthetic_profile.md", "page": None,
        "section": attribute, "source_type": "wiki",
    }]


def test_direct_event_in_absence_evidence_remains_citable():
    event = "Test Figure rescued a traveler."
    absence = "No canonical temperament is established."
    item = evidence("combined", text=f"Test Figure profile. {absence} {event}")
    synthesize = Mock(return_value={
        "answer": event, "citation_claims": [{"claim": event, "chunk_id": "combined"}],
    })

    result = compose_answer(state(item, question="What did Test Figure do?"), validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.answer == event
    assert result.status == "complete"
    assert result.citations[0]["claim"] == event
    assert result.citations[0]["filename"] == item.filename
    prompt = synthesize.call_args.args[0]
    assert "Direct factual events in the same\nevidence may still be reported" in prompt
    assert "Do not transfer an\nabsence to a different entity or attribute" in prompt


def test_absence_and_direct_event_can_both_be_supported_claims():
    absence_claim = "Test Figure's canonical temperament is not established."
    event = "Test Figure rescued a traveler."
    items = [evidence("absence", text="Test Figure: No canonical temperament is established."),
             evidence("event", filename="synthetic_event.pdf", text=event)]
    synthesize = Mock(return_value={
        "answer": f"{absence_claim} {event}",
        "citation_claims": [{"claim": absence_claim, "chunk_id": "absence"},
                            {"claim": event, "chunk_id": "event"}],
    })

    result = compose_answer(state(*items), validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.status == "complete"
    assert [citation["claim"] for citation in result.citations] == [absence_claim, event]
    assert [citation["filename"] for citation in result.citations] == [item.filename for item in items]


def test_explicit_absence_does_not_override_other_upstream_gaps():
    claim = "Test Figure's canonical temperament is not established."
    gap = "The date of Test Figure's rescue is not established by retrieved evidence."
    synthesize = Mock(return_value={
        "answer": claim, "citation_claims": [{"claim": claim, "chunk_id": "absence"}],
    })
    research = state(
        evidence("absence", text="Test Figure: No canonical temperament is established."),
        unresolved_claims=[gap],
    )

    result = compose_answer(research, validate_semantics=supported_semantics, synthesize=synthesize)

    assert result.status == "partial_gap_stated"
    assert gap in result.answer
    assert [citation["claim"] for citation in result.citations] == [claim]
    prompt = synthesize.call_args.args[0]
    assert "An explicit absence can fully answer the question" in prompt
    assert "retain those gaps" in prompt


@pytest.mark.parametrize("absence", [
    "Test Figure's physical appearance is unrecorded.",
    "No description of Test Figure's appearance is canonical.",
])
def test_absence_qualifiers_are_preserved(absence):
    synthesize = Mock(return_value={
        "answer": absence, "citation_claims": [{"claim": absence, "chunk_id": "absence"}],
    })
    result = compose_answer(state(evidence("absence", text=absence)), validate_semantics=supported_semantics, synthesize=synthesize)
    assert result.answer == absence
    assert result.status == "complete"
    prompt = synthesize.call_args.args[0]
    assert "'unrecorded' does not mean 'never happened'" in prompt
    assert "Do not treat silence\nor retrieval failure as an explicit absence statement" in prompt

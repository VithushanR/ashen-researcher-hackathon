"""Synthetic visual descriptions; no image loading, vision calls, or archive facts.

Injected judgments test the existing semantic gate and actual prompt payloads,
not the accuracy of a real vision model or semantic adapter.
"""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.answer.composer import compose_answer
from src.answer.semantic_validation import CitationSupportError
from tests.answer_helpers import complete_coverage
from tests.test_answer_composer import evidence, state


def synthesis(claim, chunk_id):
    return {"answer": claim, "citation_claims": [{"claim": claim, "chunk_id": chunk_id}]}


def data(prompt):
    return json.loads(prompt.split("INPUT DATA:\n", 1)[1])


def vision(text, **overrides):
    # Deliberately no naming link to the original chunk: provenance is metadata.
    fields = dict(chunk_id="opaque-result-7", document_id="synthetic_profile",
                  filename="synthetic_plate.png", source_type="image_derived",
                  reliability="T2_curated", page=None, section=None,
                  content_type="vision_description", text=text, entities=["Test relic"])
    fields.update(overrides)
    return evidence(**fields)


@pytest.mark.parametrize("page,section", [(None, None), (9, "Plate 2"), (None, "Plate caption")])
def test_vision_complete_uses_original_image_metadata_and_preserves_input(page, section):
    claim = "The test plate depicts three interlocking bands."
    original = evidence("original-image", filename="synthetic_plate.png", source_type="wiki",
                        reliability="T2_curated", page=None, section=None, content_type="image",
                        text="Test relic", entities=["Test relic"])
    derived = vision(claim, page=page, section=section)
    research = state(original, derived, confidence=63)
    before = deepcopy(research)
    calls = []

    def synthesize(prompt):
        calls.append("synthesis")
        assert data(prompt)["evidence"] == [
            {"chunk_id": original.chunk_id, "text": original.text},
            {"chunk_id": derived.chunk_id, "text": derived.text},
        ]
        return synthesis(claim, derived.chunk_id)

    def coverage(prompt):
        calls.append("coverage")
        assert data(prompt)["citation_claims"][0]["chunk_id"] == derived.chunk_id
        return complete_coverage(prompt)

    def semantic(prompt):
        calls.append("semantic")
        assert data(prompt)["referenced_evidence"] == {"chunk_id": derived.chunk_id, "text": claim}
        return {"support": "supported", "explicit_absence": "clear"}

    result = compose_answer(research, synthesize=synthesize, validate_coverage=coverage,
                            validate_semantics=semantic)

    assert result.status == "complete"
    assert result.confidence == 63
    assert result.iterations_used == research.iteration
    assert result.citations == [{
        "claim": claim, "filename": "synthetic_plate.png", "page": page,
        "section": section, "source_type": "image_derived",
    }]
    assert calls == ["synthesis", "coverage", "semantic"]
    assert research == before
    assert original.content_type == "image"
    assert original.text == "Test relic"


def test_provenance_is_not_decoded_from_vision_chunk_name():
    claim = "The test plate depicts three bands."
    item = vision(claim, chunk_id="misleading_codex_p999_model_name_vision_3",
                  filename="actual_original.png", page=4, section="Actual section")
    result = compose_answer(
        state(item), synthesize=Mock(return_value=synthesis(claim, item.chunk_id)),
        validate_coverage=complete_coverage,
        validate_semantics=Mock(return_value={"support": "supported", "explicit_absence": "clear"}),
    )
    assert result.citations[0] == {
        "claim": claim, "filename": "actual_original.png", "page": 4,
        "section": "Actual section", "source_type": "image_derived",
    }


def test_thin_image_caption_does_not_support_visual_fact_even_with_vision_chunk_present():
    claim = "The test plate depicts three bands."
    raw = evidence("raw", content_type="image", filename="synthetic_plate.png",
                   source_type="wiki", text="Test relic")
    derived = vision(claim)

    def semantic(prompt):
        payload = data(prompt)
        assert payload["referenced_evidence"] == {"chunk_id": "raw", "text": "Test relic"}
        assert any(item["text"] == claim for item in payload["evidence"])
        assert "Do not use a\ndifferent chunk to rescue support" in prompt
        return {"support": "unsupported", "explicit_absence": "clear",
                "reason": "The cited raw caption names the relic but states no band count."}

    with pytest.raises(CitationSupportError) as error:
        compose_answer(state(raw, derived), synthesize=Mock(return_value=synthesis(claim, "raw")),
                       validate_coverage=complete_coverage, validate_semantics=semantic)
    assert error.value.chunk_id == "raw"
    assert error.value.verdict.support == "unsupported"


def test_raw_image_caption_can_support_its_literal_text():
    claim = "The test plate depicts three bands."
    raw = evidence("raw", content_type="image", filename="synthetic_plate.png",
                   source_type="wiki", page=None, section=None, text=claim)

    def semantic(prompt):
        assert data(prompt)["referenced_evidence"]["text"] == claim
        return {"support": "supported", "explicit_absence": "clear"}

    result = compose_answer(state(raw), synthesize=Mock(return_value=synthesis(claim, "raw")),
                            validate_coverage=complete_coverage, validate_semantics=semantic)
    assert result.status == "complete"
    assert result.citations[0]["source_type"] == "wiki"
    assert result.citations[0]["filename"] == "synthetic_plate.png"


@pytest.mark.parametrize("description", [
    "The symbol appears to be a raven, but the image is unclear.",
    "The image appears to show a raven.",
])
def test_strengthened_uncertain_vision_claim_is_rejected(description):
    claim = "The banner definitely shows a raven."
    item = vision(description)

    def semantic(prompt):
        payload = data(prompt)
        assert payload["referenced_evidence"]["text"] == description
        assert payload["claim"] == claim
        assert "retaining\nits uncertainty" in prompt
        return {"support": "unsupported", "explicit_absence": "clear",
                "reason": "The claim replaces uncertain visual interpretation with certainty."}

    with pytest.raises(CitationSupportError):
        compose_answer(state(item), synthesize=Mock(return_value=synthesis(claim, item.chunk_id)),
                       validate_coverage=complete_coverage, validate_semantics=semantic)


def test_faithful_vision_hedging_is_kept_in_answer_and_citation_claim():
    claim = "The image appears to show a raven."
    item = vision(claim)

    def semantic(prompt):
        assert data(prompt)["claim"] == data(prompt)["referenced_evidence"]["text"] == claim
        return {"support": "supported", "explicit_absence": "clear"}

    result = compose_answer(state(item), synthesize=Mock(return_value=synthesis(claim, item.chunk_id)),
                            validate_coverage=complete_coverage, validate_semantics=semantic)
    assert result.answer == claim
    assert result.citations[0]["claim"] == claim
    assert result.status == "complete"


@pytest.mark.parametrize("content_type,text", [
    ("text", "The synthetic gate is blue."),
    ("table", "Gate | Color\nSynthetic gate | blue"),
])
def test_text_and_table_evidence_keep_the_same_generic_path(content_type, text):
    claim = "The synthetic gate is blue."
    item = evidence("ordinary", content_type=content_type, text=text)

    def semantic(prompt):
        assert data(prompt)["referenced_evidence"] == {"chunk_id": "ordinary", "text": text}
        return {"support": "supported", "explicit_absence": "clear"}

    result = compose_answer(state(item), synthesize=Mock(return_value=synthesis(claim, "ordinary")),
                            validate_coverage=complete_coverage, validate_semantics=semantic)
    assert result.status == "complete"
    assert result.citations == [{"claim": claim, "filename": item.filename, "page": item.page,
                                 "section": item.section, "source_type": item.source_type}]

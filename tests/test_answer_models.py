"""Contract tests using synthetic values, not archive or teammate fixtures."""

import json

import pytest
from pydantic import ValidationError

from src.answer.models import ComposedAnswer


@pytest.fixture
def answer_payload():
    return {
        "question": "Synthetic test question",
        "answer": "Synthetic test answer",
        "status": "complete",
        "confidence": 80,
        "citations": [],
        "conflicts": [],
        "iterations_used": 2,
    }


@pytest.mark.parametrize(
    "status", ["complete", "complete_with_conflict", "partial_gap_stated", "partial_validation_limited"]
)
def test_supported_statuses_serialize(answer_payload, status):
    answer_payload["status"] = status

    result = ComposedAnswer(**answer_payload)

    assert json.loads(result.model_dump_json()) == answer_payload


@pytest.mark.parametrize("status", ["", "unknown", "COMPLETE", "partial"])
def test_unsupported_statuses_are_rejected(answer_payload, status):
    answer_payload["status"] = status

    with pytest.raises(ValidationError) as exc_info:
        ComposedAnswer(**answer_payload)

    assert exc_info.value.errors()[0]["loc"] == ("status",)


@pytest.mark.parametrize(
    "field",
    ["question", "answer", "status", "confidence", "citations", "conflicts", "iterations_used"],
)
def test_all_output_fields_are_required(answer_payload, field):
    del answer_payload[field]

    with pytest.raises(ValidationError) as exc_info:
        ComposedAnswer(**answer_payload)

    assert any(
        error["loc"] == (field,) and error["type"] == "missing"
        for error in exc_info.value.errors()
    )


@pytest.mark.parametrize(
    "resolution,resolved_value",
    [(None, None), ("Upstream test resolution", "Value B")],
)
def test_conflict_and_citation_details_survive_serialization(
    answer_payload, resolution, resolved_value
):
    answer_payload["citations"] = [{
        "claim": "Synthetic claim", "filename": "test.md", "page": None,
        "source_type": "wiki", "test_details": [None, "value"],
    }]
    answer_payload["conflicts"] = [
        {
            "attribute": "Synthetic attribute",
            "claims": [
                {"claim": "Value A", "source": "Test source A", "chunk_id": "test_a"},
                {"claim": "Value B", "source": "Test source B", "chunk_id": "test_b"},
                {"claim": "Value C", "source": "Test source C", "chunk_id": None},
            ],
            "resolution": resolution,
            "resolved_value": resolved_value,
        }
    ]
    # Status is supplied explicitly; this model does not infer it from conflicts.
    answer_payload["status"] = "partial_gap_stated"

    result = ComposedAnswer(**answer_payload)

    assert result.model_dump() == answer_payload
    assert json.loads(result.model_dump_json()) == answer_payload


@pytest.fixture
def citation():
    return {
        "claim": "  Synthetic claim.  ",
        "filename": "test.pdf",
        "page": 12,
        "source_type": "codex",
        "extra": {"values": [None, "unchanged"]},
    }


@pytest.mark.parametrize(
    "location",
    [
        {"page": 12, "section": "Test section"},
        {"page": 12},
        {"page": None, "section": "Test section"},
        {"page": None, "section": None},
        {"page": None},
    ],
    ids=["page-and-section", "page-only", "section-fallback", "null-section", "filename-only"],
)
def test_citation_locations_preserve_exact_values(answer_payload, citation, location):
    citation.update(location)
    answer_payload["citations"] = [citation]

    result = ComposedAnswer(**answer_payload)

    assert result.model_dump() == answer_payload
    assert json.loads(result.model_dump_json()) == answer_payload


@pytest.mark.parametrize("field", ["claim", "filename", "source_type", "page"])
def test_missing_citation_fields_are_rejected(answer_payload, citation, field):
    del citation[field]
    answer_payload["citations"] = [citation]

    with pytest.raises(ValidationError, match=rf"citations\[0\].{field} is required"):
        ComposedAnswer(**answer_payload)


@pytest.mark.parametrize("field", ["claim", "filename", "source_type"])
@pytest.mark.parametrize("value", [None, 12, True, [], {}])
def test_citation_string_fields_reject_wrong_types(answer_payload, citation, field, value):
    citation[field] = value
    answer_payload["citations"] = [citation]

    with pytest.raises(ValidationError, match=rf"citations\[0\].{field} must be a string"):
        ComposedAnswer(**answer_payload)


@pytest.mark.parametrize("value", ["12", 12.0, True, False, [], {}])
def test_invalid_page_types_are_rejected(answer_payload, citation, value):
    citation["page"] = value
    answer_payload["citations"] = [citation]

    with pytest.raises(ValidationError, match=r"citations\[0\].page must be an int or None"):
        ComposedAnswer(**answer_payload)


@pytest.mark.parametrize("value", [12, 1.5, True, [], {}])
def test_invalid_section_types_are_rejected(answer_payload, citation, value):
    citation["section"] = value
    answer_payload["citations"] = [citation]

    with pytest.raises(ValidationError, match=r"citations\[0\].section must be a string or None"):
        ComposedAnswer(**answer_payload)


def test_multiple_citations_preserve_order_and_contents(answer_payload, citation):
    answer_payload["citations"] = [
        citation,
        {"claim": "Second claim", "filename": "test.md", "page": None,
         "section": " Section ", "source_type": "wiki"},
    ]

    result = ComposedAnswer(**answer_payload)

    assert result.model_dump() == answer_payload
    assert json.loads(result.model_dump_json()) == answer_payload


def test_invalid_second_citation_identifies_index(answer_payload, citation):
    answer_payload["citations"] = [citation, {**citation, "page": "12"}]

    with pytest.raises(ValidationError, match=r"citations\[1\].page"):
        ComposedAnswer(**answer_payload)

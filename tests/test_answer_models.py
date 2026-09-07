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
    "status", ["complete", "complete_with_conflict", "partial_gap_stated"]
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
    # These opaque citation contents test preservation, not a citation schema.
    answer_payload["citations"] = [{"test_marker": "citation", "test_details": [None, "value"]}]
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

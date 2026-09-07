"""Synthetic conflicts test presentation, not Person B's resolution policy."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.answer.composer import compose_answer
from tests.answer_helpers import complete_coverage
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

    result = compose_answer(input_state, validate_coverage=complete_coverage, synthesize=synthesize)

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

    result = compose_answer(research(supplied), validate_coverage=complete_coverage, synthesize=synthesize)

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
    result = compose_answer(research(supplied), validate_coverage=complete_coverage, synthesize=Mock())
    assert len(result.conflicts[0]["claims"]) == 3
    assert len(result.citations) == 3
    for claim in supplied.claims:
        assert claim.claim in result.answer
        assert claim.source in result.answer
    assert result.conflicts[0]["claims"] == [vars(claim) for claim in supplied.claims]


def test_mixed_conflicts_remain_partial_and_preserve_both_decisions():
    result = compose_answer(
        research(conflict(), conflict(resolved_value=None)), validate_coverage=complete_coverage, synthesize=Mock()
    )
    assert result.status == "partial_gap_stated"
    assert "The research result prefers: green." in result.answer
    assert "evidence remains contradictory" in result.answer
    assert [item["resolved_value"] for item in result.conflicts] == ["green", None]


def test_resolved_value_without_explanation_does_not_invent_reason():
    result = compose_answer(research(conflict(resolution=None)), validate_coverage=complete_coverage, synthesize=Mock())
    assert result.status == "complete_with_conflict"
    assert result.conflicts[0]["resolution"] is None
    assert "Supplied resolution:" not in result.answer


def test_missing_chunk_id_keeps_claim_without_fabricating_citation():
    supplied = conflict()
    supplied.claims[1].chunk_id = None
    result = compose_answer(research(supplied), validate_coverage=complete_coverage, synthesize=Mock())
    assert len(result.citations) == 1
    assert result.conflicts[0]["claims"][1]["chunk_id"] is None
    assert supplied.claims[1].claim in result.answer
    assert "No evidence chunk reference was supplied" in result.answer


def test_unknown_conflict_chunk_id_is_rejected():
    supplied = conflict()
    supplied.claims[0].chunk_id = "invented"
    with pytest.raises(ValueError, match="Unknown conflict chunk_id: invented"):
        compose_answer(research(supplied), validate_coverage=complete_coverage, synthesize=Mock())


def test_unresolved_conflict_preserves_reported_gaps():
    result = compose_answer(
        research(conflict(resolved_value=None), unresolved_claims=["Gate color is disputed."]),
        validate_coverage=complete_coverage, synthesize=Mock(),
    )
    assert result.status == "partial_gap_stated"
    assert "Gate color is disputed." in result.answer


def test_upstream_result_is_not_overridden_by_source_tier():
    input_state = research(conflict(resolved_value="green"))
    input_state.evidence[0].reliability = "T1_authoritative"
    input_state.evidence[1].reliability = "T4_unverified"
    result = compose_answer(input_state, validate_coverage=complete_coverage, synthesize=Mock())
    assert "The research result prefers: green." in result.answer

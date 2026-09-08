"""Synthetic B -> C integration: real loop/models/composer, no APIs or archive truth."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from agent.loop import research
from agent.state import ClaimSource, Conflict, Evidence, ResearchState, TraceStep
from src.answer.composer import compose_answer
from tests.answer_helpers import complete_coverage, supported_semantics
from tests.test_agent_loop import _no_conflicts_response, _sufficiency_response


def chunk(cid, text, **changes):
    fields = dict(chunk_id=cid, document_id="synthetic-document", filename=f"{cid}.md",
                  source_type="wiki", reliability="T2_curated", page=None,
                  section="Synthetic section", content_type="text", text=text,
                  entities=["Synthetic gate"])
    fields.update(changes)
    return fields


@pytest.fixture(autouse=True)
def block_unexpected_external_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Unexpected external boundary call")
    monkeypatch.setattr("agent.sufficiency.call_llm", unexpected)
    monkeypatch.setattr("agent.conflict.call_llm", unexpected)
    monkeypatch.setattr("agent.loop.describe_image", unexpected)


def configure_b(monkeypatch, verdicts, conflicts=None):
    checker = Mock(side_effect=verdicts)
    detector = Mock(return_value=json.dumps(conflicts) if conflicts else _no_conflicts_response())
    monkeypatch.setattr("agent.sufficiency.call_llm", checker)
    monkeypatch.setattr("agent.conflict.call_llm", detector)
    return checker, detector


def handoff(state, fact, cid):
    # Assert authoritative types; pass this exact object to C, never a dict/JSON copy.
    assert type(state) is ResearchState
    assert all(type(item) is Evidence for item in state.evidence)
    assert all(type(step) is TraceStep for step in state.trace)
    assert all(type(c) is Conflict and all(type(x) is ClaimSource for x in c.claims)
               for c in state.conflicts)
    before = deepcopy(state)
    evidence_by_id = {item.chunk_id: item for item in state.evidence}
    checked = []
    synthesize = Mock(return_value={"answer": fact,
        "citation_claims": [{"claim": fact, "chunk_id": cid}]})
    coverage = Mock(side_effect=complete_coverage)

    def semantics(prompt):
        payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
        ref = payload["referenced_evidence"]
        assert ref["text"] == evidence_by_id[ref["chunk_id"]].text
        # These deliberately literal synthetic assertions have direct passage support.
        assert payload["claim"] in ref["text"]
        checked.append((payload["claim"], ref["chunk_id"], "Conflict attribution context:" in prompt))
        return supported_semantics(prompt)

    result = compose_answer(state, synthesize=synthesize,
                            validate_coverage=coverage, validate_semantics=semantics)
    assert state == before
    assert result.question == state.question
    assert result.confidence == state.confidence
    assert result.iterations_used == state.iteration
    synthesize.assert_called_once()
    coverage.assert_called_once()
    assert json.loads(coverage.call_args.args[0].split("INPUT DATA:\n", 1)[1])["answer"] == result.answer
    assert checked[0] == (fact, cid, False)
    item = evidence_by_id[cid]
    assert result.citations[0] == dict(claim=fact, filename=item.filename, page=item.page,
                                     section=item.section, source_type=item.source_type)
    return result, checked


def test_clean_research_state_handoff(monkeypatch):
    fact = "The synthetic gate is blue."
    configure_b(monkeypatch, [_sufficiency_response("yes", "yes", None, "sufficient")])
    search = Mock(return_value=[chunk("gate", fact, filename="synthetic.pdf", page=7)])
    state = research("What color is the synthetic gate?", search_fn=search)
    assert state.trace[-1].verdict == "sufficient"
    assert state.required_claims == [state.question]
    result, checked = handoff(state, fact, "gate")
    assert result.status == "complete"
    assert result.answer == fact
    assert len(result.citations) == len(checked) == 1


@pytest.mark.parametrize("resolved", [True, False], ids=["resolved", "unresolved-tie"])
def test_real_conflict_policy_and_independent_fact_handoff(monkeypatch, resolved):
    blue, green, owner = "The gate is blue.", "The gate is green.", "Orin owns the gate."
    chunks = [chunk("blue", blue + " " + owner),
              chunk("green", green, reliability="T1_authoritative" if resolved else "T2_curated")]
    detection = {"conflicts_found": True, "conflicts": [{"attribute": "gate color", "claims": [
        {"claim": blue, "chunk_id": "blue"}, {"claim": green, "chunk_id": "green"}]}]}
    verdicts = [_sufficiency_response("yes", "conflict", "Gate color remains disputed.", "conflict_detected")]
    if resolved:
        verdicts.append(_sufficiency_response("yes", "yes", None, "sufficient"))
    configure_b(monkeypatch, verdicts, detection)
    state = research("What color is the gate and who owns it?", search_fn=Mock(return_value=chunks),
                     max_iter=2 if resolved else 1)
    supplied = state.conflicts[0]
    assert supplied.resolved_value == (green if resolved else None)
    decision = deepcopy(supplied)
    result, checked = handoff(state, owner, "blue")
    assert owner in result.answer
    assert result.status == ("complete_with_conflict" if resolved else "partial_gap_stated")
    assert result.conflicts == [dict(attribute=decision.attribute,
        claims=[dict(claim=c.claim, source=c.source, chunk_id=c.chunk_id) for c in decision.claims],
        resolution=decision.resolution, resolved_value=decision.resolved_value)]
    assert checked == [(owner, "blue", False), (blue, "blue", True), (green, "green", True)]
    for citation, claim in zip(result.citations[1:], decision.claims):
        item = next(e for e in state.evidence if e.chunk_id == claim.chunk_id)
        assert citation == dict(claim=f'{claim.source} reports: "{claim.claim}"',
            filename=item.filename, page=item.page, section=item.section, source_type=item.source_type)
    if resolved:
        assert f"The research result prefers: {green}" in result.answer
    else:
        assert "No preferred result has been established" in result.answer
        assert "The research result prefers:" not in result.answer
        assert all(gap in result.answer for gap in state.unresolved_claims)


def test_capped_research_gap_handoff(monkeypatch):
    fact, gap = "Orin owns the gate.", "The construction date is unknown."
    configure_b(monkeypatch, [_sufficiency_response("partial", "yes", gap, "insufficient")])
    state = research("Who owns the gate and when was it built?",
                     search_fn=Mock(return_value=[chunk("owner", fact)]), max_iter=1)
    assert state.iteration == 1
    assert state.unresolved_claims == [gap]
    result, checked = handoff(state, fact, "owner")
    assert result.status == "partial_gap_stated"
    assert fact in result.answer and result.answer.count(gap) == 1
    assert len(result.citations) == len(checked) == 1
    assert result.citations[0]["claim"] != gap


def test_real_vision_fallback_evidence_handoff(monkeypatch):
    fact = "The synthetic banner depicts a dog."
    configure_b(monkeypatch, [
        _sufficiency_response("no", "yes", "Identify the banner emblem.", "insufficient"),
        _sufficiency_response("yes", "yes", None, "sufficient"),
    ])
    vision = Mock(return_value=fact)
    monkeypatch.setattr("agent.loop.describe_image", vision)
    raw = chunk("image", "An unnamed banner.", filename="synthetic_banner.png",
                content_type="image", section="Plate", page=3)
    question = "What is on the synthetic banner?"
    state = research(question, search_fn=Mock(return_value=[raw]))
    vision.assert_called_once_with(raw["filename"], question)
    original = next(e for e in state.evidence if e.content_type == "image")
    derived = next(e for e in state.evidence if e.content_type == "vision_description")
    assert derived.source_type == "image_derived"
    assert derived.text == fact
    assert derived.chunk_id != original.chunk_id
    for field in ("document_id", "filename", "page", "section", "entities", "reliability"):
        assert getattr(derived, field) == getattr(original, field)
    assert state.trace[-1].verdict == "sufficient"
    result, checked = handoff(state, fact, derived.chunk_id)
    assert result.status == "complete"
    assert result.answer == fact
    assert result.citations[0]["filename"] == "synthetic_banner.png"
    assert result.citations[0]["source_type"] == "image_derived"
    assert len(checked) == 1


@pytest.fixture(autouse=True)
def mock_requirement_model(monkeypatch):
    def respond(prompt):
        question = json.loads(prompt.split("QUESTION DATA:\n", 1)[1])
        return json.dumps({"required_claims": [question]})
    monkeypatch.setattr("agent.planner.call_llm", respond)

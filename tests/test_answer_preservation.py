"""Synthetic per-claim outcomes; no archive truth, model calls or retrieval."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from tests.test_answer_composer import evidence, state
from tests.test_answer_conflicts import conflict, research


FACTS = ["The test gate is blue.", "Orin owns the test gate.", "The test gate was built in 1400."]
NEEDS = ["Gate color?", "Gate owner?", "Construction date?"]


def data(prompt):
    return json.loads(prompt.split("INPUT DATA:\n", 1)[1])


def prepared(count=3, **changes):
    fields = dict(question="Describe the synthetic gate.", required_claims=NEEDS[:count])
    fields.update(changes)
    return state(*[evidence(f"c{i}", text=FACTS[i]) for i in range(count)], **fields)


def output(count=3):
    return {"answer": " ".join(FACTS[:count]), "citation_claims": [
        {"claim": fact, "chunk_id": f"c{i}"} for i, fact in enumerate(FACTS[:count])
    ]}


def mapped_coverage(prompt):
    """Map synthetic needs to literal claims or trusted Python limitation records."""
    payload = data(prompt)
    context = payload["research_context"]
    limited = context.get("validation_limitations", [])
    assertions = []
    for i, requirement in enumerate(payload["required_claims"] or [payload["question"]]):
        refs = [j for j, item in enumerate(limited)
                if not payload["required_claims"] or i in item["requirement_indices"]]
        if refs:
            message = limited[refs[0]]["message"]
            assert message in payload["answer"]
            assertions.append(dict(requirement=requirement, status="validation_limited",
                answer_excerpt=message, limitation_indices=refs))
        elif requirement == "Unestablished maker?":
            gap = context["unresolved_claims"][0]
            assert gap in payload["answer"]
            assertions.append(dict(requirement=requirement, status="gap", answer_excerpt=gap,
                                   gap_indices=[0]))
        elif requirement == "Disputed gate color?":
            assert context["conflicts"]
            assert "Source conflict" in payload["answer"]
            assertions.append(dict(requirement=requirement, status="conflict",
                                   answer_excerpt=payload["answer"], conflict_indices=[0]))
        else:
            fact = FACTS[NEEDS.index(requirement)] if payload["required_claims"] else FACTS[0]
            indices = [j for j, item in enumerate(payload["citation_claims"]) if item["claim"] == fact]
            assert indices
            assert fact in payload["ordinary_answer"]
            assertions.append(dict(requirement=requirement, status="answered", answer_excerpt=fact,
                                   citation_claim_indices=indices))
    return {"coverage": "complete", "presentation": assertions}


@pytest.mark.parametrize("count,failed", [(1, []), (3, []), (3, [0]), (3, [1]), (3, [0, 2]),
                                         (3, [0, 1, 2]), (1, [0])])
def test_preserve_all_successes_and_withhold_all_failed_pairs(count, failed, monkeypatch):
    research_state = prepared(count)
    before = deepcopy(research_state)
    draft = output(count)
    original = deepcopy(draft)
    synthesis = Mock(return_value=draft)
    coverage = Mock(side_effect=mapped_coverage)
    checked = []
    for target in ("research", "check_sufficiency", "derive_required_claims", "resolve_conflicts"):
        monkeypatch.setattr(f"agent.loop.{target}", Mock(side_effect=AssertionError("B must not run")))

    def semantic(prompt):
        payload = data(prompt)
        index = FACTS.index(payload["claim"])
        checked.append(index)
        assert payload["referenced_evidence"] == {"chunk_id": f"c{index}", "text": FACTS[index]}
        return {"support": "unsupported" if index in failed else "supported", "explicit_absence": "clear"}

    result = compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage,
                            validate_semantics=semantic)
    assert checked == list(range(count))  # Does not stop at the first failure or recheck successes.
    synthesis.assert_called_once()
    assert coverage.call_count == (2 if failed else 1)
    assert result.status == ("partial_validation_limited" if failed else "complete")
    expected = [i for i in range(count) if i not in failed]
    assert [c["claim"] for c in result.citations] == [FACTS[i] for i in expected]
    for index in failed:
        assert FACTS[index] not in result.answer
    for citation, index in zip(result.citations, expected):
        item = research_state.evidence[index]
        assert FACTS[index] in result.answer
        assert citation == dict(claim=FACTS[index], filename=item.filename, page=item.page,
                                section=item.section, source_type=item.source_type)
    if failed:
        assert "Validation limitations:" in result.answer
        assert "Unresolved information" not in result.answer
        assert data(coverage.call_args.args[0])["answer"] == result.answer
        assert all(word not in result.answer for word in ("CitationSupportError", "chunk_id", "Traceback"))
    else:
        assert result.answer == draft["answer"]
    assert result.confidence == research_state.confidence
    assert result.iterations_used == research_state.iteration
    assert research_state == before and draft == original


@pytest.mark.parametrize("with_gap,with_conflict,resolved", [
    (True, False, False), (False, True, True), (False, True, False), (True, True, False),
])
def test_b_context_remains_separate_from_c_limitations(with_gap, with_conflict, resolved):
    needs = NEEDS[1:]
    supplied = conflict(resolved_value="green" if resolved else None)
    if with_conflict:
        research_state = research(supplied)
        needs = needs + ["Disputed gate color?"]
    else:
        research_state = state()
    if with_gap:
        research_state.unresolved_claims = ["The maker is not established by research."]
        needs = needs + ["Unestablished maker?"]
    research_state.required_claims = needs
    research_state.evidence.extend([evidence("owner", text=FACTS[1]), evidence("date", text=FACTS[2])])
    before = deepcopy(research_state)
    synthesis = Mock(return_value={"answer": f"{FACTS[1]} {FACTS[2]}", "citation_claims": [
        {"claim": FACTS[1], "chunk_id": "owner"}, {"claim": FACTS[2], "chunk_id": "date"}]})
    modes = []

    def semantic(prompt):
        payload = data(prompt)
        modes.append("conflict" if "Conflict attribution context:" in prompt else "ordinary")
        return {
            "support": "unsupported" if payload["claim"] == FACTS[2] else "supported",
            "explicit_absence": "clear"}

    result = compose_answer(research_state, synthesize=synthesis, validate_coverage=mapped_coverage,
                            validate_semantics=semantic)
    assert FACTS[1] in result.answer and FACTS[2] not in result.answer
    assert "Validation limitations:" in result.answer
    assert ("Unresolved information" in result.answer) == with_gap
    assert result.status == ("partial_gap_stated" if with_gap or (with_conflict and not resolved)
                             else "partial_validation_limited")
    if with_conflict:
        assert result.conflicts[0]["resolved_value"] == supplied.resolved_value
        assert result.conflicts[0]["resolution"] == supplied.resolution
        assert len(result.citations) == 3
    assert modes == ["ordinary", "ordinary"] + (["conflict", "conflict"] if with_conflict else [])
    assert research_state == before
    synthesis.assert_called_once()


@pytest.mark.parametrize("empty_checklist", [False, True])
def test_unmapped_failure_does_not_invent_a_requirement(empty_checklist):
    research_state = prepared(2, required_claims=[] if empty_checklist else [NEEDS[0]])
    coverage = Mock(side_effect=mapped_coverage)
    result = compose_answer(research_state, synthesize=lambda p: output(2), validate_coverage=coverage,
        validate_semantics=lambda p: {"support": "unsupported" if data(p)["claim"] == FACTS[1] else "supported",
                                     "explicit_absence": "clear"})
    limitations = data(coverage.call_args.args[0])["research_context"]["validation_limitations"]
    assert limitations[0]["requirement_indices"] == []
    assert "An additional statement" in result.answer
    assert FACTS[1] not in result.answer
    assert research_state.required_claims == ([] if empty_checklist else [NEEDS[0]])


def test_same_fact_with_one_good_and_one_bad_reference_preserves_good_reference_only():
    research_state = prepared(1)
    research_state.evidence.append(evidence("bad-source", filename="unrelated.md", text="Unrelated passage."))
    draft = output(1)
    draft["citation_claims"].append({"claim": FACTS[0], "chunk_id": "bad-source"})
    result = compose_answer(research_state, synthesize=lambda p: draft, validate_coverage=mapped_coverage,
        validate_semantics=lambda p: {"support": "unsupported" if data(p)["referenced_evidence"]["chunk_id"] == "bad-source"
                                     else "supported", "explicit_absence": "clear"})
    assert result.answer.count(FACTS[0]) == 1
    assert len(result.citations) == 1
    assert result.citations[0]["filename"] == research_state.evidence[0].filename


@pytest.mark.parametrize("failure", [RuntimeError("system down"), {}, {"support": "unsupported"}])
def test_system_and_malformed_failures_do_not_become_limitations(failure):
    semantic = Mock(side_effect=[{"support": "supported", "explicit_absence": "clear"}, failure])
    if isinstance(failure, Exception):
        expected = RuntimeError
    else:
        expected = ValidationError
    coverage = Mock(side_effect=mapped_coverage)
    with pytest.raises(expected):
        compose_answer(prepared(2), synthesize=lambda p: output(2), validate_coverage=coverage,
                       validate_semantics=semantic)
    coverage.assert_called_once()


@pytest.mark.parametrize("final_failure", ["incomplete", "uncertain", "malformed", "wrong_requirement", "omission", "reassignment"])
def test_final_coverage_cannot_reopen_repair_or_hide_a_limitation(final_failure):
    synthesis = Mock(return_value=output(2))
    semantic = Mock(side_effect=[{"support": "supported", "explicit_absence": "clear"},
                                 {"support": "unsupported", "explicit_absence": "clear"}])

    def coverage(prompt):
        result = mapped_coverage(prompt)
        if "validation_limitations" in data(prompt)["research_context"]:
            if final_failure in {"incomplete", "uncertain"}:
                result["coverage"] = final_failure
            elif final_failure == "malformed":
                return {}
            elif final_failure == "wrong_requirement":
                result["presentation"][0] = dict(result["presentation"][1], requirement=NEEDS[0])
            elif final_failure == "reassignment":
                result["presentation"][1] = dict(requirement=NEEDS[1], status="answered",
                                                answer_excerpt=FACTS[0], citation_claim_indices=[0])
            else:
                result["presentation"][1] = dict(requirement=NEEDS[1], status="omitted", evidence_ids=["c1"])
        return result

    with pytest.raises(ValueError):
        compose_answer(prepared(2), synthesize=synthesis, validate_coverage=coverage, validate_semantics=semantic)
    synthesis.assert_called_once()
    assert semantic.call_count == 2


@pytest.mark.parametrize("visual", [False, True])
def test_absence_and_visual_supported_claims_survive_rejected_inference(visual):
    from tests.answer_helpers import complete_coverage
    from tests.test_answer_visual_evidence import vision

    absence = "No canonical temperament is established for the test character."
    observed = "The synthetic plate shows three bands."
    inference = "The test character has a guarded temperament."
    kept = observed if visual else absence
    source = vision(observed) if visual else evidence("absence", text=absence)
    research_state = state(source, evidence("behavior", text="The test character avoided one question."))
    draft = {"answer": f"{kept} {inference}", "citation_claims": [
        {"claim": kept, "chunk_id": source.chunk_id}, {"claim": inference, "chunk_id": "behavior"}]}
    checked = []

    def semantic(prompt):
        payload = data(prompt)
        checked.append(payload["claim"])
        if payload["claim"] == inference:
            if visual:
                return {"support": "unsupported", "explicit_absence": "clear"}
            assert any(item["text"] == absence for item in payload["evidence"])
            return {"support": "supported", "explicit_absence": "violated",
                    "absence_findings": [{"chunk_id": source.chunk_id, "passage": absence}]}
        return {"support": "supported", "explicit_absence": "clear"}

    result = compose_answer(research_state, synthesize=lambda p: draft,
                            validate_coverage=complete_coverage, validate_semantics=semantic)
    assert checked == [kept, inference]
    assert kept in result.answer and inference not in result.answer
    assert result.status == "partial_validation_limited"
    assert result.citations == [dict(claim=kept, filename=source.filename, page=source.page,
                                   section=source.section, source_type=source.source_type)]

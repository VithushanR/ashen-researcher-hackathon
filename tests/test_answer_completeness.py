"""Synthetic presentation cases; deterministic model doubles, no archive truth."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.answer.composer import compose_answer
from src.answer.coverage_validation import (
    PresentationCompletenessError, validate_answer_coverage,
)
from src.answer.semantic_validation import CitationSupportError
from src.answer.synthesis import CitationClaim, SynthesisResult
from tests.test_answer_composer import evidence, state
from tests.test_answer_conflicts import conflict, research


FACTS = ["The test gate is blue.", "Orin owns the test gate.", "The test gate was built in 1400."]
REQUIREMENTS = ["What color is the test gate?", "Who owns the test gate?", "When was it built?"]


def data(prompt):
    return json.loads(prompt.split("INPUT DATA:\n", 1)[1])


def prepared_state(**overrides):
    fields = dict(question="What color is the test gate, who owns it, and when was it built?",
                  required_claims=list(REQUIREMENTS))
    fields.update(overrides)
    return state(*[evidence(f"fact_{i}", text=fact) for i, fact in enumerate(FACTS)], **fields)


def draft(count=3):
    return {"answer": " ".join(FACTS[:count]), "citation_claims": [
        {"claim": fact, "chunk_id": f"fact_{i}"} for i, fact in enumerate(FACTS[:count])
    ]}


def literal_coverage(prompt):
    """Inspect all requested synthetic facts, including facts NOT in citation_claims."""
    payload = data(prompt)
    assert payload["question"] == prepared_state().question
    assert payload["required_claims"] == REQUIREMENTS
    assert payload["evidence"] == [{"chunk_id": f"fact_{i}", "text": fact}
                                   for i, fact in enumerate(FACTS)]
    assert "Do NOT infer support from absence in unresolved_claims" in prompt
    assessments = []
    for i, (requirement, fact) in enumerate(zip(REQUIREMENTS, FACTS)):
        if fact in payload["answer"]:
            assert {"claim": fact, "chunk_id": f"fact_{i}"} in payload["citation_claims"]
            assessments.append(dict(requirement=requirement, status="answered", answer_excerpt=fact, citation_claim_indices=[i]))
        else:
            assessments.append(dict(requirement=requirement, status="omitted", evidence_ids=[f"fact_{i}"]))
    return {"coverage": "complete", "presentation": assessments}


def literal_semantics(prompt):
    payload = data(prompt)
    supported = payload["claim"] in payload["referenced_evidence"]["text"]
    return {"support": "supported" if supported else "unsupported", "explicit_absence": "clear"}


def test_all_requirements_present_no_repair():
    synthesis = Mock(return_value=draft())
    coverage = Mock(side_effect=literal_coverage)
    semantics = Mock(side_effect=literal_semantics)
    result = compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage,
                            validate_semantics=semantics)
    assert result.status == "complete"
    assert result.answer == draft()["answer"]
    synthesis.assert_called_once()
    coverage.assert_called_once()
    assert semantics.call_count == 3
    payload = data(synthesis.call_args.args[0])
    assert payload["required_claims"] == REQUIREMENTS
    assert "repair_feedback" not in payload


def test_one_repair_same_inputs_feedback_and_trusted_metadata(monkeypatch):
    # Any attempt to leave C and re-run research fails this test.
    for name in ("research", "check_sufficiency", "derive_required_claims", "resolve_conflicts"):
        monkeypatch.setattr(f"agent.loop.{name}", Mock(side_effect=AssertionError("B called by C")))
    research_state = prepared_state()
    before = deepcopy(research_state)
    prompts, calls = [], []

    def synthesize(prompt):
        calls.append("synthesis")
        payload = data(prompt)
        prompts.append(payload)
        if len(prompts) == 2:
            feedback = payload.pop("repair_feedback")
            assert feedback[0]["requirement"] == REQUIREMENTS[2]
            assert feedback[0]["evidence_ids"] == ["fact_2"]
            assert payload == prompts[0]
        return draft(2 if len(prompts) == 1 else 3)

    def coverage(prompt):
        calls.append("coverage")
        return literal_coverage(prompt)

    def semantics(prompt):
        calls.append("semantic")
        return literal_semantics(prompt)

    result = compose_answer(research_state, synthesize=synthesize, validate_coverage=coverage,
                            validate_semantics=semantics)
    assert calls == ["synthesis", "coverage", "synthesis", "coverage"] + ["semantic"] * 3
    assert research_state == before
    assert result.confidence == research_state.confidence
    assert result.iterations_used == research_state.iteration
    for citation, item in zip(result.citations, research_state.evidence):
        assert citation == dict(claim=item.text, filename=item.filename, page=item.page,
                                section=item.section, source_type=item.source_type)


def test_second_omission_fails_without_third_attempt_or_partial_result():
    synthesis = Mock(return_value=draft(2))
    semantics = Mock()
    with pytest.raises(PresentationCompletenessError):
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=literal_coverage,
                       validate_semantics=semantics)
    assert synthesis.call_count == 2
    semantics.assert_not_called()


def test_genuine_gap_is_appended_even_when_synthesis_omits_it_no_repair():
    gap = "The construction date remains unestablished."
    research_state = prepared_state(unresolved_claims=[gap])
    synthesis = Mock(return_value=draft(2))

    def coverage(prompt):
        payload = data(prompt)
        assert payload["research_context"]["unresolved_claims"] == [gap]
        assert payload["answer"].count(gap) == 1
        assert all(gap != item["claim"] for item in payload["citation_claims"])
        result = literal_coverage(prompt)
        result["presentation"][2] = dict(requirement=REQUIREMENTS[2], status="gap", answer_excerpt=gap, gap_indices=[0])
        return result

    result = compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage,
                            validate_semantics=literal_semantics)
    assert result.status == "partial_gap_stated"
    assert FACTS[2] not in result.answer
    synthesis.assert_called_once()


@pytest.mark.parametrize("resolved", [False, True])
def test_conflict_plus_independent_omission_repaired_without_changing_b(resolved):
    supplied = conflict(resolved_value="green" if resolved else None)
    research_state = research(supplied, required_claims=["Gate color?", "Gate owner?"])
    # Same chunk legitimately contains a competing and an independent fact.
    for item, claim in zip(research_state.evidence, supplied.claims):
        item.text = claim.claim
    research_state.evidence[0].text += " " + FACTS[1]
    before = deepcopy(research_state)
    synthesis = Mock(side_effect=[{"answer": None, "citation_claims": []},
        {"answer": FACTS[1], "citation_claims": [{"claim": FACTS[1], "chunk_id": "test_0"}]}])
    modes = []

    def coverage(prompt):
        payload = data(prompt)
        assert payload["research_context"]["conflicts"][0]["resolved_value"] == supplied.resolved_value
        owner_present = FACTS[1] in payload["answer"]
        return {"coverage": "complete", "presentation": [
            dict(requirement="Gate color?", status="conflict", answer_excerpt=payload["answer"], conflict_indices=[0]),
            (dict(requirement="Gate owner?", status="answered", answer_excerpt=FACTS[1], citation_claim_indices=[0]) if owner_present
             else dict(requirement="Gate owner?", status="omitted", evidence_ids=["test_0"]))]}

    def semantics(prompt):
        modes.append("conflict" if "Conflict attribution context:" in prompt else "ordinary")
        return literal_semantics(prompt)

    result = compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage,
                            validate_semantics=semantics)
    assert synthesis.call_count == 2
    assert result.status == ("complete_with_conflict" if resolved else "partial_gap_stated")
    assert result.conflicts[0]["resolved_value"] == supplied.resolved_value
    assert len(result.citations) == 3
    assert modes == ["ordinary", "conflict", "conflict"]
    assert research_state == before


def test_empty_checklist_question_fallback_no_persisted_decomposition():
    research_state = prepared_state(required_claims=[])
    synthesis = Mock(return_value=draft())

    def coverage(prompt):
        payload = data(prompt)
        assert payload["question"] == research_state.question
        assert payload["required_claims"] == []
        return {"coverage": "complete", "presentation": [dict(
            requirement=research_state.question, status="answered", answer_excerpt=payload["answer"])]}

    compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage,
                   validate_semantics=literal_semantics)
    assert research_state.required_claims == []
    assert data(synthesis.call_args.args[0])["required_claims"] == []
    synthesis.assert_called_once()


@pytest.mark.parametrize("bad_kind", ["unknown", "unsupported", "unchecked_model"])
def test_invalid_repaired_answer_still_fails(bad_kind):
    repaired = draft()
    if bad_kind == "unknown":
        repaired["citation_claims"][2]["chunk_id"] = "invented"
    elif bad_kind == "unsupported":
        repaired["citation_claims"][2]["claim"] = "The test gate is made of cheese."
        repaired["answer"] += " The test gate is made of cheese."
    else:
        repaired = SynthesisResult.model_construct(answer="", citation_claims=[])
    synthesis = Mock(side_effect=[draft(2), repaired])

    def coverage(prompt):
        # Explicitly isolate downstream validation: first verdict triggers repair,
        # second accepts presentation so it cannot mask a chunk/semantic failure.
        payload = data(prompt)
        if payload["answer"] == draft(2)["answer"]:
            return literal_coverage(prompt)
        return {"coverage": "complete", "presentation": [dict(
            requirement=req, status="answered", answer_excerpt=payload["citation_claims"][i]["claim"], citation_claim_indices=[i])
            for i, req in enumerate(REQUIREMENTS)]}

    expected_error = {"unknown": ValueError, "unsupported": CitationSupportError,
                      "unchecked_model": ValidationError}[bad_kind]
    semantics = Mock(side_effect=literal_semantics)
    with pytest.raises(expected_error) as caught:
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage,
                       validate_semantics=semantics)
    assert synthesis.call_count == 2
    if bad_kind == "unknown":
        assert "Unknown synthesis chunk_id: invented" in str(caught.value)
        semantics.assert_not_called()
    elif bad_kind == "unsupported":
        assert semantics.call_count == 3
    else:
        semantics.assert_not_called()


@pytest.mark.parametrize("mutation", ["uncertain", "missing", "extra", "reordered", "legacy",
                                    "bad_excerpt", "bad_id", "no_evidence", "coverage_failure"])
def test_unsafe_assessment_never_triggers_repair(mutation):
    synthesis = Mock(return_value=draft(2))
    semantics = Mock()

    def coverage(prompt):
        result = literal_coverage(prompt)
        if mutation == "uncertain":
            result["presentation"][2]["status"] = "uncertain"
        elif mutation == "missing":
            result["presentation"].pop()
        elif mutation == "extra":
            result["presentation"].append(result["presentation"][0])
        elif mutation == "reordered":
            result["presentation"].reverse()
        elif mutation == "legacy":
            del result["presentation"]
        elif mutation == "bad_excerpt":
            result["presentation"][0]["answer_excerpt"] = "Not in answer"
        elif mutation == "bad_id":
            result["presentation"][2]["evidence_ids"] = ["invented"]
        elif mutation == "no_evidence":
            result["presentation"][2]["evidence_ids"] = []
        else:
            result["coverage"] = "incomplete"
        return result

    with pytest.raises(ValueError):
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage,
                       validate_semantics=semantics)
    synthesis.assert_called_once()
    semantics.assert_not_called()


def test_gap_report_missing_from_final_text_fails_without_repair_permission():
    # Composer appends gaps deterministically. Exercise the coverage boundary
    # independently for a caller supplying an answer that omitted that report.
    def coverage(prompt):
        payload = data(prompt)
        assert payload["research_context"]["unresolved_claims"] == ["Unknown date."]
        assert "Unknown date." not in payload["answer"]
        return {"coverage": "complete", "presentation": [dict(
            requirement="Date?", status="uncertain")]}

    with pytest.raises(PresentationCompletenessError) as caught:
        validate_answer_coverage(FACTS[0], [CitationClaim(claim=FACTS[0], chunk_id="a")],
            question="Date?", required_claims=["Date?"], unresolved_claims=["Unknown date."],
            evidence=[{"chunk_id": "a", "text": FACTS[0]}], validate_coverage=coverage)
    assert not caught.value.repairable


def test_unknown_chunk_in_first_draft_is_not_repaired():
    response = draft(2)
    response["citation_claims"][0]["chunk_id"] = "unknown"
    synthesis = Mock(return_value=response)

    def coverage(prompt):
        # An otherwise valid omission verdict cannot authorize fixing an invalid ID.
        payload = data(prompt)
        return {"coverage": "complete", "presentation": [
            dict(requirement=req, status="answered", answer_excerpt=FACTS[i], citation_claim_indices=[i]) if i < 2
            else dict(requirement=req, status="omitted", evidence_ids=["fact_2"])
            for i, req in enumerate(payload["required_claims"])]}

    with pytest.raises(ValueError, match="Unknown synthesis chunk_id: unknown"):
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage)
    synthesis.assert_called_once()


def test_required_explicit_absence_is_an_answer_not_a_gap():
    fact = "No canonical temperament is established for the test figure."
    requirement = "What is the test figure's canonical temperament?"
    research_state = state(evidence("absence", text=fact), question=requirement,
                           required_claims=[requirement])
    synthesis = Mock(return_value={"answer": fact, "citation_claims": [
        {"claim": fact, "chunk_id": "absence"}]})

    def coverage(prompt):
        payload = data(prompt)
        assert payload["required_claims"] == [requirement]
        assert payload["research_context"]["unresolved_claims"] == []
        assert payload["citation_claims"] == [{"claim": fact, "chunk_id": "absence"}]
        assert "explicit supported absence counts" in prompt
        return {"coverage": "complete", "presentation": [dict(
            requirement=requirement, status="answered", answer_excerpt=fact, citation_claim_indices=[0])]}

    result = compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage,
                            validate_semantics=literal_semantics)
    assert result.status == "complete"
    synthesis.assert_called_once()


def test_empty_checklist_ambiguous_question_fails_without_repair():
    research_state = prepared_state(required_claims=[])
    synthesis = Mock(return_value=draft(2))
    coverage = Mock(return_value={"coverage": "complete", "presentation": [dict(
        requirement=research_state.question, status="uncertain")]})
    with pytest.raises(PresentationCompletenessError):
        compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage)
    synthesis.assert_called_once()
    assert research_state.required_claims == []


@pytest.mark.parametrize("mutation", ["status", "excerpt", "extra_field", "no_omission_ids"])
def test_nested_presentation_instances_are_revalidated(mutation):
    from src.answer.coverage_validation import PresentationAssessment, PresentationCoverageVerdict

    entry = PresentationAssessment(requirement=REQUIREMENTS[2], status="omitted", evidence_ids=["fact_2"])
    if mutation == "status":
        entry.status = "invented"
    elif mutation == "excerpt":
        entry.answer_excerpt = "Inconsistent with omission"
    elif mutation == "extra_field":
        entry = {**entry.model_dump(), "ignore_validation": True}
    else:
        entry.evidence_ids.clear()
    response = {"coverage": "complete", "presentation": [
        dict(requirement=REQUIREMENTS[i], status="answered", answer_excerpt=FACTS[i], citation_claim_indices=[i]) for i in range(2)
    ] + [entry]}
    if mutation == "status":
        response["presentation"][:2] = [PresentationAssessment(**item)
                                        for item in response["presentation"][:2]]
        response = PresentationCoverageVerdict.model_construct(**response)
    synthesis = Mock(return_value=draft(2))
    with pytest.raises(ValidationError):
        compose_answer(prepared_state(), synthesize=synthesis,
                       validate_coverage=Mock(return_value=response))
    synthesis.assert_called_once()


@pytest.mark.parametrize("status", ["answered", "omitted"])
def test_explicit_b_gap_cannot_be_marked_answered_or_repairable(status):
    research_state = prepared_state(unresolved_claims=[REQUIREMENTS[2]])
    synthesis = Mock(return_value=draft(2))

    def coverage(prompt):
        result = literal_coverage(prompt)
        if status == "answered":
            result["presentation"][2] = dict(requirement=REQUIREMENTS[2], status=status,
                                             answer_excerpt=REQUIREMENTS[2])
        return result

    with pytest.raises(ValueError, match="explicit B gap"):
        compose_answer(research_state, synthesize=synthesis, validate_coverage=coverage)
    synthesis.assert_called_once()


@pytest.mark.parametrize("mapping", [[0, 0, 0], [0, 1, 2], [0, 1, -1], [0, 1, True]])
def test_missing_requirement_cannot_pass_with_reused_or_invalid_mapping(mapping):
    synthesis = Mock(return_value=draft(2))
    semantics = Mock()

    def coverage(prompt):
        return {"coverage": "complete", "presentation": [
            dict(requirement=req, status="answered", answer_excerpt=FACTS[0] if index == 0 else FACTS[1],
                 citation_claim_indices=[index])
            for req, index in zip(REQUIREMENTS, mapping)]}

    with pytest.raises(ValueError):
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage,
                       validate_semantics=semantics)
    synthesis.assert_called_once()
    semantics.assert_not_called()


@pytest.mark.parametrize("indices", [[], [0, 0]])
def test_excerpt_alone_or_duplicate_indices_cannot_establish_answer(indices):
    def coverage(prompt):
        verdict = literal_coverage(prompt)
        verdict["presentation"][0]["citation_claim_indices"] = indices
        return verdict

    synthesis = Mock(return_value=draft())
    with pytest.raises(ValueError, match="nonempty unique references"):
        compose_answer(prepared_state(), synthesize=synthesis, validate_coverage=coverage)
    synthesis.assert_called_once()


def test_identical_requirements_may_share_the_same_atomic_claim():
    research_state = prepared_state(required_claims=[REQUIREMENTS[0], REQUIREMENTS[0]])

    def coverage(prompt):
        return {"coverage": "complete", "presentation": [dict(
            requirement=REQUIREMENTS[0], status="answered", answer_excerpt=FACTS[0],
            citation_claim_indices=[0]) for _ in range(2)]}

    result = compose_answer(research_state, synthesize=lambda p: draft(1),
                            validate_coverage=coverage, validate_semantics=literal_semantics)
    assert result.status == "complete"


def test_duplicate_claim_entries_do_not_evade_atomic_sole_reuse_check():
    output = draft(1)
    output["citation_claims"].append(dict(output["citation_claims"][0]))

    def coverage(prompt):
        return {"coverage": "complete", "presentation": [dict(
            requirement=req, status="answered", answer_excerpt=FACTS[0], citation_claim_indices=[i])
            for i, req in enumerate(REQUIREMENTS[:2])]}

    with pytest.raises(ValueError, match="reused as sole support"):
        compose_answer(prepared_state(required_claims=REQUIREMENTS[:2]), synthesize=lambda p: output,
                       validate_coverage=coverage)


def test_mapped_claim_must_appear_in_the_actual_ordinary_answer():
    output = draft()
    output["answer"] = draft(2)["answer"]
    research_state = prepared_state(unresolved_claims=[FACTS[2]])

    def coverage(prompt):
        # Claim 2 appears only in B's appended gap report, not in synthesis text.
        return {"coverage": "complete", "presentation": [dict(
            requirement=req, status="answered", answer_excerpt=FACTS[i], citation_claim_indices=[i])
            for i, req in enumerate(REQUIREMENTS)]}

    with pytest.raises(ValueError, match="ordinary answer and excerpt"):
        compose_answer(research_state, synthesize=lambda p: output, validate_coverage=coverage)


def test_conflict_attribution_cannot_be_used_as_an_ordinary_requirement_mapping():
    research_state = research(conflict(), required_claims=["Gate color?"])

    def coverage(prompt):
        payload = data(prompt)
        return {"coverage": "complete", "presentation": [dict(
            requirement="Gate color?", status="answered", answer_excerpt=payload["answer"],
            citation_claim_indices=[0])]}

    with pytest.raises(ValueError, match="Invalid presentation reference index"):
        compose_answer(research_state, synthesize=lambda p: {"answer": None, "citation_claims": []},
                       validate_coverage=coverage)


@pytest.mark.parametrize("kind", ["gap", "conflict"])
@pytest.mark.parametrize("indices", [[], [-1], [99]])
def test_context_mapping_must_reference_actual_b_context(kind, indices):
    research_state = research(conflict(), required_claims=["Gate color?"], unresolved_claims=["Unknown color."])

    def coverage(prompt):
        return {"coverage": "complete", "presentation": [dict(
            requirement="Gate color?", status=kind, answer_excerpt=data(prompt)["answer"],
            **{f"{kind}_indices": indices})]}

    with pytest.raises(ValueError):
        compose_answer(research_state, synthesize=lambda p: {"answer": None, "citation_claims": []},
                       validate_coverage=coverage)


def test_visual_requirement_uses_existing_atomic_claim_and_original_filename():
    fact = "The synthetic plate depicts a dog."
    research_state = state(evidence("visual", text=fact, filename="synthetic.png",
        source_type="image_derived", content_type="vision_description"),
        required_claims=["Which animal is depicted?"])

    def coverage(prompt):
        return {"coverage": "complete", "presentation": [dict(
            requirement=research_state.required_claims[0], status="answered", answer_excerpt=fact,
            citation_claim_indices=[0])]}

    result = compose_answer(research_state, synthesize=lambda p: {"answer": fact,
        "citation_claims": [{"claim": fact, "chunk_id": "visual"}]},
        validate_coverage=coverage, validate_semantics=literal_semantics)
    assert result.citations[0]["filename"] == "synthetic.png"
    assert result.citations[0]["source_type"] == "image_derived"

"""Requirement-aware B tests with synthetic evidence and mocked model calls."""
import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from agent import loop, sufficiency
from agent.state import ResearchState


def response(requirements, statuses):
    conflict = "conflict" in statuses
    all_supported = all(s == "supported" for s in statuses)
    return json.dumps(dict(coverage="yes" if all_supported else "partial",
        agreement="conflict" if conflict else "yes",
        verdict="conflict_detected" if conflict else "sufficient" if all_supported else "insufficient",
        missing_info=None, requirements=[dict(requirement=r, status=s,
            missing_info=None if s == "supported" else r) for r,s in zip(requirements,statuses)]))


@pytest.mark.parametrize("requirements", [["Who forged it?"], ["Who?", "When?", "Where?"],
    ["Codex date?", "Wiki date?", "Compare accounts?"]])
def test_all_supported_in_one_retrieval(monkeypatch, requirements):
    derive = Mock(return_value=list(requirements))
    monkeypatch.setattr(loop, "derive_required_claims", derive)
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    call = Mock(return_value=response(requirements, ["supported"] * len(requirements)))
    monkeypatch.setattr(sufficiency, "call_llm", call)
    evidence_text = ("Synthetic smith Orin forged the ring in 120 AS at Test Forge. "
                     "The synthetic codex dates it to 120 AS; the wiki dates it to 121 AS. "
                     "The wiki date is one year later.")
    search = Mock(return_value=[dict(chunk_id="synthetic-record", filename="synthetic.md",
        text=evidence_text, source_type="wiki", reliability="T2_curated")])
    state = loop.research("Synthetic question", search_fn=search)
    assert state.required_claims == requirements
    assert state.iteration == 1
    assert evidence_text in call.call_args.args[0]
    assert state.evidence[0].text == evidence_text
    search.assert_called_once_with("Synthetic question", k=8)
    derive.assert_called_once()
    assert json.loads(call.call_args.args[0].split("REQUIREMENT DATA:\n")[1])["required_claims"] == requirements


@pytest.mark.parametrize("capped", [False, True])
def test_remaining_need_drives_existing_planner(monkeypatch, capped):
    requirements = ["Who?", "When?", "Where?"]
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: list(requirements))
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    call = Mock(side_effect=[response(requirements, ["supported", "supported", "missing"]),
                            response(requirements, ["supported"] * 3)])
    monkeypatch.setattr(sufficiency, "call_llm", call)
    searches = []
    def search(query, k):
        searches.append(query)
        return [dict(chunk_id=str(len(searches)), filename="synthetic.md", text="Synthetic evidence",
            source_type="wiki", reliability="T2_curated")]
    state = loop.research("Who, when and where?", search_fn=search, max_iter=1 if capped else 5)
    assert searches == (["Who, when and where?"] if capped else ["Who, when and where?", "Where?"])
    assert state.required_claims == requirements
    assert state.unresolved_claims == (["Where?"] if capped else [])
    for entry in call.call_args_list:
        assert json.loads(entry.args[0].split("REQUIREMENT DATA:\n")[1])["required_claims"] == requirements


@pytest.mark.parametrize("change", ["absent", "reordered", "invented", "supported_gap", "missing_gap", "false_sufficient", "unknown_status"])
def test_bad_assessments_fail(monkeypatch, change):
    state = ResearchState(question="Q", required_claims=["Who?", "Where?"])
    raw = json.loads(response(state.required_claims, ["supported", "missing"]))
    if change == "absent": raw.pop("requirements")
    elif change == "reordered": raw["requirements"].reverse()
    elif change == "invented": raw["requirements"][0]["requirement"] = "Invented"
    elif change == "supported_gap": raw["requirements"][0]["missing_info"] = "gap"
    elif change == "missing_gap": raw["requirements"][1]["missing_info"] = None
    elif change == "false_sufficient": raw.update(verdict="sufficient", coverage="yes")
    else: raw["requirements"][1]["status"] = "maybe"
    before = deepcopy(state)
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(raw))
    with pytest.raises(ValueError): sufficiency.check_sufficiency(state)
    assert state == before


def test_empty_checklist_keeps_legacy_fallback(monkeypatch):
    monkeypatch.setattr("agent.planner.call_llm", Mock(side_effect=RuntimeError("synthetic failure")))
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    call = Mock(return_value=json.dumps(dict(coverage="yes", agreement="yes", missing_info=None, verdict="sufficient")))
    monkeypatch.setattr(sufficiency, "call_llm", call)
    state = loop.research("Q", search_fn=lambda *a, **kw: [])
    assert state.required_claims == []
    assert state.iteration == 1
    assert "REQUIREMENT DATA:" not in call.call_args.args[0]


def test_conflict_assessment_preserves_b_decision_and_other_gap(monkeypatch):
    from agent.state import Conflict, ClaimSource
    state = ResearchState(question="Q", required_claims=["Color?", "Owner?"], conflicts=[
        Conflict(attribute="color", claims=[ClaimSource(claim="red", source="A"),
            ClaimSource(claim="blue", source="B")], resolved_value=None)])
    before = deepcopy(state)
    call = Mock(return_value=response(state.required_claims, ["conflict", "missing"]))
    monkeypatch.setattr(sufficiency, "call_llm", call)
    verdict = sufficiency.check_sufficiency(state)
    assert verdict.verdict == "conflict_detected"
    assert verdict.missing_info == "Color?; Owner?"
    assert state == before
    assert json.loads(call.call_args.args[0].split("REQUIREMENT DATA:\n")[1])["conflicts"] == [state.conflicts[0].model_dump()]


def test_model_cannot_override_unresolved_conflict(monkeypatch):
    from agent.state import Conflict, ClaimSource
    state = ResearchState(question="Q", required_claims=["Color?"], conflicts=[
        Conflict(claims=[ClaimSource(claim="red", source="A"), ClaimSource(claim="blue", source="B")])])
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: response(state.required_claims, ["supported"]))
    with pytest.raises(ValueError, match="unresolved B conflict"):
        sufficiency.check_sufficiency(state)


@pytest.mark.parametrize("status,coverage,agreement,verdict", [
    ("supported", "yes", "yes", "conflict_detected"),
    ("missing", "partial", "yes", "sufficient"),
    ("uncertain", "partial", "yes", "sufficient"),
    ("conflict", "partial", "conflict", "sufficient"),
    ("conflict", "partial", "yes", "insufficient"),
    ("supported", "yes", "yes", "insufficient"),
    ("supported", "no", "yes", "insufficient"),
    ("missing", "yes", "conflict", "conflict_detected"),
    ("missing", "partial", "conflict", "insufficient"),
])
def test_contradictory_result_aborts_before_handoff(monkeypatch, status, coverage, agreement, verdict):
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: ["Owner?"])
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    raw = json.loads(response(["Owner?"], [status]))
    raw.update(coverage=coverage, agreement=agreement, verdict=verdict)
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(raw))
    search = Mock(return_value=[dict(chunk_id="test", filename="synthetic.md",
        text="Orin owns it.", source_type="wiki", reliability="T2_curated")])
    # A caller cannot receive a gapless state from research and hand it to C.
    handoff = Mock()
    with pytest.raises(ValueError):
        state = loop.research("Owner?", search_fn=search, max_iter=1)
        handoff(state)
    handoff.assert_not_called()
    search.assert_called_once()


def test_uncertain_requirement_targets_gap(monkeypatch):
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: ["Owner?"])
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    monkeypatch.setattr(sufficiency, "call_llm", Mock(side_effect=[
        response(["Owner?"], ["uncertain"]), response(["Owner?"], ["supported"])]))
    search = Mock(return_value=[dict(chunk_id="test", filename="synthetic.md",
        text="Orin may own it.", source_type="wiki", reliability="T2_curated")])
    state = loop.research("Who owns the item?", search_fn=search)
    assert [call.args[0] for call in search.call_args_list] == ["Who owns the item?", "Owner?"]
    assert state.required_claims == ["Owner?"]
    assert state.iteration == 2


@pytest.mark.parametrize("resolved", [False, True])
def test_authoritative_conflict_state_remains_unchanged(monkeypatch, resolved):
    from agent.state import Conflict, ClaimSource
    conflict = Conflict(attribute="color", claims=[ClaimSource(claim="red", source="A"),
        ClaimSource(claim="blue", source="B")], resolved_value="red" if resolved else None)
    state = ResearchState(question="Color?", required_claims=["Color?"], conflicts=[conflict])
    before = deepcopy(state)
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: response(["Color?"],
        ["supported" if resolved else "conflict"]))
    verdict = sufficiency.check_sufficiency(state)
    assert verdict.verdict == ("sufficient" if resolved else "conflict_detected")
    assert state == before


def test_empty_checklist_rejects_reverse_conflict_contradiction(monkeypatch):
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(dict(
        coverage="yes", agreement="yes", verdict="conflict_detected", missing_info=None)))
    with pytest.raises(ValueError, match="coverage/agreement"):
        sufficiency.check_sufficiency(ResearchState(question="Q"))


@pytest.mark.parametrize("existing", ["none", "resolved", "unresolved"])
def test_complete_consistency_matrix(existing):
    """Independent overall fields: no helper derives statuses from verdicts."""
    from itertools import product
    from agent.state import Conflict, ClaimSource
    conflicts = [] if existing == "none" else [Conflict(attribute="color", claims=[
        ClaimSource(claim="red", source="A"), ClaimSource(claim="blue", source="B")],
        resolved_value=None if existing == "unresolved" else "red")]
    state = ResearchState(question="Q", required_claims=["R1", "R2"], conflicts=conflicts)
    before = deepcopy(state)
    accepted = 0
    for statuses, coverage, agreement, verdict, top_gap in product(
        product(["supported", "missing", "uncertain", "conflict"], repeat=2),
        ["yes", "partial", "no"], ["yes", "no", "conflict"],
        ["sufficient", "insufficient", "conflict_detected"], [None, "Need", " "]):
        assessments = [dict(requirement=r, status=s, missing_info=None if s == "supported" else r)
                       for r,s in zip(state.required_claims,statuses)]
        raw = sufficiency.SufficiencyVerdict(coverage=coverage, agreement=agreement,
            verdict=verdict, missing_info=top_gap, requirements=assessments)
        conflict = existing == "unresolved" or "conflict" in statuses
        gap = any(s in {"missing", "uncertain"} for s in statuses)
        all_supported = statuses == ("supported", "supported")
        expected = "conflict_detected" if conflict else "insufficient" if gap else "sufficient"
        valid = (verdict == expected and ((agreement == "conflict") == conflict)
                 and (conflict or gap or agreement == "yes")
                 and (not all_supported or coverage == "yes")
                 and (not gap or coverage != "yes")
                 and (top_gap is None if verdict == "sufficient" else top_gap != " "))
        if valid:
            sufficiency._validate_consistency(raw, state)
            accepted += 1
            assert raw.verdict == "sufficient" or raw.missing_info or existing == "unresolved"
        else:
            with pytest.raises(ValueError): sufficiency._validate_consistency(raw, state)
    assert accepted > 0
    assert state == before


def test_fake_conflict_with_conflict_agreement_cannot_reach_handoff(monkeypatch):
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: ["Owner?"])
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    raw = dict(coverage="yes", agreement="conflict", verdict="conflict_detected", missing_info=None,
               requirements=[dict(requirement="Owner?", status="supported", missing_info=None)])
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(raw))
    with pytest.raises(ValueError):
        loop.research("Owner?", search_fn=lambda *a, **kw: [dict(chunk_id="x", filename="synthetic.md",
            text="Orin owns it.", source_type="wiki", reliability="T2_curated")], max_iter=1)


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_no_evidence_for_requirement_stops_with_gap(monkeypatch, limit):
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: ["R1", "R2", "R3"])
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(dict(
        coverage="partial", agreement="yes", verdict="insufficient", missing_info="R3",
        requirements=[dict(requirement=r, status=s, missing_info=None if s == "supported" else "R3")
        for r,s in [("R1","supported"),("R2","supported"),("R3","missing")]])))
    state = loop.research("Q", search_fn=lambda *a, **kw: [], max_iter=limit)
    assert state.iteration <= limit
    assert state.required_claims == ["R1", "R2", "R3"]
    assert state.unresolved_claims
    if limit: assert any("R3" in gap for gap in state.unresolved_claims)


@pytest.mark.parametrize("kind", ["missing", "uncertain", "conflict"])
def test_blank_per_requirement_need_rejected(kind):
    state = ResearchState(question="Q", required_claims=["R"])
    raw = sufficiency.SufficiencyVerdict(coverage="partial", agreement="conflict" if kind == "conflict" else "yes",
        verdict="conflict_detected" if kind == "conflict" else "insufficient", missing_info="overall need",
        requirements=[dict(requirement="R",status=kind,missing_info=" ")])
    with pytest.raises(ValueError): sufficiency._validate_consistency(raw,state)


def test_mixed_conflict_and_missing_need_survive_terminal_state(monkeypatch):
    from agent.state import Conflict, ClaimSource
    requirements = ["Color?", "Owner?"]
    monkeypatch.setattr(loop, "derive_required_claims", lambda q: list(requirements))
    supplied = Conflict(attribute="color", claims=[
        ClaimSource(claim="The item is red.", source="A", chunk_id="red"),
        ClaimSource(claim="The item is blue.", source="B", chunk_id="blue")])
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [supplied.model_copy(deep=True)])
    monkeypatch.setattr(sufficiency, "call_llm", lambda prompt: json.dumps(dict(
        coverage="partial", agreement="conflict", verdict="conflict_detected", missing_info=None,
        requirements=[dict(requirement="Color?", status="conflict", missing_info="Verify color."),
                      dict(requirement="Owner?", status="missing", missing_info="Find owner.")])))
    chunks = [dict(chunk_id=color, filename=f"synthetic-{color}.md", text=f"The item is {color}.",
                   source_type="wiki", reliability="T2_curated") for color in ["red", "blue"]]
    state = loop.research("Color and owner?", search_fn=lambda *a, **kw: chunks, max_iter=1)
    assert state.required_claims == requirements
    assert state.conflicts[0].resolved_value is None  # real unchanged B resolution policy
    assert state.unresolved_claims == ["Verify color.; Find owner."]
    assert state.trace[-1].missing == state.unresolved_claims[0]


@pytest.mark.parametrize("agreement,verdict,need,accept", [
    ("yes", "sufficient", None, True),
    ("yes", "sufficient", "gap", False),
    ("conflict", "conflict_detected", None, False),
    ("conflict", "conflict_detected", "Verify competing sources", True),
])
def test_fallback_requires_success_or_recoverable_need(agreement, verdict, need, accept):
    raw = sufficiency.SufficiencyVerdict(coverage="yes", agreement=agreement, verdict=verdict, missing_info=need)
    state = ResearchState(question="Q")
    if accept:
        sufficiency._validate_consistency(raw,state)
        assert raw.verdict == "sufficient" or raw.missing_info
    else:
        with pytest.raises(ValueError): sufficiency._validate_consistency(raw,state)

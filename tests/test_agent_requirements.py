"""Synthetic requirement decomposition and persistence; no network calls."""
import json
from unittest.mock import Mock

import pytest

from agent import planner, loop
from agent.state import ResearchState
from agent.sufficiency import SufficiencyVerdict


@pytest.mark.parametrize("question,requirements", [
    ("What happened at Test Ford?", ["What happened at Test Ford?"]),
    ("Who forged the test ring, when and where?", ["Who forged the test ring?", "When was the test ring forged?", "Where was the test ring forged?"]),
    ("Compare codex and wiki dates and explain source reliability.", ["What date does the codex give?", "What date does the wiki give?", "How do the dates compare?", "Which source is more reliable under archive rules?"]),
])
def test_decomposition(monkeypatch, question, requirements):
    call = Mock(return_value=json.dumps({"required_claims": requirements}))
    monkeypatch.setattr(planner, "call_llm", call)
    assert planner.derive_required_claims(question) == requirements
    call.assert_called_once()
    assert json.loads(call.call_args.args[0].split("QUESTION DATA:\n")[1]) == question
    assert call.call_args.kwargs == {}  # existing default FAST path


@pytest.mark.parametrize("raw", ["", "oops", "```json\n{}\n```", "null", "[]", "{}", '{"required_claims": []}', '{"required_claims": [" "]}', '{"required_claims": [1]}', '{"required_claims": ["Q"], "extra": true}'])
def test_malformed_falls_back(monkeypatch, raw, caplog):
    monkeypatch.setattr(planner, "call_llm", Mock(return_value=raw))
    assert planner.derive_required_claims("Q") == []
    assert "empty checklist" in caplog.text


def test_provider_failure_falls_back_without_details(monkeypatch, caplog):
    monkeypatch.setattr(planner, "call_llm", Mock(side_effect=RuntimeError("private response")))
    assert planner.derive_required_claims("Q") == []
    assert "private response" not in caplog.text


@pytest.mark.parametrize("exit_kind,expected_iterations", [("sufficient", 1), ("cap", 1), ("stale", 2), ("repeat", 1)])
def test_checklist_stable_before_search_and_at_exit(monkeypatch, exit_kind, expected_iterations):
    call = Mock(return_value='{"required_claims": ["Who?", "When?"]}')
    monkeypatch.setattr(planner, "call_llm", call)
    states = []
    def create(**kwargs):
        state = ResearchState(**kwargs)
        states.append(state)
        return state
    monkeypatch.setattr(loop, "ResearchState", create)
    snapshots = []
    def search(query, k):
        assert states[-1].required_claims == ["Who?", "When?"]
        snapshots.append(list(states[-1].required_claims))
        return []
    monkeypatch.setattr(loop, "detect_conflicts", lambda evidence: [])
    def check(state):
        sufficient = exit_kind == "sufficient"
        return SufficiencyVerdict(coverage="yes" if sufficient else "no", agreement="yes",
            missing_info=None if sufficient else ("Q" if exit_kind == "repeat" else f"gap {state.iteration}"),
            verdict="sufficient" if sufficient else "insufficient")
    monkeypatch.setattr(loop, "check_sufficiency", check)
    first = loop.research("Q", search_fn=search, max_iter=1 if exit_kind == "cap" else 5)
    assert first.iteration == expected_iterations
    assert first.required_claims == ["Who?", "When?"]
    assert len(snapshots) == expected_iterations
    call.assert_called_once()
    second = loop.research("Q", search_fn=search, max_iter=1)
    assert second.required_claims == first.required_claims
    assert second.required_claims is not first.required_claims


def test_queries_unchanged_and_do_not_call_model(monkeypatch):
    monkeypatch.setattr(planner, "call_llm", Mock(side_effect=AssertionError("unexpected")))
    monkeypatch.setattr(planner, "get_aliases", lambda word: ["Alias"] if word == "Name" else [])
    assert planner.plan_first_query("Name?") == "Name? Alias"
    assert planner.plan_next_query("When?", ["Name"], []) == "When? Name"
    assert planner.plan_next_query("When?", ["Name"], ["Name?"]) == "When?"

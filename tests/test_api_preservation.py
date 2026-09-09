"""Synthetic API handoff with actual composer and deterministic model boundaries."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from agent.state import Evidence, ResearchState
from src.answer.composer import compose_answer
from src.api import llm, pipeline
from src.api.main import app
from tests.answer_helpers import complete_coverage
from tests.test_api import collect_events


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("b_gap", [False, True])
def test_expected_support_failure_is_a_safe_api_answer(monkeypatch, stream, b_gap):
    question = "What color is the synthetic gate?"
    fact = "The synthetic gate is blue."
    state = ResearchState(question=question, required_claims=[question], iteration=2, confidence=70,
        unresolved_claims=["The maker is unknown."] if b_gap else [],
        evidence=[Evidence(chunk_id="synthetic", filename="synthetic.md", source_type="wiki",
                           reliability="T2_curated", text="Unrelated evidence.")])
    before = deepcopy(state)
    research = Mock(return_value=state)
    monkeypatch.setenv("ASHEN_PIPELINE", "real")
    monkeypatch.setattr(pipeline, "_load_real_pipeline", lambda: (research, compose_answer))
    synthesis = Mock(return_value={"answer": fact, "citation_claims": [{"claim": fact, "chunk_id": "synthetic"}]})
    semantic = Mock(return_value={"support": "unsupported", "explicit_absence": "clear"})
    coverage = Mock(side_effect=complete_coverage)
    for name, adapter in (("synthesize", synthesis), ("validate_semantics", semantic), ("validate_coverage", coverage)):
        monkeypatch.setattr(llm, name, adapter)
    client = TestClient(app)
    if stream:
        events = collect_events(client, {"question": question})
        assert not any(event["event"] == "error" for event in events)
        answers = [event["payload"] for event in events if event["event"] == "answer"]
        assert len(answers) == 1
        body = answers[0]
    else:
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 200
        body = response.json()
    assert body["status"] == ("partial_gap_stated" if b_gap else "partial_validation_limited")
    assert fact not in body["answer"]
    assert "Validation limitations:" in body["answer"]
    assert body["citations"] == []
    assert body["unresolved_claims"] == before.unresolved_claims
    assert body["confidence"] == 70 and body["iterations_used"] == 2
    assert state == before
    research.assert_called_once()
    synthesis.assert_called_once()
    semantic.assert_called_once()
    assert coverage.call_count == 2

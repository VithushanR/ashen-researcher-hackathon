"""Deterministic production-adapter tests; all model responses are mocked."""

import json
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent.state import Evidence, ResearchState
from src.api import llm, pipeline
from src.answer.composer import compose_answer


@pytest.fixture(autouse=True)
def shared_call(monkeypatch):
    call = Mock(side_effect=AssertionError("Unexpected model call"))
    monkeypatch.setattr(llm.llm_client, "call_llm", call)
    return call


@pytest.mark.parametrize("adapter,model", [
    (llm.synthesize, llm.SYNTHESIS_MODEL),
    (llm.validate_coverage, llm.FAST_MODEL),
    (llm.validate_semantics, llm.FAST_MODEL),
])
def test_decodes_json_and_passes_prompt_unchanged(shared_call, adapter, model):
    shared_call.side_effect = None
    shared_call.return_value = '{"answer": "Synthetic fact", "nested": [null, 1]}'
    prompt = "  Already built\nUnicode: café\n"
    assert adapter(prompt) == {"answer": "Synthetic fact", "nested": [None, 1]}
    shared_call.assert_called_once_with(prompt, model=model)


@pytest.mark.parametrize("adapter", [llm.synthesize, llm.validate_coverage, llm.validate_semantics])
@pytest.mark.parametrize("raw", ["", "   ", "{broken", '```json\n{}\n```'])
def test_invalid_json_fails(shared_call, adapter, raw):
    shared_call.side_effect = None
    shared_call.return_value = raw
    with pytest.raises(json.JSONDecodeError):
        adapter("prompt")
    assert shared_call.call_count == 1


@pytest.mark.parametrize("adapter", [llm.synthesize, llm.validate_coverage, llm.validate_semantics])
def test_provider_error_propagates(shared_call, adapter):
    error = RuntimeError("Synthetic transport failure")
    shared_call.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        adapter("prompt")
    assert caught.value is error
    assert shared_call.call_count == 1


def test_real_state_through_api_adapters_and_composer(shared_call):
    fact = "The synthetic gate is iron."
    question = "What is the synthetic gate made of?"
    state = ResearchState(question=question, required_claims=[question],
        confidence=85, iteration=2, evidence=[Evidence(
            chunk_id="synthetic-1", filename="synthetic.md", source_type="wiki",
            reliability="T2_curated", text=fact, content_type="text", page=None,
            section="Gate")])
    before = state.model_dump()
    shared_call.side_effect = [
        json.dumps({"answer": fact, "citation_claims": [
            {"claim": fact, "chunk_id": "synthetic-1"}]}),
        json.dumps({"coverage": "complete", "uncovered_claims": [], "reason": None,
            "presentation": [{"requirement": state.question, "status": "answered",
                "answer_excerpt": fact, "evidence_ids": [], "citation_claim_indices": [0],
                "gap_indices": [], "conflict_indices": [], "limitation_indices": []}]}),
        json.dumps({"support": "supported", "explicit_absence": "clear"}),
    ]
    answer = pipeline._compose_with_adapters(compose_answer, state)
    assert answer.answer == fact
    assert answer.status == "complete"
    assert answer.confidence == 85
    assert answer.iterations_used == 2
    assert answer.citations == [{"claim": fact, "filename": "synthetic.md",
                                 "page": None, "section": "Gate", "source_type": "wiki"}]
    assert [call.kwargs["model"] for call in shared_call.call_args_list] == [
        llm.SYNTHESIS_MODEL, llm.FAST_MODEL, llm.FAST_MODEL]
    assert state.model_dump() == before


def test_coverage_adapter_does_not_remove_unknown_output_fields(shared_call):
    raw = {"coverage": "complete", "presentation": [], "required_claims": ["Echoed input"]}
    shared_call.side_effect = None
    shared_call.return_value = json.dumps(raw)
    assert llm.validate_coverage("prompt") == raw


@pytest.mark.parametrize("override", [False, True])
def test_shared_model_configuration(monkeypatch, override):
    expected = {}
    for role, default in [("FAST", "gemini-2.5-flash"),
                          ("SYNTHESIS", "gemini-2.5-flash"),
                          ("VISION", "gemini-2.5-flash")]:
        variable = f"GEMINI_{role}_MODEL"
        monkeypatch.delenv(variable, raising=False)
        expected[f"{role}_MODEL"] = default
        if override:
            monkeypatch.setenv(variable, f"synthetic/{role}")
            expected[f"{role}_MODEL"] = f"synthetic/{role}"
    # Isolated execution avoids changing imported module/default-argument identity.
    config = runpy.run_path(str(Path(__file__).parents[1] / "src/agent/models.py"))
    for name, value in expected.items():
        assert config[name] == value
    assert config["DEFAULT_FAST_MODEL"] == config["FAST_MODEL"]

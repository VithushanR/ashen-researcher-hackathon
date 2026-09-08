"""Tests for Person D's API surface: the wire contract and the integration seam.

Run from the repo root:

    pytest tests/test_api.py -v

These pin the behaviour the UI depends on. Several of them exist because of a
bug that actually happened during development rather than one imagined in
advance -- those are called out in the test docstrings.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import pipeline
from src.api.main import app
from src.api.schemas import AskResponse, normalise_conflict, normalise_response

CLEAN_QUESTION = "Which war was won by the organization Isolde Mournvale was a member of?"
CONFLICT_QUESTION = "In which year was the Gauntlet of Sorrowfell actually forged?"
PARTIAL_QUESTION = "Whose dominion encompasses the lair of the Gravemaw Wyrm?"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client pinned to the fixture pipeline with no artificial step delay.

    Pinned to ``stub`` so the suite behaves identically before and after Persons
    B and C merge -- otherwise these tests would start making real API calls the
    moment their code lands, and a rate limit would look like a test failure.
    """
    monkeypatch.setenv("ASHEN_PIPELINE", "stub")
    monkeypatch.setattr(pipeline, "STUB_STEP_DELAY", 0.0)
    return TestClient(app)


def collect_events(client: TestClient, payload: dict) -> list[dict]:
    """Drain an SSE response into a list of parsed events."""
    events: list[dict] = []
    with client.stream("POST", "/ask/stream", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_reports_what_is_wired(client: TestClient) -> None:
    """ASHEN_PIPELINE=stub must report "stub" even once real components exist.

    The availability flags deliberately are not asserted to specific values.
    They describe which teammates have merged, which changes as the hackathon
    proceeds -- pinning them here would mean every merge breaks this test for
    no reason. What matters is that the mode override wins over availability,
    which is the part a demo depends on.
    """
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["pipeline"] == "stub"
    assert isinstance(body["research_available"], bool)
    assert isinstance(body["compose_available"], bool)
    assert body["streaming"] in {"live", "replayed"}


REPO_ROOT = Path(__file__).resolve().parents[1]

# Skip only when Person B's file is genuinely absent from this checkout -- i.e.
# main has not been merged into this branch yet. This is deliberately a check on
# the *file*, not a try/except around the import: "B has not landed here" and "B
# landed but we cannot import them" look identical to an ImportError, and only
# the second is a bug. Guarding on the file means the moment loop.py is present
# these tests must pass, and can no longer skip their way to green.
agent_loop_merged = pytest.mark.skipif(
    not (REPO_ROOT / "src" / "agent" / "loop.py").exists(),
    reason="src/agent/loop.py is not in this checkout; merge origin/main to run this",
)


@agent_loop_merged
def test_health_sees_person_bs_merged_research_loop() -> None:
    """The seam must actually find agent.loop:research now that it is on main.

    This is the assertion that would have caught the import-path bug: _load()
    swallows ImportError by design, so a wrong module path degrades to the stub
    silently instead of failing. Without a test that names the real target, the
    suite stays green while ASHEN_PIPELINE=auto quietly never leaves the stub.
    """
    assert pipeline.DEFAULT_RESEARCH_TARGET == "agent.loop:research"
    research = pipeline._load("ASHEN_RESEARCH_TARGET", pipeline.DEFAULT_RESEARCH_TARGET)
    assert research is not None, "Person B's research() is on main but not importable"
    assert "question" in inspect.signature(research).parameters


@agent_loop_merged
def test_the_research_loop_is_importable_without_pytests_path_help() -> None:
    """Reproduce the runtime environment, not the test environment.

    pytest.ini's `pythonpath = src .` configures the *test* process only. Under
    `uvicorn src.api.main:app` from the repo root, src/ is not on sys.path and
    Person B's `from agent.state import ...` fails. That is why this runs in a
    subprocess with a bare path and PYTHONPATH stripped: an in-process
    assertion would pass on pytest's path and prove nothing about the demo.

    _ensure_import_paths() is what makes this pass.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["ASHEN_PIPELINE"] = "auto"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.api.pipeline import pipeline_status;"
            " print(pipeline_status()['research_available'])",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True", (
        "the real research loop is not reachable the way the API actually runs; "
        f"stderr={result.stderr}"
    )


def test_health_detail_reports_robustness_and_models(client: TestClient) -> None:
    """Diagnostics must survive a missing component rather than 500.

    An endpoint you check *because* something is broken is useless if it breaks
    too. When src/api/llm.py is absent it is reported as unavailable, not raised.
    """
    body = client.get("/health/detail").json()
    assert body["robustness"]["backoff_seconds"] == [1, 2, 4, 8]
    assert body["pipeline"]["mode"]
    assert "archive_present" in body
    assert "available" in body["llm"]
    if body["llm"]["available"]:
        assert body["llm"]["fast_model"] and body["llm"]["strong_model"]
    else:
        assert body["llm"]["reason"], "an unavailable adapter must say why"


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

def test_ask_returns_the_full_contract(client: TestClient) -> None:
    response = client.post("/ask", json={"question": CLEAN_QUESTION})
    assert response.status_code == 200

    payload = AskResponse.model_validate(response.json())
    assert payload.question == CLEAN_QUESTION
    assert payload.answer
    assert payload.status in {"complete", "complete_with_conflict", "partial_gap_stated"}
    assert 0 <= payload.confidence <= 100
    assert payload.trace, "the UI trace panel needs at least one round"
    assert payload.is_stub is True, "fixture answers must be flagged, never passed off as real"


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_empty_question_is_rejected(client: TestClient, question: str) -> None:
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_typo_in_a_field_name_is_rejected_not_ignored(client: TestClient) -> None:
    """A silently-ignored ``baselin`` would disable the demo contrast with no error."""
    assert client.post("/ask", json={"question": "x", "baselin": True}).status_code == 422


def test_missing_question_is_rejected(client: TestClient) -> None:
    assert client.post("/ask", json={}).status_code == 422


@pytest.mark.parametrize(
    ("question", "expected_status"),
    [
        (CLEAN_QUESTION, "complete"),
        (CONFLICT_QUESTION, "complete_with_conflict"),
        (PARTIAL_QUESTION, "partial_gap_stated"),
    ],
)
def test_all_three_states_are_reachable(
    client: TestClient, question: str, expected_status: str
) -> None:
    """The UI must be exercised against conflict and partial cases, not just the happy path."""
    assert client.post("/ask", json={"question": question}).json()["status"] == expected_status


def test_conflict_case_carries_the_multi_claim_shape(client: TestClient) -> None:
    """docs/contracts.md replaced pairwise claim_a/claim_b with a claims list."""
    body = client.post("/ask", json={"question": CONFLICT_QUESTION}).json()

    assert len(body["conflicts"]) == 1
    conflict = body["conflicts"][0]
    assert len(conflict["claims"]) >= 2, "a conflict needs the competing claims, not just the winner"
    assert all(c["claim"] and c["source"] for c in conflict["claims"])
    assert conflict["resolved_value"], "a resolved conflict must say what won"
    assert conflict["resolution"], "and why it won"

    # The losing claim must survive to the UI. Collapsing a disagreement into a
    # single value is the failure mode this feature exists to prevent.
    #
    # Asserted structurally rather than against literal years. The fixture was
    # rewritten to quote the real archive, and hardcoded values ("312 AS",
    # "341 AS") were invented ones that pinned the test to content rather than
    # to the property being tested.
    claims = [c["claim"] for c in conflict["claims"]]
    assert conflict["resolved_value"] in " ".join(claims), "the winning claim must be present"
    dissenting = [c for c in claims if conflict["resolved_value"] not in c]
    assert dissenting, "every claim agrees -- then this is not a conflict"


def test_partial_case_states_its_gap(client: TestClient) -> None:
    body = client.post("/ask", json={"question": PARTIAL_QUESTION}).json()
    assert body["unresolved_claims"], "a partial answer must name what is missing"
    assert body["trace"][-1]["verdict"] == "capped_unresolved"


def test_clean_case_has_no_conflicts_or_gaps(client: TestClient) -> None:
    body = client.post("/ask", json={"question": CLEAN_QUESTION}).json()
    assert body["conflicts"] == []
    assert body["unresolved_claims"] == []


def test_every_answer_carries_citations(client: TestClient) -> None:
    """Mandatory citations are Tier 1 in the spec: an uncited claim is the failure."""
    for question in (CLEAN_QUESTION, CONFLICT_QUESTION, PARTIAL_QUESTION):
        citations = client.post("/ask", json={"question": question}).json()["citations"]
        assert citations, f"no citations for {question!r}"
        for citation in citations:
            assert citation["claim"] and citation["filename"]


# ---------------------------------------------------------------------------
# POST /ask/stream
# ---------------------------------------------------------------------------

def test_stream_emits_start_then_steps_then_answer(client: TestClient) -> None:
    events = collect_events(client, {"question": CLEAN_QUESTION})

    assert events[0]["event"] == "start"
    assert events[-1]["event"] == "answer"

    steps = [e["step"] for e in events if e["event"] == "step"]
    assert steps, "the trace panel needs at least one round"
    assert [s["step"] for s in steps] == sorted(s["step"] for s in steps), "rounds out of order"


def test_stream_terminates_exactly_once(client: TestClient) -> None:
    """A stream that never terminates leaves the UI spinning forever."""
    for question in (CLEAN_QUESTION, CONFLICT_QUESTION, PARTIAL_QUESTION):
        kinds = [e["event"] for e in collect_events(client, {"question": question})]
        assert kinds.count("start") == 1
        assert len([k for k in kinds if k in {"answer", "error"}]) == 1
        assert kinds[-1] in {"answer", "error"}


def test_stream_payload_matches_the_non_streaming_answer(client: TestClient) -> None:
    """Both endpoints must agree; the UI uses one and scripts use the other."""
    streamed = collect_events(client, {"question": CONFLICT_QUESTION})[-1]["payload"]
    direct = client.post("/ask", json={"question": CONFLICT_QUESTION}).json()
    assert streamed["status"] == direct["status"]
    assert streamed["confidence"] == direct["confidence"]
    assert len(streamed["trace"]) == len(direct["trace"])


def test_stream_rejects_an_empty_question(client: TestClient) -> None:
    assert client.post("/ask/stream", json={"question": "  "}).status_code == 422


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def test_baseline_toggle_returns_a_single_round(client: TestClient) -> None:
    body = client.post("/ask", json={"question": CLEAN_QUESTION, "baseline": True}).json()
    assert body["iterations_used"] == 1
    assert len(body["trace"]) == 1


def test_baseline_round_is_never_labelled_insufficient(client: TestClient) -> None:
    """Regression: the degraded baseline path reused the stub's first trace step.

    That step carries ``verdict: "insufficient"``, which claims the baseline made
    a sufficiency judgement. It makes none -- it retrieves once and answers. The
    baseline exists precisely to show what happens without that check, so
    labelling its round "insufficient" misrepresents the comparison.
    """
    body = client.post("/ask", json={"question": CLEAN_QUESTION, "baseline": True}).json()
    assert body["trace"][0]["verdict"] == "baseline_no_check"


# ---------------------------------------------------------------------------
# GET /source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "attack",
    [
        "../../../../etc/passwd",
        "..\\..\\..\\Windows\\System32\\drivers\\etc\\hosts",
        "/etc/passwd",
        "....//....//etc/passwd",
        "../.env",
    ],
)
def test_source_rejects_path_traversal(client: TestClient, attack: str) -> None:
    """The filename arrives from a model-generated citation, so it is untrusted."""
    response = client.get("/source", params={"filename": attack})
    assert response.status_code in {400, 404}
    assert "content" not in response.json() or response.json().get("content") is None


def test_source_rejects_an_empty_filename(client: TestClient) -> None:
    assert client.get("/source", params={"filename": ""}).status_code == 422


def test_source_returns_404_for_an_unknown_file(client: TestClient) -> None:
    response = client.get("/source", params={"filename": "definitely_not_in_the_archive.txt"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Pipeline modes
# ---------------------------------------------------------------------------

def test_real_mode_fails_loudly_when_the_pipeline_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI must not go green because the stub answered.

    Without this, the suite passes whether or not the real pipeline is wired --
    which makes it worthless at exactly the moment it matters.
    """
    monkeypatch.setenv("ASHEN_PIPELINE", "real")
    monkeypatch.setenv("ASHEN_RESEARCH_TARGET", "does.not.exist:research")
    monkeypatch.setenv("ASHEN_COMPOSE_TARGET", "does.not.exist:compose_answer")

    response = TestClient(app).post("/ask", json={"question": CLEAN_QUESTION})
    assert response.status_code == 503
    assert "not importable" in response.json()["detail"]


def test_a_broken_teammate_module_degrades_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A half-merged branch must not stop the server answering.

    If the import were at module scope, a teammate's module raising on import
    would take the whole API down and nobody could demo anything.
    """
    broken = tmp_path / "broken_teammate.py"
    broken.write_text("raise ImportError('chromadb not installed here')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("ASHEN_PIPELINE", "auto")
    monkeypatch.setenv("ASHEN_RESEARCH_TARGET", "broken_teammate:research")
    monkeypatch.setattr(pipeline, "STUB_STEP_DELAY", 0.0)

    response = TestClient(app).post("/ask", json={"question": CLEAN_QUESTION})
    assert response.status_code == 200
    assert response.json()["is_stub"] is True


# ---------------------------------------------------------------------------
# Shape normalisation
# ---------------------------------------------------------------------------

def test_normalise_conflict_converts_the_legacy_pairwise_shape() -> None:
    """The fixtures PDF predates the multi-claim schema in docs/contracts.md."""
    converted = normalise_conflict(
        {
            "claim_a": "Forged in 312 AS",
            "source_a": "letter_gravemaw_sighting.txt (ephemera)",
            "claim_b": "Forged in 341 AS",
            "source_b": "annals_of_the_ashen_era_vol1.pdf (codex)",
            "resolution": "Codex preferred.",
            "resolved_value": "341 AS",
        }
    )
    assert [c["claim"] for c in converted["claims"]] == ["Forged in 312 AS", "Forged in 341 AS"]
    assert converted["claims"][0]["source"].startswith("letter_gravemaw")
    assert converted["resolved_value"] == "341 AS"
    assert converted["claim_a"] == "Forged in 312 AS", "original keys are preserved, not stripped"


def test_normalise_conflict_handles_a_three_way_disagreement() -> None:
    """The reason the schema changed: pairwise cannot represent three sources."""
    converted = normalise_conflict(
        {"claim_a": "a", "source_a": "1", "claim_b": "b", "source_b": "2",
         "claim_c": "c", "source_c": "3"}
    )
    assert len(converted["claims"]) == 3


def test_normalise_conflict_leaves_the_current_shape_untouched() -> None:
    current = {"claims": [{"claim": "a", "source": "s"}], "resolved_value": "a"}
    assert normalise_conflict(current) == current


def test_normalise_response_tolerates_a_missing_conflicts_key() -> None:
    assert normalise_response({"question": "q", "answer": "a"})["conflicts"] == []
    assert normalise_response({"question": "q", "conflicts": None})["conflicts"] == []

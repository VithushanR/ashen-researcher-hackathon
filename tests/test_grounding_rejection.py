"""What happens when the composer refuses to stand behind its own answer.

Person C's grounding gate raises ``CitationCoverageError`` or
``CitationSupportError`` when a synthesised answer is not supported by the
evidence. On the first end-to-end run with all four components merged, that
fired -- and the API turned it into an HTTP 500 with the validator's message as
a stack-trace detail.

That was the wrong response to the right behaviour. The refusal *is* the system
working: it researched, it checked its own answer, it found it unsupported, and
it declined to present it. Sub-track 1C exists to reward exactly that. Showing a
judge a 500 at the moment the system behaves best is the worst possible framing
of it.

So a rejection now produces an honest ``partial_gap_stated`` answer that keeps
the trace and states the reason.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import pipeline
from src.api.main import app


class _CitationCoverageError(ValueError):
    """Same class name as Person C's, so the seam's name-based match applies."""

    __name__ = "CitationCoverageError"


# The seam matches on type(error).__name__, so the class must really be named it.
CitationCoverageError = type("CitationCoverageError", (ValueError,), {})
CitationSupportError = type("CitationSupportError", (ValueError,), {})


class _State:
    def __init__(self) -> None:
        self.trace = [
            {"step": 1, "query": "gauntlet forged", "verdict": "insufficient"},
            {"step": 2, "query": "codex forging year", "verdict": "conflict_detected"},
        ]
        self.unresolved_claims = ["Which year the ephemera intended."]
        self.conflicts = [{"attribute": "forging year", "claims": []}]
        self.route = "multihop"
        self.iteration = 2


def _wire(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Point the seam at a loop that succeeds and a composer that rejects."""
    monkeypatch.setenv("ASHEN_PIPELINE", "auto")
    monkeypatch.setattr(pipeline, "_load_retrieval", lambda: None)

    def research(question: str, **kwargs: Any) -> _State:
        return _State()

    def compose(state: Any, **kwargs: Any) -> dict:
        raise error

    monkeypatch.setattr(pipeline, "_load_real_pipeline", lambda: (research, compose))


REJECTIONS = [
    CitationCoverageError(
        "Citation coverage rejected: incomplete; the ordinary_section answer "
        "repeats a disputed assertion from the supplied conflict"
    ),
    CitationSupportError("Claim is not supported by chunk codex_02_p47_c3"),
]


@pytest.mark.parametrize("error", REJECTIONS, ids=["coverage", "support"])
def test_rejection_becomes_an_honest_partial_not_a_500(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Regression: this returned HTTP 500 on the first four-component run."""
    _wire(monkeypatch, error)
    response = TestClient(app).post("/ask", json={"question": "when was it forged?"})

    assert response.status_code == 200, "the quality gate firing is not a server error"
    body = response.json()
    assert body["status"] == "partial_gap_stated"
    assert body["confidence"] == 0
    assert body["is_stub"] is False, "this was a real run -- it must not claim to be a fixture"


@pytest.mark.parametrize("error", REJECTIONS, ids=["coverage", "support"])
def test_rejection_keeps_the_trace_and_states_the_reason(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The research still happened. Throwing the trace away hides the good part.

    A judge watching a refused answer should still see every round the agent
    performed -- that is the evidence the system did the work before declining.
    """
    _wire(monkeypatch, error)
    body = TestClient(app).post("/ask", json={"question": "when was it forged?"}).json()

    assert len(body["trace"]) == 2, "the trace must survive a rejection"
    assert body["route"] == "multihop"
    assert any(str(error)[:30] in claim for claim in body["unresolved_claims"]), (
        "the reason the answer was withheld must be stated, not summarised away"
    )
    assert body["unresolved_claims"][0] == "Which year the ephemera intended.", (
        "the loop's own unresolved claims must be kept alongside ours"
    )


def test_rejection_does_not_fall_back_to_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silently serving canned output here would be the worst of the options.

    It would show a judge a confident fixture answer at the exact moment the
    real system declined to answer -- turning honest behaviour into a lie.
    """
    _wire(monkeypatch, REJECTIONS[0])
    body = TestClient(app).post("/ask", json={"question": "when was it forged?"}).json()

    assert body["is_stub"] is False
    assert "War of Drowned Light" not in body["answer"], "that is fixture content"
    assert "grounding check" in body["answer"]


def test_a_genuine_bug_is_still_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The match is narrow on purpose.

    Both of Person C's rejections subclass ValueError, so catching ValueError
    would swallow every ordinary bug in their composer and present it as an
    honest partial answer -- which would hide real breakage behind a message
    claiming the system chose not to answer.
    """
    _wire(monkeypatch, ValueError("citations[0].filename is required"))
    response = TestClient(app).post("/ask", json={"question": "when was it forged?"})
    assert response.status_code == 500, "a real bug must not be dressed up as a considered refusal"


def test_streaming_path_degrades_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    """The UI uses /ask/stream, so fixing only /ask would fix nothing visible."""
    import json

    _wire(monkeypatch, REJECTIONS[0])
    monkeypatch.setattr(pipeline, "STUB_STEP_DELAY", 0.0)

    events = []
    with TestClient(app).stream(
        "POST", "/ask/stream", json={"question": "when was it forged?"}
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    kinds = [e["event"] for e in events]
    assert "error" not in kinds, "a refused answer is not a stream error"
    assert kinds[-1] == "answer"
    assert events[-1]["payload"]["status"] == "partial_gap_stated"

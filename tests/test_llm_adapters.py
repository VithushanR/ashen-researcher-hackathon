"""Tests for ``src/api/llm.py`` -- the model adapters Person C's composer takes.

Not one of these makes a network call. ``agent.llm_client.call_llm`` is replaced
throughout, which is the whole point of the adapter being thin: there is exactly
one place to intercept, and everything above it is prompt construction, model
selection and response parsing -- all of it testable offline.

The parsing tests use real model misbehaviour, not tidy JSON. A cheap model
wraps its answer in prose and code fences no matter what the prompt says, and a
validator that only works on clean output is a validator that fails during the
demo.
"""

from __future__ import annotations

import pytest

from src.api import llm


@pytest.fixture(autouse=True)
def no_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point diskcache somewhere disposable and bypass it.

    ``_call`` is wrapped in ``@cached``, so without this the second test to send
    the same prompt would get the first test's canned answer and pass for the
    wrong reason.
    """
    monkeypatch.setenv("ASHEN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ASHEN_NO_CACHE", "1")


def fake_call(response: str, captured: dict | None = None):
    """Stand in for Person B's ``call_llm``, recording prompt and model."""
    def _call(prompt: str, model: str = "unused") -> str:
        if captured is not None:
            captured["prompt"] = prompt
            captured["model"] = model
        return response
    return _call


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``call_llm`` into the module the adapter imports from."""
    def _install(response: str) -> dict:
        captured: dict = {}
        import agent.llm_client as client

        monkeypatch.setattr(client, "call_llm", fake_call(response, captured))
        return captured
    return _install


# ---------------------------------------------------------------------------
# Model tiering
# ---------------------------------------------------------------------------

def test_synthesis_uses_the_strong_tier(patched) -> None:
    """The whole point of the split. Synthesis is the one call worth paying for."""
    captured = patched("The Gauntlet was forged in 341 AS [chunk_002].")
    assert llm.synthesize("write the answer") == "The Gauntlet was forged in 341 AS [chunk_002]."
    assert captured["model"] == llm.STRONG_MODEL


@pytest.mark.parametrize(
    "call",
    [
        lambda: llm.validate_semantics("a claim", "some evidence"),
        lambda: llm.validate_coverage("a question", "an answer"),
    ],
)
def test_validators_use_the_cheap_tier(patched, call) -> None:
    """Both validators run on every composed answer.

    Putting them on the strong tier is how a free-tier quota dies halfway
    through a demo, so this is pinned rather than left to a comment.
    """
    captured = patched('{"supported": true, "complete": true, "missing": []}')
    call()
    assert captured["model"] == llm.FAST_MODEL


# ---------------------------------------------------------------------------
# Parsing what a cheap model actually returns
# ---------------------------------------------------------------------------

def test_json_is_extracted_from_fences_and_prose(patched) -> None:
    """Real output from a cheap model, not the output the prompt asked for."""
    patched(
        "Sure! Here is my assessment:\n\n"
        '```json\n{"supported": false, "reason": "The evidence names the smith '
        'but never dates the forging."}\n```\n\nLet me know if you need more.'
    )
    result = llm.validate_semantics("Forged in 341 AS", "Forged by Aldric Vane.")
    assert result["supported"] is False
    assert "never dates" in result["reason"]


def test_unparseable_semantics_defaults_to_supported(patched) -> None:
    """A validator with no answer must not invent a failure.

    Defaulting to unsupported would let a broken parse strip citations off a
    correct answer -- claiming a problem that was never detected, which is the
    same dishonesty as claiming a check that was never run.
    """
    patched("I'm sorry, I can't help with that.")
    assert llm.validate_semantics("c", "e")["supported"] is True


def test_unparseable_coverage_defaults_to_incomplete(patched) -> None:
    """Coverage defaults the other way, and the asymmetry is deliberate.

    An unchecked answer claiming full coverage is the failure sub-track 1C is
    about. Saying "I could not verify this" costs a sentence in the UI; saying
    "complete" when nothing checked is the thing the system exists to prevent.
    """
    patched("no json here")
    result = llm.validate_coverage("q", "a")
    assert result["complete"] is False
    assert result["missing"] == ["coverage check response unparseable"]


def test_coverage_with_gaps_is_never_reported_complete(patched) -> None:
    """A model that says complete:true and then lists a gap contradicts itself."""
    patched('{"complete": true, "missing": ["the second half of the question"]}')
    result = llm.validate_coverage("two-part question", "half an answer")
    assert result["complete"] is False
    assert result["missing"] == ["the second half of the question"]


def test_coverage_normalises_a_bare_string_missing_field(patched) -> None:
    patched('{"complete": false, "missing": "who forged it"}')
    assert llm.validate_coverage("q", "a")["missing"] == ["who forged it"]


# ---------------------------------------------------------------------------
# What /health/detail reports
# ---------------------------------------------------------------------------

def test_llm_unavailable_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert llm.llm_available() is False


def test_placeholder_model_ids_count_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key alone is not readiness.

    ``agent/models.py`` ships ``PLACEHOLDER_FAST_MODEL`` until someone confirms
    the ids against OpenRouter's live catalogue. That reaches the API and returns
    HTTP 400 with a message that never mentions the model id, so it is worth
    naming here rather than debugging at 11pm on submission day.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "FAST_MODEL", "PLACEHOLDER_FAST_MODEL")
    monkeypatch.setattr(llm, "STRONG_MODEL", "deepseek/deepseek-r1:free")
    assert llm.llm_available() is False
    assert llm.model_status()["models_confirmed"] is False


def test_llm_available_when_key_and_models_are_both_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm, "FAST_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    monkeypatch.setattr(llm, "STRONG_MODEL", "deepseek/deepseek-r1:free")
    assert llm.llm_available() is True


def test_health_detail_survives_a_broken_llm_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint you check *because* something is broken must never 500.

    This already caught one bug on 7 September, when ``/health/detail`` raised
    ``ModuleNotFoundError`` because ``llm.py`` did not exist. It exists now, so
    the check is re-pointed at the general case rather than deleted -- the next
    thing to break in here will not be a missing file.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app

    def explode(*args, **kwargs):
        raise RuntimeError("models.py is mid-merge")

    monkeypatch.setattr(llm, "model_status", explode)
    body = TestClient(app).get("/health/detail").json()

    assert body["llm"]["available"] is False
    assert "mid-merge" in body["llm"]["reason"]
    assert "pipeline" in body, "the rest of the diagnostic must still be reported"


# ---------------------------------------------------------------------------
# The adapter is an adapter
# ---------------------------------------------------------------------------

def test_there_is_exactly_one_openrouter_client() -> None:
    """No second provider integration hiding in the interface layer.

    Person B owns the transport. If this module ever grows its own ``requests``
    call and its own backoff, the repo has two OpenRouter clients with two retry
    policies -- a question the team would deserve to be asked at judging.
    """
    source = (
        __import__("pathlib").Path(llm.__file__).read_text(encoding="utf-8")
    )
    assert "openrouter.ai" not in source, "the HTTP endpoint belongs to agent/llm_client.py"
    assert "import requests" not in source
    assert "from agent.llm_client import call_llm" in source

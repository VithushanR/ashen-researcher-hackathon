"""
Tests for the shared OpenRouter LLM client wrapper.

`requests.post` is monkeypatched everywhere, so these tests make NO real
network calls and need NO API key — fast, deterministic, quota-safe.
They verify:
  - call_llm(prompt) uses the default fast model
  - call_llm(prompt, model=...) sends the explicitly chosen model
  - the response text is extracted from OpenRouter's response shape
  - retry/backoff retries on failure and eventually raises
"""
import json

import pytest

from agent import llm_client
from agent.models import DEFAULT_FAST_MODEL, SYNTHESIS_MODEL


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _capture_post(captured, content="ok"):
    """Return a fake requests.post that records the payload it was given."""
    def _post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return _FakeResponse(content)
    return _post


def test_default_model_is_fast(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", _capture_post(captured, "hello"))

    result = llm_client.call_llm("some prompt")

    assert result == "hello"
    assert captured["payload"]["model"] == DEFAULT_FAST_MODEL
    assert captured["payload"]["messages"][0]["content"] == "some prompt"


def test_explicit_model_is_sent(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("requests.post", _capture_post(captured, "synthesized"))

    result = llm_client.call_llm("write the answer", model=SYNTHESIS_MODEL)

    assert result == "synthesized"
    assert captured["payload"]["model"] == SYNTHESIS_MODEL


def test_retry_then_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)  # don't actually wait

    calls = {"n": 0}
    def _always_fail(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        raise RuntimeError("network down")
    monkeypatch.setattr("requests.post", _always_fail)

    with pytest.raises(RuntimeError, match="failed after"):
        llm_client.call_llm("prompt")

    assert calls["n"] == 4  # _MAX_RETRIES attempts made

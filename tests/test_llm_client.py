"""
Tests for the shared Gemini LLM client wrapper.

`google.genai.Client` is monkeypatched everywhere, so these tests make NO
real network calls and need NO API key — fast, deterministic, quota-safe.
They verify:
  - call_llm(prompt) uses the default fast model
  - call_llm(prompt, model=...) sends the explicitly chosen model
  - the response text is extracted from Gemini's response shape
  - retry/backoff retries on failure and eventually raises
"""
import pytest

from agent import llm_client
from agent.models import DEFAULT_FAST_MODEL, SYNTHESIS_MODEL


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    """Stands in for client.models, recording the call it was given."""

    def __init__(self, captured, content="ok"):
        self._captured = captured
        self._content = content

    def generate_content(self, model=None, contents=None):
        self._captured["model"] = model
        self._captured["contents"] = contents
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, captured, content="ok"):
        self.models = _FakeModels(captured, content)


def _fake_genai_client(captured, content="ok"):
    """Return a fake genai.Client(...) constructor that records the call."""
    def _client(api_key=None):
        captured["api_key"] = api_key
        return _FakeClient(captured, content)
    return _client


def test_default_model_is_fast(monkeypatch):
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", _fake_genai_client(captured, "hello"))

    result = llm_client.call_llm("some prompt")

    assert result == "hello"
    assert captured["model"] == DEFAULT_FAST_MODEL
    assert captured["contents"] == "some prompt"


def test_explicit_model_is_sent(monkeypatch):
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", _fake_genai_client(captured, "synthesized"))

    result = llm_client.call_llm("write the answer", model=SYNTHESIS_MODEL)

    assert result == "synthesized"
    assert captured["model"] == SYNTHESIS_MODEL


def test_markdown_code_fence_is_stripped(monkeypatch):
    """Gemini wraps JSON in ```json ... ``` even when told not to; every
    caller does raw json.loads(call_llm(...)), so this must come off here."""
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    fenced = '```json\n{"coverage": "complete"}\n```'
    monkeypatch.setattr(google.genai, "Client", _fake_genai_client(captured, fenced))

    result = llm_client.call_llm("prompt")

    assert result == '{"coverage": "complete"}'


def test_plain_text_without_fence_is_unchanged(monkeypatch):
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", _fake_genai_client(captured, "plain text, no fence"))

    assert llm_client.call_llm("prompt") == "plain text, no fence"


def test_retry_then_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)  # don't actually wait

    calls = {"n": 0}

    class _AlwaysFailClient:
        class models:
            @staticmethod
            def generate_content(model=None, contents=None):
                calls["n"] += 1
                raise RuntimeError("network down")

    import google.genai
    monkeypatch.setattr(google.genai, "Client", lambda api_key=None: _AlwaysFailClient())

    with pytest.raises(RuntimeError, match="failed after"):
        llm_client.call_llm("prompt")

    assert calls["n"] == 4  # _MAX_RETRIES attempts made

"""
Shared LLM call wrapper — every component that needs an LLM call goes
through here, not through a provider SDK directly.

Currently using Gemini directly as a temporary stand-in. Per the
Technical Spec's Technology Stack, the intended provider is OpenRouter
(gives model flexibility + free-tier quota across 4 team accounts) —
swap _client and call_llm's request logic to OpenRouter once everyone's
accounts are set up. The call_llm(prompt) -> str signature should not
need to change, so nothing calling this file (sufficiency.py, later
conflict.py) needs to change when that swap happens.

Retry/backoff (spec §2.13, explicitly requested by the challenge
document) lives here, in one place, rather than duplicated across every
file that needs an LLM call.
"""
import os
import time

_MODEL_NAME = "gemini-2.0-flash"  # cheap/fast tier — confirm current free-tier model name before relying on this

_MAX_RETRIES = 4
_BACKOFF_SECONDS = [1, 2, 4, 8]  # exponential backoff, per spec §2.13

# The model is created lazily on first real use, NOT at import time.
#
# Why: the API key and the provider SDK are only needed when an actual
# LLM call is made. Configuring the client at import made this whole
# module (and everything that imports it — sufficiency.py, conflict.py,
# loop.py) crash the instant it was imported on any machine without
# GEMINI_API_KEY set. That broke `pytest` on a clean checkout even
# though the tests monkeypatch call_llm and never hit the network — a
# judge running the suite would see a KeyError, not passing tests.
# Deferring construction keeps import side-effect-free.
#
# NOTE: google.generativeai is deprecated by Google. This still works
# today but should move to the maintained `google.genai` package, or
# (per the Technical Spec's stack) to OpenRouter. The call_llm(prompt)
# -> str signature will not change when that swap happens, so nothing
# that imports this file needs to change.
_model = None


def _get_model():
    """Build (once) and return the LLM client. Called only on a real request."""
    global _model
    if _model is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # never hardcode — .env + .gitignore
        _model = genai.GenerativeModel(_MODEL_NAME)
    return _model


def call_llm(prompt: str) -> str:
    """
    Send a single-turn prompt to the LLM and return the raw text response.

    Retries with exponential backoff on failure. Raises the last
    exception if all retries fail, since a silent empty-string return
    would corrupt whatever parses this response downstream (e.g.
    sufficiency.py's JSON parsing).
    """
    model = _get_model()
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as e:  # narrow this once we know Gemini's actual exception types
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"LLM call failed after {_MAX_RETRIES} attempts") from last_error

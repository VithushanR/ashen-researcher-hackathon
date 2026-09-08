"""
Central model configuration — the single source of truth for which
OpenRouter model each tier uses. Both Person B and Person C read these
names; nobody hardcodes a model id in their own module.

Why centralized: the spec's model-tiering note wants a cheap model for
the many loop steps and a stronger one reserved for final synthesis.
Keeping the ids here means locking the real OpenRouter ids is a
one-file edit that both B and C pick up automatically — no hunting
through call sites.

IMPORTANT — these values are UNVERIFIED PLACEHOLDERS.
The exact OpenRouter ids have not been confirmed against OpenRouter's
live free-tier catalog yet (model availability changes). Do NOT trust
these as correct — verify each against https://openrouter.ai/models
and replace here before any real (non-mocked) run. They are wired to
read from environment variables first so they can be locked without a
code change:

    FAST_MODEL      = os.environ["OPENROUTER_FAST_MODEL"]  (or the default below)
    SYNTHESIS_MODEL = os.environ["OPENROUTER_SYNTHESIS_MODEL"]
    VISION_MODEL    = os.environ["OPENROUTER_VISION_MODEL"]
"""
import os

# TODO(team): confirm these three against OpenRouter's live catalog and
# lock them. Until then they are placeholders, overridable via env so we
# never have to touch code to change them.

# Cheap/fast text model — B's sufficiency + conflict checks, C's coverage
# + semantic validation. Runs many times per question, so it must be cheap.
FAST_MODEL = os.environ.get("OPENROUTER_FAST_MODEL", "PLACEHOLDER_FAST_MODEL")

# Stronger text model — reserved for C's final answer synthesis only.
SYNTHESIS_MODEL = os.environ.get("OPENROUTER_SYNTHESIS_MODEL", "PLACEHOLDER_SYNTHESIS_MODEL")

# Vision-capable model — B's vision fallback (image -> description).
VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "PLACEHOLDER_VISION_MODEL")

# Default model used by call_llm() when a caller doesn't specify one.
# Points at the fast tier so B's existing no-arg calls stay cheap.
DEFAULT_FAST_MODEL = FAST_MODEL

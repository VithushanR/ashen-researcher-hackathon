"""
Central model configuration — the single source of truth for which
Gemini model each tier uses. Both Person B and Person C read these
names; nobody hardcodes a model id in their own module.

Why centralized: the spec's model-tiering note wants a cheap model for
the many loop steps and a stronger one reserved for final synthesis.
Keeping the ids here means locking the real Gemini ids is a one-file
edit that both B and C pick up automatically — no hunting through call
sites.

Defaults below were confirmed live against the real Gemini API (not
guessed) before being set: `gemini-2.5-pro` returned 404 "no longer
available to new users" for this key's project, and `gemini-2.0-flash`
no longer appears in this account's model list at all, so both are
avoided here in favour of `gemini-2.5-flash`, which responded
successfully (including multimodal image input, for the vision tier)
at the time of writing. Re-verify with `genai.Client(...).models.list()`
before changing these, since Google's available-model lineup moves.

Environment overrides are read at import time; set them before
importing the shared client.
"""
import os

# Cheap/fast text model — B's sufficiency + conflict checks, C's coverage
# + semantic validation. Runs many times per question, so it must be cheap.
FAST_MODEL = os.environ.get("GEMINI_FAST_MODEL", "gemini-2.5-flash")

# Stronger text model — reserved for C's final answer synthesis only.
SYNTHESIS_MODEL = os.environ.get("GEMINI_SYNTHESIS_MODEL", "gemini-2.5-flash")

# Vision-capable model — B's vision fallback (image -> description).
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

# Default model used by call_llm() when a caller doesn't specify one.
# Points at the fast tier so B's existing no-arg calls stay cheap.
DEFAULT_FAST_MODEL = FAST_MODEL

"""
Central model configuration — the single source of truth for which
OpenRouter model each tier uses. Both Person B and Person C read these
names; nobody hardcodes a model id in their own module.

Why centralized: the spec's model-tiering note wants a cheap model for
the many loop steps and a stronger one reserved for final synthesis.
Keeping the ids here means locking the real OpenRouter ids is a
one-file edit that both B and C pick up automatically — no hunting
through call sites.

Defaults below are the team's selected model IDs. Environment overrides
are read at import time; set them before importing the shared client.
Availability and pricing are not verified by this module.
"""
import os

# Cheap/fast text model — B's sufficiency + conflict checks, C's coverage
# + semantic validation. Runs many times per question, so it must be cheap.
FAST_MODEL = os.environ.get("OPENROUTER_FAST_MODEL", "nex-agi/nex-n2.5-pro:free")

# Stronger text model — reserved for C's final answer synthesis only.
SYNTHESIS_MODEL = os.environ.get("OPENROUTER_SYNTHESIS_MODEL", "nex-agi/nex-n2.5-pro:free")

# Vision-capable model — B's vision fallback (image -> description).
VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "nex-agi/nex-n2.5-pro:free")

# Default model used by call_llm() when a caller doesn't specify one.
# Points at the fast tier so B's existing no-arg calls stay cheap.
DEFAULT_FAST_MODEL = FAST_MODEL

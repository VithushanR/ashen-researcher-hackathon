"""
Vision fallback (spec §2.15 — Tier 3).

Fires only when retrieval surfaces an image chunk (content_type ==
"image") whose text/caption does NOT actually answer the question — i.e.
the fact lives in the pixels, not in any indexed text. In that case the
loop sends the image itself to a vision-capable model and feeds the
description back in as ordinary evidence.

Kept separate from llm_client.py because it's a different call shape
(an image goes up, not just a prompt) and a different, more expensive
model tier. Retry/backoff still mirrors llm_client so a flaky vision
call doesn't crash the loop.

Provider: Google Gemini, via the `google-genai` package -- see
llm_client.py's docstring for why (OpenRouter's free tier became
unusable; the older `google-generativeai` SDK is deprecated). The
client is built lazily on first real call so importing this module
never requires a key or network — same fix applied to llm_client.py, so
pytest and a clean checkout stay side-effect-free.
"""
import os
import time
from pathlib import Path

from agent.models import VISION_MODEL

_MAX_RETRIES = 4
_BACKOFF_SECONDS = [1, 2, 4, 8]  # exponential backoff, per spec §2.13

# Same archive-root env vars build_corpus.py resolves against (ASHEN_ARCHIVE_ROOT
# preferred, CORPUS_PATH as the pre-existing fallback) -- one shared root rather
# than a second, independently-named var that can drift from ingestion's.
def _archive_root() -> Path | None:
    root = os.environ.get("ASHEN_ARCHIVE_ROOT") or os.environ.get("CORPUS_PATH")
    return Path(root) if root else None


# Ingestion writes standalone images under three separate folders, not one --
# see build_corpus.py's images/, wiki/ and codex/ passes. A bare filename from
# an image chunk does not say which, so each candidate is checked in turn.
_IMAGE_SUBDIRS = ("images", "wiki/images", "codex/images")


def _resolve_image_path(filename: str) -> Path:
    """Find the real file behind an image chunk's bare filename.

    Raises FileNotFoundError if it isn't in any of the three known locations
    (a broken index is a real problem, not something to paper over with an
    empty description that would silently mislead the loop).
    """
    root = _archive_root()
    if root is None:
        raise FileNotFoundError(
            f"Image chunk referenced {filename!r} but neither ASHEN_ARCHIVE_ROOT "
            "nor CORPUS_PATH is set, so no archive root is known."
        )

    candidates = [root / subdir / filename for subdir in _IMAGE_SUBDIRS]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Image chunk referenced {filename!r} but no file was found in any of: "
        f"{', '.join(str(c) for c in candidates)}. Check ASHEN_ARCHIVE_ROOT/"
        "CORPUS_PATH and Person A's ingestion output."
    )


def _read_image_bytes(path: Path) -> bytes:
    """Read an image file's raw bytes for the API."""
    return path.read_bytes()


def describe_image(filename: str, question: str) -> str:
    """
    Send one image to the vision model with a targeted prompt and return
    its textual description, to be fed back into the loop as evidence.

    `filename` is the bare filename from the image chunk; it's resolved
    against the archive's images/, wiki/images/ and codex/images/ folders in
    turn (see _resolve_image_path). Raises FileNotFoundError if the file
    isn't in any of them (a broken index is a real problem, not something to
    paper over with an empty description that would silently mislead the
    loop).

    Retries with exponential backoff; raises RuntimeError if every
    attempt fails, rather than returning "" — a silent empty description
    would be indistinguishable from "the image shows nothing relevant"
    and would corrupt the loop's reasoning.
    """
    image_path = _resolve_image_path(filename)

    prompt = (
        "You are examining a single image from a fantasy lore archive. "
        f"The question being researched is: {question}\n\n"
        "Describe only what is visually depicted that is relevant to that "
        "question — motifs, emblems, objects, figures, heraldry, colours. "
        "State plainly what you can see. If the image does not contain "
        "anything relevant to the question, say exactly: NOTHING RELEVANT."
    )
    image_bytes = _read_image_bytes(image_path)

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            from google import genai  # imported lazily so module import needs no deps/key
            from google.genai import types
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(
                model=VISION_MODEL,
                contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
            )
            return resp.text or ""
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"Vision call failed after {_MAX_RETRIES} attempts") from last_error

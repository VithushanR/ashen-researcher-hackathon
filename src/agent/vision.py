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

Provider: OpenRouter, per the Technical Spec's stack. The model is
built lazily on first real call so importing this module never requires
a key or network — same fix applied to llm_client.py, so pytest and a
clean checkout stay side-effect-free.
"""
import base64
import os
import time
from pathlib import Path

from agent.models import VISION_MODEL

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MAX_RETRIES = 4
_BACKOFF_SECONDS = [1, 2, 4, 8]  # exponential backoff, per spec §2.13

# Where standalone image files live, so a bare filename from an image
# chunk can be resolved to an actual file to send. Person A's ingestion
# owns this directory; confirm the real path with them before relying on
# it. Overridable via env so it isn't hardcoded to one machine.
_IMAGE_DIR = Path(os.environ.get("ASHEN_IMAGE_DIR", "corpus/images"))


def _encode_image(path: Path) -> str:
    """Read an image file and return a base64 data string for the API."""
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def describe_image(filename: str, question: str) -> str:
    """
    Send one image to the vision model with a targeted prompt and return
    its textual description, to be fed back into the loop as evidence.

    `filename` is the bare filename from the image chunk; it's resolved
    against _IMAGE_DIR. Raises FileNotFoundError if the file isn't there
    (a broken index is a real problem, not something to paper over with
    an empty description that would silently mislead the loop).

    Retries with exponential backoff; raises RuntimeError if every
    attempt fails, rather than returning "" — a silent empty description
    would be indistinguishable from "the image shows nothing relevant"
    and would corrupt the loop's reasoning.
    """
    image_path = _IMAGE_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image chunk referenced {filename!r} but no file found at {image_path}. "
            f"Check ASHEN_IMAGE_DIR and Person A's ingestion output."
        )

    prompt = (
        "You are examining a single image from a fantasy lore archive. "
        f"The question being researched is: {question}\n\n"
        "Describe only what is visually depicted that is relevant to that "
        "question — motifs, emblems, objects, figures, heraldry, colours. "
        "State plainly what you can see. If the image does not contain "
        "anything relevant to the question, say exactly: NOTHING RELEVANT."
    )
    b64 = _encode_image(image_path)

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    }

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            import requests  # imported lazily so module import needs no deps/key
            resp = requests.post(
                _OPENROUTER_URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"Vision call failed after {_MAX_RETRIES} attempts") from last_error

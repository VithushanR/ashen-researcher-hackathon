"""
Extracts text and structured references from wiki/ markdown articles.

Wiki markdown carries two reference syntaxes that the rest of the
pipeline needs pulled out separately from the prose: ![alt](images/x.png)
image references (so images/ files can be matched back to the wiki page
that displays them) and [[Name]] wikilinks (entity signals used later by
the alias glossary). Heading hierarchy is left untouched in the returned
text, "#"-prefixed, exactly as authored — the chunker/metadata layer
resolves the nearest preceding heading into the "section" field, the same
way it does for docx headings re-emitted as markdown.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_REF_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def parse_markdown(path: str) -> dict:
    """
    Parse a markdown file, extracting its raw text plus image references
    and wikilinks.

    Args:
        path: Path to the .md file.

    Returns:
        Dict with keys:
            "text": the full markdown text, headings preserved as "#" lines.
            "image_refs": list of {"alt_text": str, "path": str} dicts,
                one per ![alt](path) reference found.
            "wikilinks": list of entity name strings extracted from
                [[Name]] (or [[Name|display text]]) tokens.
        Returns {"text": "", "image_refs": [], "wikilinks": []} if the
        file can't be read.
    """
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read markdown file: %s", path)
        return {"text": "", "image_refs": [], "wikilinks": []}

    image_refs = [
        {"alt_text": alt.strip(), "path": ref_path.strip()}
        for alt, ref_path in _IMAGE_REF_RE.findall(text)
    ]

    wikilinks = [name.strip() for name in _WIKILINK_RE.findall(text)]

    return {"text": text, "image_refs": image_refs, "wikilinks": wikilinks}

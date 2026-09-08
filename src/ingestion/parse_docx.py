"""
Extracts text from .docx sources (chronicles, codex, ephemera).

Heading structure is recoverable from a paragraph's Word style name
("Heading 1", "Heading 2", ...), so headings are re-emitted as markdown
-style "#" lines inline with the text. This lets downstream code (the
chunker, section-fallback metadata) recover section boundaries the same
way it already does for markdown files, without a second data shape.

Paragraph text is NOT read via python-docx's own `paragraph.text`
property: that property concatenates run text directly and silently
drops <w:br/> (manual line break, i.e. Shift+Enter) and <w:tab/>
elements entirely, since neither is a run of text. A paragraph that's
been manually wrapped inside Word — common in narrative prose — loses
the space at every wrap point as a result (e.g. "the ribsof the
ceiling"), with no error raised anywhere. _paragraph_text() below walks
each run's XML children directly so a line break/tab becomes a space
instead of disappearing.
"""

import logging
import re
from pathlib import Path

import docx
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

_HEADING_STYLE_RE = re.compile(r"^Heading (\d+)$", re.IGNORECASE)

_BREAK_TAGS = {qn("w:br"), qn("w:cr"), qn("w:tab")}


def _paragraph_text(paragraph) -> str:
    """Extract a paragraph's text, treating <w:br/>/<w:cr/>/<w:tab/> as
    spaces instead of silently dropping them (see module docstring)."""
    parts: list[str] = []
    for run in paragraph.runs:
        for child in run._element:
            if child.tag == qn("w:t"):
                parts.append(child.text or "")
            elif child.tag in _BREAK_TAGS:
                parts.append(" ")
    return "".join(parts)


def parse_docx(path: str) -> str:
    """
    Extract text from a .docx file, preserving heading structure as
    markdown-style "#" prefixes wherever Word heading styles are used.

    Args:
        path: Path to the .docx file.

    Returns:
        The document's text as a single string, paragraphs separated by
        newlines. Returns an empty string if the file can't be parsed.
    """
    path = Path(path)

    try:
        document = docx.Document(path)
    except Exception:
        logger.exception("Failed to open DOCX: %s", path)
        return ""

    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = re.sub(r" {2,}", " ", _paragraph_text(paragraph)).strip()
        if not text:
            continue

        heading_match = _HEADING_STYLE_RE.match(paragraph.style.name or "")
        if heading_match:
            level = int(heading_match.group(1))
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)

    return "\n".join(lines)

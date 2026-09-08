"""
Extracts text from born-digital PDFs (chronicles, codex, ephemera) page by page.

PyMuPDF indexes pages from 0 internally. Every other part of the pipeline
(citations, chunk metadata, downstream consumers) expects 1-based, human
-readable page numbers, so the conversion happens right here at the source
and nowhere else needs to know PyMuPDF's convention exists.
"""

import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def parse_pdf(path: str) -> list[tuple[int, str]]:
    """
    Extract text from a PDF, one entry per page.

    Args:
        path: Path to the PDF file.

    Returns:
        List of (page_number, text) tuples, page_number is 1-based.
        Returns an empty list if the file can't be opened or parsed.
    """
    path = Path(path)
    pages: list[tuple[int, str]] = []

    try:
        doc = pymupdf.open(path)
    except Exception:
        logger.exception("Failed to open PDF: %s", path)
        return pages

    try:
        for zero_indexed_page_num, page in enumerate(doc):
            try:
                text = page.get_text()
            except Exception:
                logger.exception(
                    "Failed to extract text from page %d of %s",
                    zero_indexed_page_num + 1,
                    path,
                )
                continue
            pages.append((zero_indexed_page_num + 1, text))
    finally:
        doc.close()

    return pages

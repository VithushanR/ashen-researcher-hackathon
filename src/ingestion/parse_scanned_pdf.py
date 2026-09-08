"""
Extracts text from scanned PDFs (no text layer) via OCR.

Files named with ".scan." (e.g. crookgate_keep.scan.pdf) are simulated
scans and go straight to OCR without checking for a text layer first,
since ephemera intentionally ships both a clean and a .scan. version of
some documents as separate items. Files without ".scan." in the name are
still checked for near-zero extractable text as a fallback, in case a
genuinely scanned document wasn't named that way.

Return shape matches parse_pdf's list[(page_number, text)] rather than a
flat string, so OCR'd pages get real 1-based page numbers in the chunk
contract instead of page=null.
"""

import logging
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

_OCR_RENDER_ZOOM = 2.0  # ~144 DPI, a reasonable OCR accuracy/speed tradeoff
_NEAR_ZERO_TEXT_CHARS = 10  # per page, below this we treat it as no text layer


def is_scanned(path: str) -> bool:
    """
    Decide whether a PDF should be routed to OCR: either its filename
    contains ".scan." (case-insensitive), or PyMuPDF extracts near-zero
    text from every page.
    """
    path = Path(path)

    if ".scan." in path.name.lower():
        return True

    try:
        doc = pymupdf.open(path)
    except Exception:
        logger.exception("Failed to open PDF while checking for text layer: %s", path)
        return False

    try:
        total_chars = sum(len(page.get_text().strip()) for page in doc)
        page_count = doc.page_count
    finally:
        doc.close()

    return total_chars < _NEAR_ZERO_TEXT_CHARS * page_count if page_count else False


def parse_scanned_pdf(path: str) -> list[tuple[int, str]]:
    """
    OCR a scanned PDF page by page.

    Args:
        path: Path to the PDF file.

    Returns:
        List of (page_number, text) tuples, page_number is 1-based.
        Returns an empty list if the file can't be opened or OCR'd.
    """
    path = Path(path)
    pages: list[tuple[int, str]] = []

    try:
        doc = pymupdf.open(path)
    except Exception:
        logger.exception("Failed to open PDF for OCR: %s", path)
        return pages

    try:
        matrix = pymupdf.Matrix(_OCR_RENDER_ZOOM, _OCR_RENDER_ZOOM)
        for zero_indexed_page_num, page in enumerate(doc):
            page_number = zero_indexed_page_num + 1
            try:
                pixmap = page.get_pixmap(matrix=matrix)
                image = Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
                text = pytesseract.image_to_string(image)
            except Exception:
                logger.exception(
                    "OCR failed on page %d of %s", page_number, path
                )
                continue
            pages.append((page_number, text))
    finally:
        doc.close()

    return pages

"""
Extracts tables from codex PDFs/DOCX, and indexes standalone figure plates
from images/.

Tables are pulled out via pdfplumber (not PyMuPDF, which doesn't do table
structure) and kept as single atomic units — a table is never split
across chunk boundaries, so it's handed downstream pre-formed rather than
passed through chunk_text().

Standalone images in images/ have no text layer, so they can't carry more
than a thin caption. index_standalone_images tries to attribute each one
to the document that actually names it, in priority order:
  1. exact match against a wiki page's parsed ![alt](images/x.png) refs
     (the alt text becomes the caption, the wiki page's document_id/
     entities are inherited)
  2. filename-stem match against a wiki page's filename (e.g.
     isolde_mournvale.png <-> isolde_mournvale.md)
  3. no match — the image still gets a chunk, just with no document_id
     and a caption guessed from its filename

Only wiki_chunks is available to match against (per the function
signature), so figure plates whose only mention is inside a codex PDF's
prose (not a wiki page) fall through to the filename-stem/no-match path;
attributing those would need the codex document text as well, which
build_corpus.py doesn't currently pass in here.

Both functions return intermediate dicts, not final locked-contract
chunks — chunk_id, source_type, and reliability are assigned uniformly
later by metadata.attach_metadata (file 7), the same as text chunks.
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _serialize_table(rows: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows of cells) as readable text."""
    lines = []
    for row in rows:
        cells = [(cell or "").strip() for cell in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_codex_tables(path: str) -> list[dict]:
    """
    Extract every table from a PDF, one atomic dict per table.

    Args:
        path: Path to the PDF file.

    Returns:
        List of dicts with keys:
            "text": the table serialized as readable pipe-delimited rows.
            "page": 1-based page number the table appears on.
            "content_type": always "table".
        Returns an empty list if the file can't be opened or has no tables.
    """
    path = Path(path)
    tables: list[dict] = []

    try:
        pdf = pdfplumber.open(path)
    except Exception:
        logger.exception("Failed to open PDF for table extraction: %s", path)
        return tables

    try:
        for zero_indexed_page_num, page in enumerate(pdf.pages):
            try:
                page_tables = page.extract_tables()
            except Exception:
                logger.exception(
                    "Table extraction failed on page %d of %s",
                    zero_indexed_page_num + 1,
                    path,
                )
                continue

            for raw_table in page_tables:
                if not raw_table:
                    continue
                tables.append(
                    {
                        "text": _serialize_table(raw_table),
                        "page": zero_indexed_page_num + 1,
                        "content_type": "table",
                    }
                )
    finally:
        pdf.close()

    return tables


def _humanize_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").strip().title()


def index_standalone_images(images_dir: str, wiki_chunks: list[dict]) -> list[dict]:
    """
    Build one image chunk per file in images_dir, attributing each to the
    wiki document (if any) that references it.

    Args:
        images_dir: Path to the images/ directory.
        wiki_chunks: Intermediate per-document dicts produced while
            parsing wiki/ markdown, each expected to carry:
                "filename": source .md filename
                "document_id": that document's id
                "entities": list of entity names for that document
                "image_refs": list of {"alt_text", "path"} from
                    parse_markdown, i.e. the raw ![alt](path) references
                    found in that page before chunking.

    Returns:
        List of intermediate image dicts with keys:
            "text": caption (alt text if matched, else a humanized
                filename guess).
            "filename": the image's own filename.
            "document_id": inherited from the matched wiki document, or
                None if unmatched.
            "entities": inherited from the matched wiki document, or a
                single-item list guessed from the filename if unmatched.
            "page": always None (standalone images aren't paginated).
            "section": always None.
            "content_type": always "image".
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        logger.warning("Images directory not found: %s", images_dir)
        return []

    # Build a lookup from referenced image basename -> owning wiki doc.
    ref_index: dict[str, dict] = {}
    for wiki_chunk in wiki_chunks:
        for image_ref in wiki_chunk.get("image_refs", []):
            basename = Path(image_ref["path"]).name.lower()
            ref_index[basename] = {
                "document_id": wiki_chunk.get("document_id"),
                "entities": wiki_chunk.get("entities", []),
                "alt_text": image_ref.get("alt_text", ""),
            }

    # Build a lookup from wiki filename stem -> owning wiki doc, for the
    # filename-matching fallback.
    stem_index: dict[str, dict] = {}
    for wiki_chunk in wiki_chunks:
        filename = wiki_chunk.get("filename")
        if not filename:
            continue
        stem_index[Path(filename).stem.lower()] = {
            "document_id": wiki_chunk.get("document_id"),
            "entities": wiki_chunk.get("entities", []),
        }

    image_chunks: list[dict] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        basename = image_path.name.lower()
        stem = image_path.stem.lower()

        match = ref_index.get(basename)
        if match:
            caption = match["alt_text"] or _humanize_filename(image_path.name)
            document_id = match["document_id"]
            entities = list(match["entities"])
        else:
            stem_match = stem_index.get(stem)
            if stem_match:
                caption = _humanize_filename(image_path.name)
                document_id = stem_match["document_id"]
                entities = list(stem_match["entities"])
            else:
                caption = _humanize_filename(image_path.name)
                document_id = None
                entities = [caption] if caption else []

        image_chunks.append(
            {
                "text": caption,
                "filename": image_path.name,
                "document_id": document_id,
                "entities": entities,
                "page": None,
                "section": None,
                "content_type": "image",
            }
        )

    return image_chunks

"""
Orchestrates the full ingestion pipeline over the real corpus at
CORPUS_PATH (.env) and writes chunks.jsonl + alias_glossary.json.

Per-file flow:
  chronicles/*.pdf|*.docx  -> parse_pdf/parse_docx -> chunk_text -> "novel" T3
  wiki/*.md                -> parse_markdown -> chunk_text -> "wiki" T2
                              (image_refs collected for standalone-image matching)
  codex/*.pdf|*.docx       -> parse_pdf/parse_docx -> chunk_text -> "codex" T1
                              (+ parse_codex_tables for *.pdf, one atomic chunk/table)
  ephemera/*.pdf|*.docx|*.txt -> ".scan." or near-zero text layer -> OCR via
                              parse_scanned_pdf; otherwise parse_pdf/parse_docx/
                              raw read -> chunk_text -> "ephemera" T4
  images/*                 -> index_standalone_images, matched against wiki
                              image_refs -> "wiki" T2 if matched, else "ephemera"
                              T4 (see metadata.py module docstring)

A file that fails to parse is logged and skipped — it does not abort the run.

Two deliberate scope decisions, since neither was specified in the task list:
  - Text/table chunks get entities from entities.extract_entities(), a
    regex capitalized-phrase heuristic (no NER model is installed in this
    project). Image chunks keep their own separate entity source —
    filename/reference matching in parse_codex_tables.py — since a thin
    caption isn't good heuristic-extraction material.
  - PDF-sourced chunks carry `page` (no headings to parse from get_text()
    output); docx/markdown-sourced chunks carry `section` instead, resolved
    from the nearest preceding "#"-style heading (both parse_docx and
    parse_markdown emit heading lines in that form) since they have no
    page numbers at all. This applies the contract's stated
    page-null-therefore-section-fallback purpose to docx too, not just
    markdown/wiki as literally written, because a docx-sourced novel chunk
    with both page and section null would have no citation fallback at all.
  - PDF text is chunked per-page (not across page boundaries), so that
    every chunk's page number stays accurate — a short page may therefore
    yield a chunk below min_words. Citation accuracy is prioritized over
    strict word-count adherence.
"""

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion.aliases import build_alias_glossary
from src.ingestion.chunker import chunk_text
from src.ingestion.entities import extract_entities
from src.ingestion.metadata import attach_metadata, derive_document_id
from src.ingestion.parse_codex_tables import index_standalone_images, parse_codex_tables
from src.ingestion.parse_docx import parse_docx
from src.ingestion.parse_markdown import parse_markdown
from src.ingestion.parse_pdf import parse_pdf
from src.ingestion.parse_scanned_pdf import is_scanned, parse_scanned_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNK_DEFAULTS = {"min_words": 400, "max_words": 600, "overlap_words": 50}

_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


class ChunkIdCounter:
    """Per-document running chunk index, so text/table chunks from the same
    source file never collide even when built in separate passes."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = Counter()

    def next(self, document_id: str) -> int:
        index = self._counts[document_id]
        self._counts[document_id] += 1
        return index


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, body) blocks on '#'-style heading lines.
    Body with no preceding heading gets section=None."""
    matches = list(_HEADING_LINE_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections


def _build_text_chunks(
    text: str,
    source_file_path: str,
    folder_name: str,
    document_id: str,
    id_counter: ChunkIdCounter,
) -> list[dict]:
    """Chunk plain text (docx/txt) into final contract dicts, one per
    section-scoped sub-chunk, section populated from nearest heading."""
    chunks = []
    for section, body in _split_into_sections(text):
        for piece in chunk_text(body, **CHUNK_DEFAULTS):
            chunks.append(
                attach_metadata(
                    piece,
                    source_file_path,
                    folder_name,
                    id_counter.next(document_id),
                    document_id=document_id,
                    page=None,
                    section=section,
                    content_type="text",
                    entities=extract_entities(piece),
                )
            )
    return chunks


def _build_paginated_text_chunks(
    pages: list[tuple[int, str]],
    source_file_path: str,
    folder_name: str,
    document_id: str,
    id_counter: ChunkIdCounter,
) -> list[dict]:
    """Chunk PDF pages independently so each chunk keeps an accurate page
    number; never merges text across a page boundary."""
    chunks = []
    for page_number, page_text in pages:
        for piece in chunk_text(page_text, **CHUNK_DEFAULTS):
            chunks.append(
                attach_metadata(
                    piece,
                    source_file_path,
                    folder_name,
                    id_counter.next(document_id),
                    document_id=document_id,
                    page=page_number,
                    section=None,
                    content_type="text",
                    entities=extract_entities(piece),
                )
            )
    return chunks


def _process_pdf_or_docx(
    path: Path, folder_name: str, id_counter: ChunkIdCounter
) -> list[dict]:
    document_id = derive_document_id(str(path))

    if path.suffix.lower() == ".pdf":
        pages = parse_scanned_pdf(str(path)) if is_scanned(str(path)) else parse_pdf(str(path))
        chunks = _build_paginated_text_chunks(
            pages, str(path), folder_name, document_id, id_counter
        )
        if folder_name == "codex":
            for table in parse_codex_tables(str(path)):
                chunks.append(
                    attach_metadata(
                        table["text"],
                        str(path),
                        folder_name,
                        id_counter.next(document_id),
                        document_id=document_id,
                        page=table["page"],
                        section=None,
                        content_type="table",
                        entities=extract_entities(table["text"]),
                    )
                )
        return chunks

    # .docx
    text = parse_docx(str(path))
    return _build_text_chunks(text, str(path), folder_name, document_id, id_counter)


def _process_ephemera_txt(
    path: Path, id_counter: ChunkIdCounter
) -> list[dict]:
    document_id = derive_document_id(str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to read ephemera text file: %s", path)
        return []
    return _build_text_chunks(text, str(path), "ephemera", document_id, id_counter)


def _drop_duplicate_pdf_versions(paths: list[Path]) -> list[Path]:
    """
    Given every file in a folder, drop the .pdf half of any exact-stem
    .docx/.pdf pair (both files' content is identical — confirmed by
    direct chunk-text comparison during manual retrieval testing; this
    was previously indexing both, wasting retrieval slots on duplicate
    evidence for the same query).

    Only triggers on an exact base-filename match with just the
    extension differing, and never on a filename containing ".scan."
    (case-insensitive) — those are simulated-scan documents with
    genuinely different content from their clean counterpart and must
    both be indexed. Deliberately filename-pattern-only, no content
    comparison, since that's all this specific duplication needs.
    """
    by_stem: dict[str, dict[str, Path]] = {}
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in {".docx", ".pdf"}:
            continue
        by_stem.setdefault(path.stem, {})[suffix] = path

    pdf_paths_to_skip: set[Path] = set()
    for stem, variants in by_stem.items():
        if ".scan." in stem.lower():
            continue
        if ".docx" in variants and ".pdf" in variants:
            pdf_path = variants[".pdf"]
            pdf_paths_to_skip.add(pdf_path)
            logger.info(
                "Skipping duplicate PDF %s, using .docx version instead.", pdf_path
            )

    return [path for path in paths if path not in pdf_paths_to_skip]


def build_corpus() -> None:
    load_dotenv()
    corpus_path = os.getenv("CORPUS_PATH")
    if not corpus_path:
        raise RuntimeError(
            "CORPUS_PATH is not set in .env — add CORPUS_PATH=<path to corpus root> "
            "before running build_corpus."
        )

    corpus_root = Path(corpus_path)
    if not corpus_root.is_dir():
        raise RuntimeError(f"CORPUS_PATH does not point to a directory: {corpus_root}")

    output_dir = Path(os.getenv("OUTPUT_DIR", "data"))
    output_dir.mkdir(parents=True, exist_ok=True)

    id_counter = ChunkIdCounter()
    all_chunks: list[dict] = []
    wiki_documents_for_image_matching: list[dict] = []

    per_folder_counts: Counter = Counter()
    scan_ocr_count = 0
    failed_files: list[str] = []

    # --- chronicles ---
    chronicles_dir = corpus_root / "chronicles"
    chronicles_paths = (
        _drop_duplicate_pdf_versions(sorted(chronicles_dir.rglob("*")))
        if chronicles_dir.is_dir()
        else []
    )
    for path in chronicles_paths:
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        try:
            chunks = _process_pdf_or_docx(path, "chronicles", id_counter)
            all_chunks.extend(chunks)
            per_folder_counts["chronicles"] += len(chunks)
        except Exception:
            logger.exception("Failed to process chronicles file: %s", path)
            failed_files.append(str(path))

    # --- codex ---
    codex_dir = corpus_root / "codex"
    codex_paths = (
        _drop_duplicate_pdf_versions(sorted(codex_dir.rglob("*")))
        if codex_dir.is_dir()
        else []
    )
    for path in codex_paths:
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        try:
            chunks = _process_pdf_or_docx(path, "codex", id_counter)
            all_chunks.extend(chunks)
            per_folder_counts["codex"] += len(chunks)
        except Exception:
            logger.exception("Failed to process codex file: %s", path)
            failed_files.append(str(path))

    # --- ephemera ---
    ephemera_dir = corpus_root / "ephemera"
    for path in sorted(ephemera_dir.rglob("*")) if ephemera_dir.is_dir() else []:
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            continue
        try:
            if suffix == ".pdf" and is_scanned(str(path)):
                scan_ocr_count += 1
            if suffix == ".txt":
                chunks = _process_ephemera_txt(path, id_counter)
            else:
                chunks = _process_pdf_or_docx(path, "ephemera", id_counter)
            all_chunks.extend(chunks)
            per_folder_counts["ephemera"] += len(chunks)
        except Exception:
            logger.exception("Failed to process ephemera file: %s", path)
            failed_files.append(str(path))

    # --- wiki ---
    wiki_dir = corpus_root / "wiki"
    for path in sorted(wiki_dir.rglob("*.md")) if wiki_dir.is_dir() else []:
        try:
            document_id = derive_document_id(str(path))
            parsed = parse_markdown(str(path))
            chunks = _build_text_chunks(
                parsed["text"], str(path), "wiki", document_id, id_counter
            )
            all_chunks.extend(chunks)
            per_folder_counts["wiki"] += len(chunks)
            wiki_documents_for_image_matching.append(
                {
                    "filename": path.name,
                    "document_id": document_id,
                    "entities": [],
                    "image_refs": parsed["image_refs"],
                }
            )
        except Exception:
            logger.exception("Failed to process wiki file: %s", path)
            failed_files.append(str(path))

    # --- images ---
    images_dir = corpus_root / "images"
    images_indexed = 0
    if images_dir.is_dir():
        try:
            image_entries = index_standalone_images(
                str(images_dir), wiki_documents_for_image_matching
            )
        except Exception:
            logger.exception("Failed to index standalone images in: %s", images_dir)
            image_entries = []

        for entry in image_entries:
            image_path = images_dir / entry["filename"]
            matched = entry["document_id"] is not None
            folder_for_metadata = "wiki" if matched else "ephemera"
            try:
                chunk = attach_metadata(
                    entry["text"],
                    str(image_path),
                    folder_for_metadata,
                    id_counter.next(entry["document_id"] or derive_document_id(str(image_path))),
                    document_id=entry["document_id"],
                    page=entry["page"],
                    section=entry["section"],
                    content_type="image",
                    entities=entry["entities"],
                )
            except Exception:
                logger.exception("Failed to build metadata for image: %s", image_path)
                failed_files.append(str(image_path))
                continue
            all_chunks.append(chunk)
            images_indexed += 1

    # --- alias glossary ---
    alias_glossary = build_alias_glossary(all_chunks)

    # --- write outputs ---
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    glossary_path = output_dir / "alias_glossary.json"
    with glossary_path.open("w", encoding="utf-8") as f:
        json.dump(alias_glossary, f, ensure_ascii=False, indent=2)

    print("=== Ingestion summary ===")
    for folder in ("chronicles", "wiki", "codex", "ephemera"):
        print(f"  {folder}: {per_folder_counts[folder]} chunks")
    print(f"  images indexed: {images_indexed}")
    print(f"  .scan. files OCR'd: {scan_ocr_count}")
    print(f"  total chunks: {len(all_chunks)}")
    print(f"  alias glossary entries: {len(alias_glossary)}")
    if failed_files:
        print(f"  FAILED to parse ({len(failed_files)}):")
        for f in failed_files:
            print(f"    - {f}")
    print(f"  wrote {chunks_path}")
    print(f"  wrote {glossary_path}")


if __name__ == "__main__":
    build_corpus()

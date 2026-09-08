"""
Assembles the final locked-contract chunk dict from a chunk's text plus
what's known about where it came from.

This is the single place source_type/reliability get decided (from which
corpus folder a file lives in) and where chunk_id/document_id are minted,
so every other ingestion module only has to produce intermediate dicts —
this one is what turns them into the exact shape the retrieval layer and
both teammates depend on.

Standalone images (folder images/) don't map to one of the four corpus
folders directly. Per team decision: build_corpus.py resolves that ahead
of calling this function — it passes folder_name="wiki" when
index_standalone_images matched an image to a wiki page (inheriting that
page's tier), or folder_name="ephemera" when unmatched (lowest-trust
default, since provenance is unconfirmed).
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_FOLDER_TO_SOURCE_TYPE = {
    "codex": "codex",
    "wiki": "wiki",
    "chronicles": "novel",
    "ephemera": "ephemera",
}

_SOURCE_TYPE_TO_RELIABILITY = {
    "codex": "T1_authoritative",
    "wiki": "T2_curated",
    "novel": "T3_narrative",
    "ephemera": "T4_unverified",
}

_ID_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _sanitize_for_id(value: str) -> str:
    return _ID_SANITIZE_RE.sub("_", value.lower()).strip("_")


def derive_document_id(source_file_path: str) -> str:
    """Stable document id derived from a source file's path stem."""
    return _sanitize_for_id(Path(source_file_path).stem)


def attach_metadata(
    chunk_text: str,
    source_file_path: str,
    folder_name: str,
    chunk_index: int,
    *,
    document_id: str | None = None,
    page: int | None = None,
    section: str | None = None,
    content_type: str = "text",
    entities: list[str] | None = None,
) -> dict:
    """
    Build a final, locked-contract chunk dict.

    Args:
        chunk_text: The chunk's text content.
        source_file_path: Path to the original source file.
        folder_name: Which corpus folder the source file lives in —
            one of "codex", "wiki", "chronicles", "ephemera". Determines
            source_type and reliability. (See module docstring for how
            standalone images map onto this.)
        chunk_index: This chunk's position within its source document,
            used to build a stable, deterministic chunk_id.
        document_id: Id linking this chunk to others from the same
            source document. Defaults to a sanitized form of the source
            file's stem if not given.
        page: 1-based page number, or None.
        section: Heading/section string, or None.
        content_type: "text" | "table" | "image".
        entities: Entity names mentioned in/depicted by this chunk.

    Returns:
        Dict matching the locked chunk contract exactly.
    """
    if folder_name not in _FOLDER_TO_SOURCE_TYPE:
        raise ValueError(
            f"Unknown folder_name {folder_name!r}; expected one of "
            f"{sorted(_FOLDER_TO_SOURCE_TYPE)}"
        )

    source_type = _FOLDER_TO_SOURCE_TYPE[folder_name]
    reliability = _SOURCE_TYPE_TO_RELIABILITY[source_type]

    filename = Path(source_file_path).name
    resolved_document_id = document_id or derive_document_id(source_file_path)
    chunk_id = f"{resolved_document_id}:{chunk_index:04d}"

    return {
        "chunk_id": chunk_id,
        "document_id": resolved_document_id,
        "filename": filename,
        "source_type": source_type,
        "reliability": reliability,
        "page": page,
        "section": section,
        "content_type": content_type,
        "text": chunk_text,
        "entities": entities or [],
    }

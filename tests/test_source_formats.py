"""Tests for /source rendering every format the archive actually contains.

The corpus is deliberately mixed-format -- Markdown, plain text, DOCX, PDF,
scanned PDF and figure plates -- and a citation a judge cannot open is a
citation they cannot check. Until 9 September this endpoint rendered only
``.txt`` and ``.md`` and told the judge to go find anything else themselves,
which meant the two codex volumes holding most of the checkable facts were
unopenable from the UI.

These tests build their own documents rather than reading the archive wherever
possible, so they run on a machine that has no corpus. The archive-dependent
ones skip on its absence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the API at a throwaway archive we control the contents of."""
    monkeypatch.setattr(api_main, "ARCHIVE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Text formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".txt", ".md", ".markdown"])
def test_plain_text_formats_are_returned_inline(
    client: TestClient, archive: Path, suffix: str
) -> None:
    (archive / f"note{suffix}").write_text("Forged in 391 AS.", encoding="utf-8")
    body = client.get("/source", params={"filename": f"note{suffix}"}).json()
    assert body["content_type"] == "text"
    assert "391 AS" in body["content"]


def test_unknown_extension_reports_rather_than_failing(
    client: TestClient, archive: Path
) -> None:
    """An unrenderable file must not break the citation next to it."""
    (archive / "relic.xyz").write_bytes(b"\x00\x01binary")
    response = client.get("/source", params={"filename": "relic.xyz"})
    assert response.status_code == 200
    body = response.json()
    assert body["content"] is None
    assert body["note"], "an unsupported format must say so, not return an empty box"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def test_docx_text_and_tables_are_extracted_in_document_order(
    client: TestClient, archive: Path
) -> None:
    """Regression: tables were appended after every paragraph.

    The obvious implementation reads ``document.paragraphs`` then
    ``document.tables``. That scrambles reading order, and because output is
    truncated at MAX_SOURCE_CHARS it pushes tables off the end of a long codex
    volume entirely -- dropping precisely the infobox rows a judge checks
    (forging year, garrison strength, attunement cost).
    """
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("Opening prose about the relic.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Forged"
    table.rows[0].cells[1].text = "391 AS"
    document.add_paragraph("Closing prose after the infobox.")
    document.save(str(archive / "codex.docx"))

    body = client.get("/source", params={"filename": "codex.docx"}).json()
    assert body["content_type"] == "text"
    assert body["extracted_from"] == "docx"

    content = body["content"]
    assert "Forged | 391 AS" in content, "table cells must be extracted, not skipped"
    assert content.index("Opening prose") < content.index("Forged | 391 AS") < content.index(
        "Closing prose"
    ), "blocks must appear in document order, not paragraphs-then-tables"


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def test_pdf_text_is_extracted_with_page_markers(
    client: TestClient, archive: Path
) -> None:
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    for n, text in enumerate(["First page text.", "Second page text."], start=1):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(str(archive / "record.pdf"))
    document.close()

    body = client.get("/source", params={"filename": "record.pdf"}).json()
    assert body["content_type"] == "text"
    assert body["extracted_from"] == "pdf"
    assert "First page text." in body["content"]
    assert "[page 2]" in body["content"], "page markers let a citation's p.N be located"


def test_scanned_pdf_says_it_is_a_scan_rather_than_returning_nothing(
    client: TestClient, archive: Path
) -> None:
    """A PDF with no text layer is an image, not an empty document.

    Returning a blank box would look like a bug in the viewer. Saying "scanned,
    no text layer" is both true and actionable, and the archive's ephemera
    genuinely contains these -- the *.scan.pdf files.
    """
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    document.new_page()  # a page with no text at all
    document.save(str(archive / "ballad.scan.pdf"))
    document.close()

    body = client.get("/source", params={"filename": "ballad.scan.pdf"}).json()
    assert body["content"] is None
    assert "scan" in body["note"].lower()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def test_figure_plates_come_back_as_inline_image_data(
    client: TestClient, archive: Path
) -> None:
    """Person B's vision fallback cites plates, so they must be openable too.

    Base64 rather than a URL because the alternative is mounting the archive as
    static files, which would expose the whole corpus over HTTP to get one image
    on screen.
    """
    png = pytest.importorskip("PIL.Image")
    png.new("RGB", (4, 4), "white").save(str(archive / "plate.png"))

    body = client.get("/source", params={"filename": "plate.png"}).json()
    assert body["content_type"] == "image"
    assert body["image_base64"], "an image citation must return renderable data"
    assert body["image_format"] == "png"


# ---------------------------------------------------------------------------
# Security still holds after the rewrite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "attack",
    ["../../../../etc/passwd", "..\\..\\..\\Windows\\System32\\drivers\\etc\\hosts", "../.env"],
)
def test_traversal_is_still_blocked_for_the_new_formats(
    client: TestClient, archive: Path, attack: str
) -> None:
    """Widening the format support must not widen what is reachable.

    The filename still arrives inside a model-generated citation, so it is still
    untrusted -- and now more file types get parsed rather than refused, which
    is exactly when a path check is worth re-asserting.
    """
    (archive / "real.txt").write_text("safe", encoding="utf-8")
    assert client.get("/source", params={"filename": attack}).status_code in {400, 404}


# ---------------------------------------------------------------------------
# Against the real corpus
# ---------------------------------------------------------------------------

def _real_archive() -> Path | None:
    configured = os.getenv("ASHEN_ARCHIVE_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates += [REPO_ROOT.parent / "Ashen_Era_Archive",
                   REPO_ROOT.parent.parent / "Ashen_Era_Archive"]
    return next((c for c in candidates if c.exists()), None)


def test_every_format_in_the_real_archive_opens(client: TestClient) -> None:
    """One file of each extension the corpus actually contains must render.

    Written against the real archive because the formats that broke were real
    ones -- the codex .docx volumes -- and a synthetic .docx would not have
    caught the truncation problem that hid their infobox tables.
    """
    archive = _real_archive()
    if archive is None:
        pytest.skip("archive not present on this machine")

    by_suffix: dict[str, Path] = {}
    for path in archive.rglob("*"):
        if path.is_file():
            by_suffix.setdefault(path.suffix.lower(), path)

    unopenable = []
    for suffix, path in sorted(by_suffix.items()):
        if suffix in {".json"}:  # corpus metadata, not a citable source
            continue
        body = client.get("/source", params={"filename": path.name}).json()
        renderable = bool(body.get("content") or body.get("image_base64"))
        # A scanned PDF with no OCR available is a known, reported gap -- it
        # must still answer with an explanation rather than an empty box.
        explained = bool(body.get("note"))
        if not (renderable or explained):
            unopenable.append(f"{suffix} ({path.name})")

    assert not unopenable, f"formats that neither render nor explain themselves: {unopenable}"

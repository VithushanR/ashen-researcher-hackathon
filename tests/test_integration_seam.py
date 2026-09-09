"""Tests for the seam against Persons A and B as actually merged.

Every test here exists because of a bug that was real on 8 September, not one
imagined in advance. They share one theme, which is the theme of this whole
project: **a green signal is not evidence that the real thing is reachable.**
Each bug below passed 113 tests, reported itself truthfully in ``/health``, and
was still wrong.

The discipline these follow, carried over from ``test_api.py``: a test skips on a
teammate's *file being absent*, never on a caught ``ImportError``. "Person A has
not landed in this checkout" and "Person A landed but we cannot reach them" are
indistinguishable to ``except ImportError``, and only the second is a bug. Once
the file exists, these must pass and can no longer skip their way to green.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from src.api import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configured targets must name something that actually exists
# ---------------------------------------------------------------------------

def _target_source_path(target: str) -> Path | None:
    """Map a ``module:function`` target to the file it should live in.

    Resolved on disk rather than by importing, on purpose. Person A's retrieval
    modules import ``voyageai`` and ``chromadb`` at module scope, so importing
    them to check a name requires the full install and a built index -- which
    means the check would be skipped on most machines and would therefore never
    have caught the mis-spelled target it exists to catch.

    A path plus an AST parse needs neither.
    """
    module_name = target.partition(":")[0]
    parts = module_name.split(".")
    for root in (REPO_ROOT, REPO_ROOT / "src"):
        candidate = root.joinpath(*parts).with_suffix(".py")
        if candidate.exists():
            return candidate
    return None


def _defines(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign))
        and (
            getattr(node, "name", None) == name
            or any(
                isinstance(t, ast.Name) and t.id == name
                for t in getattr(node, "targets", [])
            )
        )
        for node in tree.body
    )


@pytest.mark.parametrize(
    ("label", "target", "landed_when"),
    [
        ("research", pipeline.DEFAULT_RESEARCH_TARGET, "src/agent/loop.py"),
        ("retrieval", pipeline.DEFAULT_RETRIEVAL_TARGET, "src/retrieval/hybrid_search.py"),
        ("baseline", pipeline.DEFAULT_BASELINE_TARGET, "src/retrieval/baseline_rag.py"),
        ("compose", pipeline.DEFAULT_COMPOSE_TARGET, "src/answer/composer.py"),
    ],
)
def test_default_target_names_a_function_that_exists(
    label: str, target: str, landed_when: str
) -> None:
    """Regression: ``ASHEN_BASELINE_TARGET`` pointed at ``src.retrieval.baseline``.

    Person A's module is ``baseline_rag.py``. The target named ``baseline``, so
    :func:`pipeline._load` raised ``ModuleNotFoundError``, swallowed it exactly
    as designed, and the demo's side-by-side comparison served a labelled stub
    forever. Nothing failed. ``/health`` was not even wrong -- it does not report
    the baseline at all.

    This is the third instance of one failure mode and the reason the check is
    now parametrised over *every* target rather than written once for the one
    that broke.
    """
    if not (REPO_ROOT / landed_when).exists():
        pytest.skip(f"{landed_when} has not landed in this checkout")

    module_name, _, attribute = target.partition(":")
    path = _target_source_path(target)
    assert path is not None, (
        f"{label}: {module_name!r} does not resolve to a file, but {landed_when} exists. "
        "The target is misspelled and _load() will silently serve the stub."
    )
    assert _defines(path, attribute), f"{label}: {path} defines no {attribute!r}"


# ---------------------------------------------------------------------------
# The loop must not be handed fixture chunks while /health says "real"
# ---------------------------------------------------------------------------

class _FakeState:
    def __init__(self) -> None:
        self.trace: list[Any] = []
        self.unresolved_claims: list[str] = []
        self.route = "simple"


def test_real_answer_injects_the_real_search_into_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``research()`` defaults ``search_fn`` to the fixture search.

    ``_real_answer`` called ``research(question)`` and inherited that default, so
    the genuine agent loop would have run over ``fixtures/fake_chunks.json`` --
    real router, real sufficiency checks, real conflict detection, all of it
    reasoning about invented chunks -- while ``/health`` reported
    ``pipeline: "real"`` and every test passed.
    """
    real_search = lambda query, k=8: [{"chunk_id": "real_001", "text": "from the archive"}]
    monkeypatch.setattr(pipeline, "_load_retrieval", lambda: real_search)

    captured: dict[str, Any] = {}

    def research(question: str, search_fn: Any = "THE FIXTURE DEFAULT", max_iter: int = 5):
        captured["search_fn"] = search_fn
        return _FakeState()

    pipeline._real_answer(
        "q", research, lambda state, **kw: {"answer": "a", "citations": [], "conflicts": []}
    )
    assert captured["search_fn"] is real_search, "the loop inherited its fixture default"


def test_real_answer_falls_back_without_pretending(monkeypatch: pytest.MonkeyPatch) -> None:
    """No retrieval is a legitimate state. Claiming it is real is not.

    On a machine with no ``VOYAGE_API_KEY`` the loop still runs, on fixtures.
    That is fine and useful. What must not happen is ``/health`` calling it real.
    """
    monkeypatch.setattr(pipeline, "_load_retrieval", lambda: None)

    captured: dict[str, Any] = {}

    def research(question: str, search_fn: Any = "THE FIXTURE DEFAULT", max_iter: int = 5):
        captured["search_fn"] = search_fn
        return _FakeState()

    pipeline._real_answer(
        "q", research, lambda state, **kw: {"answer": "a", "citations": [], "conflicts": []}
    )
    assert captured["search_fn"] == "THE FIXTURE DEFAULT"

    monkeypatch.setenv("ASHEN_PIPELINE", "auto")
    monkeypatch.setattr(pipeline, "_load_real_pipeline", lambda: (research, lambda s: s))
    status = pipeline.pipeline_status()
    assert status["retrieval"] == "fixture"
    assert status["retrieval_available"] is False


def test_status_reports_retrieval_separately_from_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The state that reads green and is not: real loop, fixture chunks."""
    monkeypatch.setenv("ASHEN_PIPELINE", "auto")
    research = lambda question, search_fn=None, max_iter=5: _FakeState()
    monkeypatch.setattr(pipeline, "_load_real_pipeline", lambda: (research, lambda s: s))
    monkeypatch.setattr(pipeline, "_load_retrieval", lambda: None)

    status = pipeline.pipeline_status()
    assert status["pipeline"] == "real"
    assert status["retrieval"] == "fixture", (
        "pipeline and retrieval must be able to disagree -- collapsing them is "
        "how a demo ends up searching canned chunks under a green banner"
    )


def test_real_answer_passes_nothing_the_loop_does_not_declare() -> None:
    """The probe is what keeps this file from needing a coordinated edit.

    A loop that takes only ``question`` must still be callable. This is what lets
    Person B add ``on_step`` whenever they get to it without anyone touching the
    seam.
    """
    def minimal_research(question: str):
        return _FakeState()

    payload = pipeline._real_answer(
        "q",
        minimal_research,
        lambda state, **kw: {"answer": "a", "citations": [], "conflicts": []},
        on_step=lambda step: None,
    )
    assert payload["is_stub"] is False


# ---------------------------------------------------------------------------
# The baseline contract is a plain string
# ---------------------------------------------------------------------------

def test_baseline_accepts_a_plain_string_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ``dict(_as_dict(result))`` on a ``str`` raises ValueError.

    Person A's contract is ``baseline_rag(question) -> str``. The seam assumed a
    dict. This never fired only because the *target was also wrong*, so the real
    baseline was never reached -- two bugs stacked such that the first hid the
    second, and fixing only the target would have turned a silent stub into a
    500 on the demo's comparison toggle.
    """
    monkeypatch.setattr(
        pipeline, "_load", lambda env, default: (lambda q: "A one-shot answer, no citations.")
    )
    payload = pipeline._baseline_answer("who forged it?")

    assert payload["answer"] == "A one-shot answer, no citations."
    assert payload["citations"] == [], "the baseline tracks no citations, by design"
    assert payload["iterations_used"] == 1
    assert payload["trace"][0]["verdict"] == "baseline_no_check"
    assert payload["is_stub"] is False


def test_baseline_still_accepts_a_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the baseline later grows citations, the seam must not need an edit."""
    monkeypatch.setattr(
        pipeline,
        "_load",
        lambda env, default: (lambda q: {"answer": "a", "citations": [{"claim": "c",
                                                                      "filename": "f.txt"}]}),
    )
    payload = pipeline._baseline_answer("q")
    assert payload["citations"][0]["filename"] == "f.txt"


def test_baseline_failure_degrades_instead_of_500ing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The baseline needs a live key and a built index. A demo laptop may lack both.

    The comparison toggle exists to make a point in the video. It should never be
    the thing that breaks a live run, so a raising baseline shows a labelled stub
    with the reason attached rather than a traceback.
    """
    def explode(question: str) -> str:
        raise RuntimeError("VOYAGE_API_KEY is not set in .env")

    monkeypatch.setattr(pipeline, "_load", lambda env, default: explode)
    payload = pipeline._baseline_answer("q")
    assert "VOYAGE_API_KEY" in payload["answer"]
    assert payload["trace"][0]["verdict"] == "baseline_no_check"


# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------

# Import name -> distribution name, where they differ.
_DISTRIBUTION_NAMES = {
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "PIL": "pillow",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "rank_bm25": "rank-bm25",
    "multipart": "python-multipart",
}


def _third_party_imports(paths: list[Path]) -> set[str]:
    """Top-level packages our own code imports, excluding stdlib and first-party."""
    # Every top-level package this repo defines itself. Kept as an explicit list
    # rather than inferred, so a teammate adding a package has to add it here --
    # which is a one-line change, and the alternative is a guard that silently
    # stops guarding the moment the layout grows.
    first_party = {"src", "agent", "fixtures", "tests", "api", "ui", "retrieval",
                   "ingestion", "answer", "evaluation", "scripts"}
    found: set[str] = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {
        name
        for name in found
        if name not in first_party
        and name not in sys.stdlib_module_names
        and not name.startswith("_")
    }


def test_requirements_covers_every_import() -> None:
    """Regression: the main merge regenerated requirements.txt from one venv.

    That venv had Person A's stack and not Person D's, so ``fastapi``,
    ``streamlit``, ``diskcache`` and ``pytest`` were dropped while ``uvicorn``
    survived -- a file that installs cleanly on a fresh clone and then fails at
    ``uvicorn src.api.main:app`` with ``ModuleNotFoundError: No module named
    'fastapi'``.

    Nothing caught it because every existing test ran in an already-provisioned
    venv. A test suite cannot notice a missing dependency it is currently
    importing, so this walks the *source* instead of the environment.
    """
    sources = sorted(
        p
        for directory in ("src", "tests", "fixtures", "scripts")
        for p in (REPO_ROOT / directory).rglob("*.py")
    )
    assert sources, "no source files found -- the walk is broken, not the requirements"

    pinned = {
        re.split(r"[=<>!~\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    missing = sorted(
        name
        for name in _third_party_imports(sources)
        if _DISTRIBUTION_NAMES.get(name, name).lower().replace("_", "-") not in pinned
    )
    assert not missing, (
        f"imported but not pinned in requirements.txt: {missing}. "
        "A fresh clone will install cleanly and then fail at runtime."
    )


# ---------------------------------------------------------------------------
# Fixture citations must name real archive files
# ---------------------------------------------------------------------------

def _archive_root() -> Path | None:
    """The archive, if this machine has it. Same resolution order as the API."""
    configured = os.getenv("ASHEN_ARCHIVE_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates += [REPO_ROOT.parent / "Ashen_Era_Archive",
                   REPO_ROOT.parent.parent / "Ashen_Era_Archive"]
    return next((c for c in candidates if c.exists()), None)


def test_every_fixture_citation_names_a_real_archive_file() -> None:
    """Regression: the demo's "Open source file" button 404'd on every click.

    The fixtures were written on day one from the spec PDF, before anyone had
    seen the corpus, so they cited invented filenames -- letter_gravemaw_sighting.txt,
    annals_of_the_ashen_era_vol1.pdf. Nothing failed: the fixtures are internally
    consistent, every citation rendered, every marker linkified, and 113 tests
    passed. The button just quietly reported that the file was not in the archive,
    which was true.

    Worse than the dead button, the invented *claims* contradicted the archive.
    The fixture had Isolde Mournvale as a sworn member of the Ashen Order; the
    real wiki has her as a falconer in The Silent Choir. A judge who clicked
    through would have caught the demo citing a source that says something else.
    """
    archive = _archive_root()
    if archive is None:
        pytest.skip("archive not present on this machine")

    fixtures = json.loads(
        (REPO_ROOT / "fixtures" / "fake_api_response.json").read_text(encoding="utf-8")
    )
    missing = []
    for name, case in fixtures.items():
        if not isinstance(case, dict):
            continue
        for citation in case.get("citations", []):
            filename = citation["filename"]
            if not any(p.is_file() for p in archive.rglob(filename)):
                missing.append(f"{name}: {filename}")

    assert not missing, (
        "fixture citations naming files that do not exist in the archive: "
        f"{missing}. The 'Open source file' button will 404 on these."
    )

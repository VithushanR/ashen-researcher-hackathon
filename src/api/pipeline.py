"""The integration seam: fixtures now, the real pipeline the moment B and C land.

Everything in the API and the UI goes through this module. Nothing else imports
Person B's or Person C's code directly. That single rule is what lets the whole
interface exist before either of their components does, and it means integration
day touches this file and nothing else.

Three modes, selected by ``ASHEN_PIPELINE``:

===========  =============================================================
``auto``     default. Real pipeline if importable, fixture stub otherwise.
``real``     real only. Fails loudly if a component is missing -- use in CI,
             so a green build can never mean "the stub answered".
``stub``     fixtures even when the real pipeline exists. For UI work, and
             for rehearsing the demo on a machine without the archive.
===========  =============================================================

Where the real functions live is configurable, so a teammate moving or renaming
their entry point is a config change rather than a code change::

    ASHEN_RESEARCH_TARGET=agent.loop:research
    ASHEN_COMPOSE_TARGET=src.answer.composer:compose_answer

Person B's loop has landed on main, so ``agent.loop:research`` is now a real
module rather than a guess. Person C's composer has not, so ``auto`` still
serves the stub -- correctly, and ``/health`` says so.

Note the asymmetry in those two module paths. It is not a typo. Person B's
modules import each other as ``from agent.state import ...``, so they only
resolve with ``src/`` itself on ``sys.path``, and they must be addressed as
``agent.loop``. Person C's modules use relative imports, so they resolve as
``src.answer.composer`` from the repo root. :func:`_ensure_import_paths` puts
both roots on the path so either style works at runtime.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .schemas import normalise_response
from src.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "fake_api_response.json"

# Where teammates' code lives. Overridable per the note above.
#
# research() is addressed as "agent.loop", not "src.agent.loop", on purpose.
# Both spellings can be made to import, but they produce two *different* module
# objects for every agent module -- so agent.state.ResearchState and
# src.agent.state.ResearchState become distinct classes that fail isinstance
# against each other. Person B's own tests import `agent.loop`, so that is the
# identity we match.
DEFAULT_RESEARCH_TARGET = "agent.loop:research"
DEFAULT_COMPOSE_TARGET = "src.answer.composer:compose_answer"
DEFAULT_BASELINE_TARGET = "src.retrieval.baseline:baseline_rag"

# Repo root and src/. Both are needed to import Person B's agent package: the
# modules import each other as `agent.*` (needs src/) and pull their fake
# search from `fixtures.*` (needs the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"

# Seconds between stub trace steps. Fast enough not to bore a judge, slow enough
# that the trace panel visibly fills in round by round instead of appearing whole.
STUB_STEP_DELAY = float(os.getenv("ASHEN_STUB_DELAY", "0.8"))


class PipelineUnavailable(RuntimeError):
    """ASHEN_PIPELINE=real was requested but a real component is missing."""


def _mode() -> str:
    return os.getenv("ASHEN_PIPELINE", "auto").lower().strip()


# ---------------------------------------------------------------------------
# Locating the real pipeline
# ---------------------------------------------------------------------------

def _ensure_import_paths() -> None:
    """Put the repo root and ``src/`` on ``sys.path`` before importing teammates.

    This exists because of a bug that would have been invisible until the demo.

    Person B's agent modules import each other as ``from agent.state import ...``
    rather than ``from src.agent.state import ...``. That resolves under pytest,
    because ``pytest.ini`` sets ``pythonpath = src .`` -- but ``pytest.ini`` only
    configures the *test* process. Under ``uvicorn src.api.main:app`` from the
    repo root, ``src/`` is not on ``sys.path``, so importing the loop raises
    ``ModuleNotFoundError: No module named 'agent'``.

    :func:`_load` swallows ImportError by design, so that failure would not have
    crashed anything or logged anything at INFO. It would simply have served the
    fixture stub, forever, while ``/health`` truthfully reported ``stub`` and
    everyone assumed the real agent was wired. A green test suite and a working
    demo, both fake. Four lines here are what make ``auto`` mean what it says.

    Idempotent, and prepends rather than appends so our ``agent`` package wins
    over any similarly-named installed distribution.
    """
    for root in (_REPO_ROOT, _SRC_ROOT):
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _load(target_env: str, default: str) -> Callable | None:
    """Import ``module:function`` and return it, or None if it is not there yet.

    Every import happens *inside* this function rather than at module scope, and
    every failure is swallowed. That is deliberate. If a teammate's branch is
    half-merged and their module raises on import -- a syntax error, a missing
    dependency, a circular import -- a module-level import here would propagate
    and the entire API would refuse to start. Nobody could demo anything.

    Catching broadly degrades that to "the stub answers this request", which is
    the difference between a working demo and a dead one. The reason is logged
    at debug level so the cause is still findable.
    """
    target = os.getenv(target_env, default)
    module_name, _, attribute = target.partition(":")
    if not attribute:
        logger.warning("%s=%r is malformed; expected 'module:function'", target_env, target)
        return None

    _ensure_import_paths()
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except ImportError as error:
        # The ordinary case for most of the hackathon: not written yet.
        logger.debug("%s not importable yet: %s", target, error)
    except AttributeError:
        logger.warning("%s imported but has no attribute %r", module_name, attribute)
    except Exception as error:  # noqa: BLE001 - a broken teammate module must not kill the server
        logger.warning("%s raised on import, falling back to stub: %s", target, error)
    return None


def _load_real_pipeline() -> tuple[Callable | None, Callable | None]:
    """Person B's ``research`` and Person C's ``compose_answer``, if they exist."""
    return (
        _load("ASHEN_RESEARCH_TARGET", DEFAULT_RESEARCH_TARGET),
        _load("ASHEN_COMPOSE_TARGET", DEFAULT_COMPOSE_TARGET),
    )


def _supports_callback(research: Callable | None) -> bool:
    """Does Person B's ``research()`` accept an ``on_step`` callback?

    If it does, the UI trace is genuinely live: each round appears the instant
    the agent finishes it. If it does not, we run the loop and replay the
    finished trace afterwards -- which looks similar but is not the same thing.

    That difference is reported in /health as "live" or "replayed" rather than
    papered over. Claiming a live trace we do not have is exactly the sort of
    thing a judge will ask about, and being straight about it costs nothing.

    As merged, Person B's signature is ``research(question, search_fn, max_iter)``
    -- no ``on_step`` -- so this returns False and the real path reports
    "replayed". The request for the callback is outstanding; it is two lines in
    loop.py (call ``on_step(step)`` after appending each TraceStep) and nothing
    here needs to change when it lands, which is why the check is a signature
    probe rather than a hardcoded flag.
    """
    if research is None:
        return False
    try:
        return "on_step" in inspect.signature(research).parameters
    except (TypeError, ValueError):  # builtins and C functions have no signature
        return False


def _search_fn_is_real() -> bool:
    """Guard against the bug that hid behind a green ``/health`` before: an
    import succeeding is not proof the *right* thing was imported.

    ``_real_answer`` now always passes ``search_fn=hybrid_search`` explicitly
    (see below), so this checks that the ``hybrid_search`` this module bound
    at load time really does resolve to Person A's real
    ``src.retrieval.hybrid_search.hybrid_search`` and not something that
    quietly shadowed it -- e.g. ``fixtures.fake_hybrid_search`` re-exported
    under the same name.
    """
    return (
        callable(hybrid_search)
        and getattr(hybrid_search, "__module__", "") == "src.retrieval.hybrid_search"
    )


def pipeline_status() -> dict[str, Any]:
    """What is actually wired right now. Surfaced by /health and the UI sidebar."""
    research, compose = _load_real_pipeline()
    mode = _mode()
    search_wired = _search_fn_is_real()
    using_real = (
        mode != "stub" and research is not None and compose is not None and search_wired
    )
    return {
        "mode": mode,
        "pipeline": "real" if using_real else "stub",
        "research_available": research is not None,
        "compose_available": compose is not None,
        # True only if research() will actually be called against the real
        # archive search, not the fixtures/ stand-in. "real" above already
        # folds this in, but it is reported separately so a false "stub" can
        # be traced to the search wiring specifically rather than guessed at.
        "search_wired": search_wired,
        "streaming": "live" if using_real and _supports_callback(research) else "replayed",
    }


# ---------------------------------------------------------------------------
# The stub path
# ---------------------------------------------------------------------------

def _load_fixtures() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return [
        data["clean_answer_example"],
        data["conflict_answer_example"],
        data["partial_answer_example"],
    ]


def _stub_answer(question: str) -> dict[str, Any]:
    """Pick a fixture case for this question.

    Keyword matching first, so asking the same question twice gives the same
    case and the demo can be rehearsed. Random otherwise, so the UI keeps being
    exercised against all three states rather than settling into the happy path.
    """
    examples = _load_fixtures()
    lowered = question.lower()

    if any(word in lowered for word in ("actually", "true", "forged", "conflict", "disputed")):
        chosen = examples[1]
    elif any(word in lowered for word in ("dominion", "gravemaw", "unknown", "whose")):
        chosen = examples[2]
    elif any(word in lowered for word in ("war", "organization", "organisation", "faction", "member")):
        chosen = examples[0]
    else:
        chosen = random.choice(examples)

    response = dict(chosen)
    response["question"] = question
    response["is_stub"] = True
    return response


# ---------------------------------------------------------------------------
# The real path
# ---------------------------------------------------------------------------

def _compose_with_adapters(compose: Callable, state: Any) -> Any:
    """Call Person C's composer with Person D's model adapters injected.

    ``compose_answer`` deliberately selects no model itself; it requires the
    caller to supply provider wiring. This is the matching half of that split.

    Imported here rather than at module scope so that a missing or misconfigured
    ``llm`` module only affects the real path, never the stub path.
    """
    from . import llm

    return compose(
        state,
        synthesize=llm.synthesize,
        validate_semantics=llm.validate_semantics,
        validate_coverage=llm.validate_coverage,
    )


# Both retries below share one problem: a strict structured-output validator
# (src/answer/coverage_validation.py, src/agent/sufficiency.py) occasionally
# rejects a real model response because ONE of that response's own JSON fields
# doesn't logically cohere with another -- a mapped claim missing from the
# answer text, a verdict field that doesn't match what the model's own
# per-requirement statuses imply, a status of "omitted" carrying an excerpt
# that only "answered" is allowed to carry. Confirmed as sampling variance,
# not a code bug, three separate times now, across two different model
# providers, each with its own distinct exact message -- an exhaustive
# property test (tests/test_agent_sufficiency.py::test_complete_consistency_
# matrix) and a targeted regression test (test_model_cannot_override_
# unresolved_conflict) both confirm the checks themselves are correct and
# deliberate, so the fix belongs in retry orchestration, not in loosening
# either validator. Matching on the exact message was already a whack-a-mole
# list after the second one; _originates_in scopes the retry to WHERE an
# error came from instead, so a fourth or fifth phrasing of the same kind of
# contradiction is covered without another string added here by hand.
def _originates_in(error: BaseException, *module_names: str) -> bool:
    """True if any frame in error's traceback belongs to one of these modules.

    Confirmed empirically (not assumed) that this still works when the error
    is a pydantic ValidationError raised from inside a @model_validator: the
    validator's own raise happens inside pydantic-core, written in Rust, and
    does NOT appear in the traceback -- but the frame for the actual call
    that triggered validation (e.g. PresentationCoverageVerdict.model_validate
    (...), inside coverage_validation.py) does, because that call is a normal
    Python frame in the chain from raise to catch. ValidationError is itself
    a ValueError subclass (confirmed: issubclass(pydantic.ValidationError,
    ValueError) is True), so callers here only need to catch ValueError.
    """
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_globals.get("__name__") in module_names:
            return True
        tb = tb.tb_next
    return False


class ComposeExhausted(RuntimeError):
    """All _MAX_COMPOSE_ATTEMPTS attempts hit the same self-consistency family.

    Distinct from a bare re-raise so the caller (_real_answer) can tell "the
    known, expected retry-exhaustion case" apart from any other exception
    _compose_with_retry lets through unretried (a real bug, CitationCoverage
    Error, a transport error) -- only this one gets degraded into a graceful
    research-only response instead of a raw 500. ``original`` is kept for
    logging; chaining with `from error` keeps the real traceback visible too.
    """

    def __init__(self, original: Exception):
        self.original = original
        super().__init__(str(original))


_MAX_COMPOSE_ATTEMPTS = 3


def _compose_with_retry(compose: Callable, state: Any, question: str) -> Any:
    """Retry a self-consistency failure from coverage_validation.py, up to 3 attempts.

    CitationCoverageError/PresentationCompletenessError are deliberately NOT
    included even though they also originate there: those mean the verdict
    itself says the answer is genuinely incomplete, not that the verdict's
    own fields contradict each other, and composer.py already gives that
    case its own smarter repair-with-feedback attempt (MAX_COMPOSITION_
    REPAIRS) before ever re-raising here. Blindly retrying past that would
    blur a deliberate, already-handled distinction rather than fix variance.

    Any other exception -- those two types, a real bug in composer.py's own
    checks, a 429/quota error, anything not from coverage_validation.py at
    all -- propagates on the first attempt. Retrying those would burn quota
    for no benefit and could mask a genuine failure.
    """
    from src.answer.coverage_validation import CitationCoverageError

    for attempt in range(1, _MAX_COMPOSE_ATTEMPTS + 1):
        try:
            result = _compose_with_adapters(compose, state)
        except ValueError as error:
            if isinstance(error, CitationCoverageError) or not _originates_in(
                error, "src.answer.coverage_validation"
            ):
                raise
            if attempt == _MAX_COMPOSE_ATTEMPTS:
                logger.error(
                    "Retry attempt %d/%d exhausted after: %s", attempt, _MAX_COMPOSE_ATTEMPTS, error,
                )
                raise ComposeExhausted(error) from error
            logger.warning(
                "Retry attempt %d/%d after: %s", attempt, _MAX_COMPOSE_ATTEMPTS, error,
            )
            continue
        logger.info(
            "Coverage self-consistency check: attempt %d/%d passed for %r",
            attempt, _MAX_COMPOSE_ATTEMPTS, question,
        )
        return result


# Unlike the compose retry, a retry here re-runs Person B's ENTIRE research()
# loop from scratch (up to max_iter rounds, each with several LLM calls), not
# one cheap call -- so this stays capped at one retry (2 attempts total), not
# 3, to bound the extra quota cost of a mistake this expensive to redo.
_MAX_RESEARCH_ATTEMPTS = 2


def _research_with_retry(research: Callable, question: str) -> Any:
    """Retry a self-consistency failure from agent/sufficiency.py, up to 2 attempts.

    Any other exception -- a real bug, a 429/quota error, anything not from
    sufficiency.py at all (a malformed-output error from conflict.py's
    detector, say) -- propagates on the first attempt, exactly as in
    _compose_with_retry: retrying those would burn quota for no benefit.
    """
    for attempt in range(1, _MAX_RESEARCH_ATTEMPTS + 1):
        try:
            state = research(question, search_fn=hybrid_search)
        except ValueError as error:
            if not _originates_in(error, "agent.sufficiency"):
                raise
            if attempt == _MAX_RESEARCH_ATTEMPTS:
                logger.error(
                    "Retry attempt %d/%d exhausted after: %s", attempt, _MAX_RESEARCH_ATTEMPTS, error,
                )
                raise
            logger.warning(
                "Retry attempt %d/%d after: %s", attempt, _MAX_RESEARCH_ATTEMPTS, error,
            )
            continue
        logger.info(
            "Sufficiency self-consistency check: attempt %d/%d passed for %r",
            attempt, _MAX_RESEARCH_ATTEMPTS, question,
        )
        return state


def _field(state: Any, name: str, default: Any) -> Any:
    """Read a field from a Pydantic state or a plain dict, whichever we get."""
    value = state.get(name) if isinstance(state, dict) else getattr(state, name, None)
    return default if value is None else value


def _as_dict(item: Any) -> Any:
    return item.model_dump() if hasattr(item, "model_dump") else item


def _to_payload(state: Any, composed: Any, question: str) -> dict[str, Any]:
    """Merge Person B's state and Person C's answer into one wire response.

    ``ComposedAnswer`` carries the answer, citations and conflicts. The trace and
    the unresolved claims live on the ``ResearchState``. The UI needs both, so
    the API is where the two halves are joined -- neither teammate produces the
    whole payload alone.
    """
    payload = dict(_as_dict(composed))
    payload.setdefault("question", question)
    payload["trace"] = [_as_dict(step) for step in _field(state, "trace", [])]
    payload.setdefault("unresolved_claims", list(_field(state, "unresolved_claims", [])))
    payload.setdefault("route", _field(state, "route", None))
    payload["is_stub"] = False
    return payload


_COMPOSITION_FAILED_MESSAGE = (
    "Research completed successfully, but the answer composer could not "
    "produce a fully validated response after 3 attempts. This is a known "
    "model-consistency limitation, not a system failure."
)


def _degraded_payload(state: Any, question: str, error: Exception) -> dict[str, Any]:
    """Research succeeded even though composition never did -- surface that
    honestly (trace, evidence, iterations, route all still attached) instead
    of discarding real work behind a bare 500. status="composition_failed" is
    intentionally distinct from the three real ComposedAnswer statuses so the
    UI (src/ui/render.py's status_badge, src/ui/app.py's render_answer) can
    render it calmly rather than as either a normal answer or a raw error.
    """
    logger.warning(
        "Composition exhausted all retries for %r; returning a degraded "
        "research-only response instead of a 500: %s", question, error,
    )
    return {
        "question": question,
        "answer": _COMPOSITION_FAILED_MESSAGE,
        "status": "composition_failed",
        "confidence": _field(state, "confidence", 0),
        "citations": [],
        "conflicts": [],
        "unresolved_claims": list(_field(state, "unresolved_claims", [])),
        "iterations_used": _field(state, "iteration", 0),
        "route": _field(state, "route", None),
        "trace": [_as_dict(step) for step in _field(state, "trace", [])],
        "evidence": [_as_dict(item) for item in _field(state, "evidence", [])],
        "is_stub": False,
    }


def _real_answer(
    question: str,
    research: Callable,
    compose: Callable,
    on_step: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Run the real pipeline: Person B's loop, then Person C's composer."""
    started = time.monotonic()
    if on_step is not None and _supports_callback(research):
        # A retry here would need to somehow un-emit or relabel the steps
        # already streamed to the UI for the discarded attempt -- a real UX
        # problem, not just orchestration, and research() doesn't support
        # on_step yet in this deployment (see _supports_callback's docstring)
        # so this path is dormant. Left un-retried rather than solved blind.
        state = research(question, search_fn=hybrid_search, on_step=on_step)
    else:
        state = _research_with_retry(research, question)
    try:
        composed = _compose_with_retry(compose, state, question)
    except ComposeExhausted as exhausted:
        # Only the specific, known, already-retried exhaustion case degrades
        # gracefully. Anything else _compose_with_retry lets through --
        # CitationCoverageError, a real bug, a transport error -- still
        # propagates unchanged and still becomes a real 500: those are not
        # "a known model-consistency limitation," and claiming they are would
        # be its own dishonesty.
        return _degraded_payload(state, question, exhausted.original)
    payload = _to_payload(state, composed, question)
    # The one shared choke point for every real-pipeline response -- the
    # non-streaming /ask path and both streaming workers below all call this
    # function, so logging here covers all three without duplicating it.
    logger.info(
        "Answer ready: status=%s, confidence=%s, iterations=%s, took %.1fs",
        payload.get("status"), payload.get("confidence"),
        payload.get("iterations_used"), time.monotonic() - started,
    )
    return payload


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def answer_question(question: str, *, baseline: bool = False) -> dict[str, Any]:
    """Answer one question. The non-streaming path behind ``POST /ask``."""
    logger.info("Received question: %r", question)
    if baseline:
        return _baseline_answer(question)

    mode = _mode()
    if mode == "stub":
        return normalise_response(_stub_answer(question))

    research, compose = _load_real_pipeline()
    if research is None or compose is None:
        if mode == "real":
            missing = [
                name
                for name, func in (("research()", research), ("compose_answer()", compose))
                if func is None
            ]
            raise PipelineUnavailable(
                f"ASHEN_PIPELINE=real but {' and '.join(missing)} not importable"
            )
        logger.info("Real pipeline not ready - serving fixture stub")
        return normalise_response(_stub_answer(question))

    return normalise_response(_real_answer(question, research, compose))


def _baseline_answer(question: str) -> dict[str, Any]:
    """Person A's one-shot top-k RAG, for the side-by-side demo contrast.

    Degrades to a labelled stub rather than raising. The baseline toggle exists
    to make a point in the demo video; it should never be the thing that breaks
    a live run.
    """
    # The baseline runs one retrieval and answers. It performs no sufficiency
    # check at all, so its single round is always labelled baseline_no_check --
    # never "insufficient", which would imply a judgement it never made. Shared
    # between the real and degraded paths so both tell the same truth.
    trace = [
        {
            "step": 1,
            "query": question,
            "found": "Single top-k retrieval, no loop.",
            "verdict": "baseline_no_check",
            "missing": None,
        }
    ]

    baseline_rag = _load("ASHEN_BASELINE_TARGET", DEFAULT_BASELINE_TARGET)
    if baseline_rag is None:
        stub = normalise_response(_stub_answer(question))
        stub["answer"] = "[Baseline RAG not wired yet - showing stub output.]\n\n" + stub["answer"]
        stub["trace"] = trace
        stub["iterations_used"] = 1
        return stub

    result = baseline_rag(question)
    payload = dict(_as_dict(result))
    payload.setdefault("question", question)
    payload.setdefault("status", "complete")
    payload.setdefault("confidence", 0)
    payload["iterations_used"] = 1
    payload["trace"] = trace
    payload["is_stub"] = False
    return normalise_response(payload)


def stream_question(question: str, *, baseline: bool = False) -> Iterator[dict[str, Any]]:
    """Yield events as the research happens. Drives the live trace panel.

    Event shapes::

        {"event": "start",  "question": ...}
        {"event": "step",   "step": {...}}       one per research round
        {"event": "answer", "payload": {...}}    the full AskResponse
        {"event": "error",  "message": ...}

    Always terminates with exactly one ``answer`` or one ``error``, so the UI
    never waits on a stream that has quietly stopped producing.
    """
    logger.info("Received question: %r", question)
    yield {"event": "start", "question": question}

    mode = _mode()
    research, compose = _load_real_pipeline()
    use_stub = baseline or mode == "stub" or research is None or compose is None

    if use_stub and mode == "real" and not baseline:
        yield {
            "event": "error",
            "message": "ASHEN_PIPELINE=real but the real pipeline is not importable",
        }
        return

    if use_stub:
        try:
            payload = normalise_response(
                _baseline_answer(question) if baseline else _stub_answer(question)
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("Stub failed")
            yield {"event": "error", "message": str(error)}
            return
        for step in payload.get("trace", []):
            time.sleep(STUB_STEP_DELAY)
            yield {"event": "step", "step": step}
        yield {"event": "answer", "payload": payload}
        return

    if _supports_callback(research):
        yield from _stream_live(question, research, compose)
        return

    # Person B's research() has no callback hook, so there is nothing to stream
    # while it runs. Run it, then replay the finished trace. Honest but not
    # live -- /health reports this as "replayed".
    yield from _stream_replayed(question, research, compose)


def _stream_live(question: str, research: Callable, compose: Callable) -> Iterator[dict[str, Any]]:
    """Emit each round the instant Person B's loop finishes it.

    The loop runs on a worker thread and pushes steps onto a queue; this
    generator drains the queue as the HTTP response streams. A thread rather
    than asyncio because Person B's loop is ordinary blocking Python -- forcing
    it to be async would push complexity into the component being judged, to
    solve a problem that belongs to the transport layer.

    The ``done`` sentinel is pushed in a ``finally``, so the generator cannot
    hang if the worker dies unexpectedly.
    """
    events: queue.Queue = queue.Queue()
    result: dict[str, Any] = {}

    def on_step(step: Any) -> None:
        events.put(("step", _as_dict(step)))

    def worker() -> None:
        try:
            result["payload"] = normalise_response(
                _real_answer(question, research, compose, on_step=on_step)
            )
        except Exception as error:  # noqa: BLE001
            logger.exception("Pipeline failed")
            result["error"] = str(error)
        finally:
            events.put(("done", None))

    thread = threading.Thread(target=worker, daemon=True, name="ashen-research")
    thread.start()

    while True:
        kind, value = events.get()
        if kind == "done":
            break
        yield {"event": kind, "step": value}

    thread.join(timeout=5)
    if "error" in result:
        yield {"event": "error", "message": result["error"]}
    elif "payload" in result:
        yield {"event": "answer", "payload": result["payload"]}
    else:  # worker died without setting either -- should not happen, but must not hang
        yield {"event": "error", "message": "Research thread ended without producing a result"}


# Comfortably below ASHEN_UI_TIMEOUT (300s default, src/ui/app.py) so a
# heartbeat always lands well before the client's socket read-timeout could
# fire from inactivity alone.
STREAM_HEARTBEAT_SECONDS = float(os.getenv("ASHEN_STREAM_HEARTBEAT", "15"))


def _stream_replayed(question: str, research: Callable, compose: Callable) -> Iterator[dict[str, Any]]:
    """Run the pipeline on a worker thread; emit a heartbeat while it works.

    In "replayed" mode (research() has no on_step callback -- see
    _supports_callback) the old code called _real_answer() directly in the
    generator, which blocks the whole request for the run's entire duration
    with the SSE connection sending zero bytes throughout. Now that a single
    real question can involve a retried research() and/or a retried compose()
    (_research_with_retry, _compose_with_retry above), that silent stretch can
    comfortably exceed the UI's requests.post(timeout=ASHEN_UI_TIMEOUT) --
    which for a streaming response is an inactivity timeout, not a total-
    duration one, so it fires on a long silence even though the backend is
    still legitimately working, not hung. This confirmed diagnosis (reproduced:
    a run needing one retry took long enough with no bytes sent to trip a
    300s client read-timeout) is why the fix is a heartbeat, not a bigger
    ASHEN_UI_TIMEOUT number -- raising the number only postpones the same
    failure at some new, still-guessable worst case; a heartbeat resets the
    inactivity clock regardless of how long the retry sequence actually runs.
    app.py's stream_answer() loop already ignores any event kind it doesn't
    recognise, so a "heartbeat" event is inert there without a UI change.

    Trace steps still only appear once the whole run finishes -- that part of
    the "replayed" contract is unchanged; this only keeps the wire alive while
    waiting for it.
    """
    events: queue.Queue = queue.Queue()
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["payload"] = normalise_response(_real_answer(question, research, compose))
        except Exception as error:  # noqa: BLE001 - a demo must never show a traceback
            logger.exception("Pipeline failed")
            result["error"] = str(error)
        finally:
            events.put(("done", None))

    thread = threading.Thread(target=worker, daemon=True, name="ashen-research")
    thread.start()

    while True:
        try:
            events.get(timeout=STREAM_HEARTBEAT_SECONDS)
        except queue.Empty:
            yield {"event": "heartbeat"}
            continue
        break  # only "done" is ever queued in this (non-callback) path

    thread.join(timeout=5)
    if "error" in result:
        yield {"event": "error", "message": result["error"]}
        return
    payload = result.get("payload")
    if payload is None:  # worker died without setting either -- should not happen
        yield {"event": "error", "message": "Research thread ended without producing a result"}
        return
    for step in payload.get("trace", []):
        yield {"event": "step", "step": step}
    yield {"event": "answer", "payload": payload}

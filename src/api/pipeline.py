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

Where the real functions live is configurable, because Persons B and C have not
committed theirs yet and the names may differ from our guess::

    ASHEN_RESEARCH_TARGET=src.agent.loop:research
    ASHEN_COMPOSE_TARGET=src.answer.composer:compose_answer

When their real names land, that is a config change, not a code change.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import queue
import random
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .schemas import normalise_response

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "fake_api_response.json"

# Our best guess at where teammates' code will live. Overridable per the note above.
DEFAULT_RESEARCH_TARGET = "src.agent.loop:research"
DEFAULT_COMPOSE_TARGET = "src.answer.composer:compose_answer"
DEFAULT_BASELINE_TARGET = "src.retrieval.baseline:baseline_rag"

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
    """
    if research is None:
        return False
    try:
        return "on_step" in inspect.signature(research).parameters
    except (TypeError, ValueError):  # builtins and C functions have no signature
        return False


def pipeline_status() -> dict[str, Any]:
    """What is actually wired right now. Surfaced by /health and the UI sidebar."""
    research, compose = _load_real_pipeline()
    mode = _mode()
    using_real = mode != "stub" and research is not None and compose is not None
    return {
        "mode": mode,
        "pipeline": "real" if using_real else "stub",
        "research_available": research is not None,
        "compose_available": compose is not None,
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


def _real_answer(
    question: str,
    research: Callable,
    compose: Callable,
    on_step: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Run the real pipeline: Person B's loop, then Person C's composer."""
    if on_step is not None and _supports_callback(research):
        state = research(question, on_step=on_step)
    else:
        state = research(question)
    return _to_payload(state, _compose_with_adapters(compose, state), question)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def answer_question(question: str, *, baseline: bool = False) -> dict[str, Any]:
    """Answer one question. The non-streaming path behind ``POST /ask``."""
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
    baseline_rag = _load("ASHEN_BASELINE_TARGET", DEFAULT_BASELINE_TARGET)
    if baseline_rag is None:
        stub = normalise_response(_stub_answer(question))
        stub["answer"] = "[Baseline RAG not wired yet - showing stub output.]\n\n" + stub["answer"]
        stub["trace"] = stub["trace"][:1]
        stub["iterations_used"] = 1
        return stub

    result = baseline_rag(question)
    payload = dict(_as_dict(result))
    payload.setdefault("question", question)
    payload.setdefault("status", "complete")
    payload.setdefault("confidence", 0)
    payload["iterations_used"] = 1
    payload["trace"] = [
        {
            "step": 1,
            "query": question,
            "found": "Single top-k retrieval, no loop.",
            "verdict": "baseline_no_check",
            "missing": None,
        }
    ]
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
    try:
        payload = normalise_response(_real_answer(question, research, compose))
    except Exception as error:  # noqa: BLE001 - a demo must never show a traceback
        logger.exception("Pipeline failed")
        yield {"event": "error", "message": str(error)}
        return
    for step in payload.get("trace", []):
        yield {"event": "step", "step": step}
    yield {"event": "answer", "payload": payload}


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

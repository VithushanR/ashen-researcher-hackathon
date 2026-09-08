"""Retry, backoff and caching for every outbound call in the pipeline.

Non-negotiable #4 and spec section 2.13: exponential backoff on every API call,
from day one, plus response caching. Free tiers return HTTP 429 unpredictably,
and a rate-limit error during the demo recording is an avoidable, self-inflicted
failure.

Two decorators, meant to be stacked in this order:

    @cached("answers")        # outer
    @resilient("openrouter")  # inner
    def call_model(prompt: str) -> str: ...

The order matters. In Python the decorator listed first is the outermost
wrapper, so ``cached`` sees the call before ``resilient`` does. A cache hit
therefore returns without ever entering the retry machinery -- no timers, no
attempt counting, no chance of a cached value being re-fetched. Reversing them
would cache the retry wrapper instead, which still works but pays the
bookkeeping cost on every hit.

Both decorators degrade rather than fail. If ``tenacity`` or ``diskcache`` is
not installed they become pass-throughs and log a warning, so a teammate who
cloned the repo mid-hackathon without a full install is never blocked.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Five attempts produces four waits: 1s, 2s, 4s, 8s -- the schedule the
# challenge document asks for by name. tenacity computes
# multiplier * 2 ** (attempt - 1), so multiplier=1 gives 1, 2, 4, 8 and the
# max=8 cap keeps a fifth wait from becoming 16.
MAX_ATTEMPTS = 5
BACKOFF_MULTIPLIER = 1.0
BACKOFF_MIN = 1.0
BACKOFF_MAX = 8.0

CACHE_DIR = Path(os.getenv("ASHEN_CACHE_DIR", ".cache/ashen"))

try:  # import-time capability probe
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

    _HAS_TENACITY = True
except ImportError:  # pragma: no cover - exercised only on an incomplete install
    _HAS_TENACITY = False

try:  # import-time capability probe
    import diskcache

    _HAS_DISKCACHE = True
except ImportError:  # pragma: no cover
    _HAS_DISKCACHE = False


class TransientAPIError(Exception):
    """A failure worth retrying: rate limit, timeout, 5xx.

    Provider errors should be wrapped in this before being raised, so the retry
    decision is explicit rather than guessed from a message.
    """


class _Miss:
    """Sentinel so a cached ``None`` is not mistaken for a cache miss."""


_MISS = _Miss()


def is_retryable(error: BaseException) -> bool:
    """Decide whether an exception is worth another attempt.

    This is the function that keeps a genuine bug from being retried five times.
    A ``KeyError`` from a typo, or a ``ValueError`` from a malformed prompt, is
    not going to succeed on attempt four -- retrying it just multiplies the wait
    before you see the real traceback, and burns quota doing it.

    Explicit ``TransientAPIError`` always retries, as do timeouts and connection
    failures. Beyond that we inspect the message for the markers every provider
    surfaces, because we call more than one provider (OpenRouter, Voyage) and
    they do not share an exception hierarchy. Message sniffing is a fallback,
    not the primary path: prefer raising ``TransientAPIError`` at the call site.
    """
    if isinstance(error, TransientAPIError):
        return True
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(error).__name__} {error}".lower()
    markers = (
        "429", "rate limit", "ratelimit", "too many requests",
        "timeout", "timed out", "temporarily", "try again",
        "500", "502", "503", "504", "overloaded", "unavailable", "connection",
    )
    return any(marker in text for marker in markers)


def _log_retry(retry_state: Any) -> None:
    """Log each backoff so a slow demo is explainable after the fact."""
    error = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "attempt %s failed (%s) - backing off %.0fs",
        retry_state.attempt_number,
        error,
        getattr(retry_state.next_action, "sleep", 0),
    )


def resilient(label: str = "api") -> Callable[[F], F]:
    """Retry a function with exponential backoff, but only on transient failures.

    ``retry_if_exception(is_retryable)`` is what enforces the "only rate limits,
    timeouts and 5xx" rule. Using ``retry_if_exception_type(Exception)`` here
    and re-raising inside the wrapper would *look* correct and silently retry
    every bug in the codebase.
    """

    def decorator(func: F) -> F:
        if not _HAS_TENACITY:
            logger.warning("tenacity not installed - %s runs without retry", label)
            return func

        @retry(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(
                multiplier=BACKOFF_MULTIPLIER, min=BACKOFF_MIN, max=BACKOFF_MAX
            ),
            retry=retry_if_exception(is_retryable),
            before_sleep=_log_retry,
            reraise=True,  # surface the provider's real error, not RetryError
        )
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return decorator


def _fingerprint(func_name: str, args: tuple, kwargs: dict) -> str:
    """Stable cache key for a call, falling back to repr for odd arguments."""
    try:
        payload = json.dumps([args, sorted(kwargs.items())], default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        payload = repr((args, kwargs))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{func_name}:{digest}"


def cached(namespace: str = "default", ttl: int | None = None) -> Callable[[F], F]:
    """Persist results to disk so we never pay twice for the same call.

    Protects the free quota and makes the demo fast: a pre-warmed cache turns a
    30-second research run into an instant one.

    Only successful results are stored. An exception is never cached -- a rate
    limit at 11pm must not become a permanent failure for that question.

    Set ``ASHEN_NO_CACHE=1`` to bypass reads and writes entirely. That is what
    the demo video needs for the "question nobody has tested" segment, so the
    run is provably live rather than replayed from disk.
    """

    def decorator(func: F) -> F:
        if not _HAS_DISKCACHE:
            logger.warning("diskcache not installed - %s runs uncached", namespace)
            return func

        store = diskcache.Cache(str(CACHE_DIR / namespace))

        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if os.getenv("ASHEN_NO_CACHE") == "1":
                return func(*args, **kwargs)
            key = _fingerprint(func.__name__, args, kwargs)
            hit = store.get(key, default=_MISS)
            if hit is not _MISS:
                logger.debug("cache hit %s/%s", namespace, key)
                return hit
            value = func(*args, **kwargs)
            store.set(key, value, expire=ttl)
            return value

        inner.cache_clear = store.clear  # type: ignore[attr-defined]
        inner.cache_store = store  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return decorator


def timed(label: str) -> Callable[[F], F]:
    """Log wall-clock duration, so latency in the report is measured not guessed."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.info("%s took %.2fs", label, time.perf_counter() - start)

        return inner  # type: ignore[return-value]

    return decorator


def cache_status() -> dict[str, Any]:
    """Report what the robustness layer actually has available. Used by /health."""
    return {
        "retry": "tenacity" if _HAS_TENACITY else "disabled (tenacity not installed)",
        "cache": "diskcache" if _HAS_DISKCACHE else "disabled (diskcache not installed)",
        "cache_dir": str(CACHE_DIR),
        "cache_bypassed": os.getenv("ASHEN_NO_CACHE") == "1",
        "max_attempts": MAX_ATTEMPTS,
        "backoff_seconds": [1, 2, 4, 8],
    }

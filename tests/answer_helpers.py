"""Explicit adapter doubles for composition tests that do not judge semantics."""


def supported_semantics(prompt):
    """Assume support for existing orchestration regressions only.

    Runtime semantic rejection tests inject their own verdicts and exercise the
    real validator. This helper is never imported by production code.
    """
    return {"support": "supported", "explicit_absence": "clear"}


def complete_coverage(prompt):
    """Assume coverage only in existing tests focused on other behavior."""
    return {"coverage": "complete"}

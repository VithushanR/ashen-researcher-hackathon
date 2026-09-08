"""
Route classification — derived, not predicted.

Originally this module pre-classified a question as "simple" or
"multihop" using keyword heuristics before running any search. That
approach was dropped: any fixed keyword list is guessed from a handful
of worked examples and won't generalize to the hidden judging question
set (different phrasing, different fantasy terms, same underlying
structure).

Instead, routing falls out of the loop's own behavior for free: if the
sufficiency checker says "sufficient" after iteration 1, the question
was simple. If it takes more rounds, it was multihop. This is more
robust because it works on any question by construction — it doesn't
depend on the question matching a known phrase — at the cost of always
running at least one sufficiency check, even for questions that turn
out to be simple. That tradeoff is intentional and documented in
docs/decisions.md.

This module now just derives the label after the fact, for the trace
and the UI, rather than gating anything before the loop runs.
"""


def derive_route(iterations_used: int) -> str:
    """
    Label a completed run as "simple" or "multihop" based on how many
    iterations it actually took.

    Called once, after the loop has finished — never before or during.
    """
    return "simple" if iterations_used == 1 else "multihop"

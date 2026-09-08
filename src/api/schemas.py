"""Wire contract for the /ask endpoint.

This is Person D's public shape: what the API returns and what the Streamlit UI
renders. It is a superset of Person C's ``ComposedAnswer`` (docs/contracts.md)
plus the two fields the UI needs that ``ComposedAnswer`` does not carry --
``trace`` and ``unresolved_claims`` -- both of which come from Person B's
``ResearchState``. The API is where those two halves get joined.

Validation is deliberately asymmetric:

* **Responses are permissive.** Persons B and C own the meaning of these fields.
  Every response model allows extra keys and unknown trace verdicts pass
  straight through, so a teammate adding a field does not break the UI. An
  unrecognised field is carried, not dropped.
* **Requests are strict.** ``AskRequest`` forbids extras, because the only
  caller is our own UI. A typo like ``baselin=True`` should fail loudly with a
  422 rather than being silently ignored and quietly disabling the baseline
  comparison during a demo.

Being lenient about what we accept from teammates and strict about what we send
ourselves is the split that keeps integration cheap without hiding our own bugs.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The statuses Person C's ComposedAnswer is allowed to emit (docs/contracts.md).
# Kept as a Literal on purpose: this one really is a closed set, and an unknown
# status means C's contract changed and we need to know immediately.
AnswerStatus = Literal["complete", "complete_with_conflict", "partial_gap_stated"]


class Citation(BaseModel):
    """One claim in the answer, tied to the source that supports it.

    ``claim``, ``filename``, ``page`` and ``source_type`` are the required
    contract fields. The rest are optional enrichments -- when Person C passes
    them through we show the source-tier badge, the originating chunk, and the
    actual supporting sentence inside the expandable citation. When they are
    absent the citation still renders, just with less detail.
    """

    model_config = ConfigDict(extra="allow")

    claim: str
    filename: str
    page: int | None = None
    source_type: str = "unknown"
    section: str | None = None
    reliability: str | None = None
    chunk_id: str | None = None
    text: str | None = None


class ClaimSource(BaseModel):
    """One side of a disagreement: a claim and where it came from."""

    model_config = ConfigDict(extra="allow")

    claim: str
    source: str
    chunk_id: str | None = None


class Conflict(BaseModel):
    """Two or more incompatible claims about the same attribute.

    Person B detects and resolves, Person C presents, Person D displays. The
    current contract is the multi-claim ``claims`` list; the older pairwise
    shape is converted by :func:`normalise_conflict` before it reaches here.

    ``resolution`` and ``resolved_value`` are optional because an unresolved
    conflict is a legitimate outcome -- the system is required to say "these
    sources disagree and we could not settle it" rather than pick one silently.
    """

    model_config = ConfigDict(extra="allow")

    attribute: str | None = None
    claims: list[ClaimSource] = Field(default_factory=list)
    resolution: str | None = None
    resolved_value: str | None = None


class TraceStep(BaseModel):
    """One round of the research loop, as rendered in the live trace panel.

    Person B's merged ``TraceStep`` is ``{step, query, verdict, missing}``, which
    this matches. ``found`` is kept because the fixtures carry it and the panel
    renders it when present; the loop does not emit it, so it is simply absent
    on a real run and the round renders without that line.

    Everything except ``step`` stays optional and extras are preserved anyway.
    Now that the shapes agree, that permissiveness is no longer about waiting on
    a contract -- it is so a field added mid-hackathon reaches the UI as an
    extra instead of being rejected at the door.
    """

    model_config = ConfigDict(extra="allow")

    step: int
    query: str | None = None
    found: str | None = None
    verdict: str | None = None
    missing: str | None = None


class AskRequest(BaseModel):
    """A question from the UI. Strict: our own bugs should not pass silently."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    # Routes to Person A's one-shot top-k RAG instead of the agent loop, for the
    # side-by-side contrast in the demo video (spec section 2.10).
    baseline: bool = False


class AskResponse(BaseModel):
    """The full payload the UI renders.

    Person C's ComposedAnswer supplies question, answer, status, confidence,
    citations, conflicts and iterations_used. Person B's ResearchState supplies
    trace, unresolved_claims and route.
    """

    model_config = ConfigDict(extra="allow")

    question: str
    answer: str
    status: AnswerStatus
    confidence: int = Field(ge=0, le=100)
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    unresolved_claims: list[str] = Field(default_factory=list)
    iterations_used: int = 0
    route: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    # True when the answer came from the fixture stub rather than the real
    # pipeline. The UI shows a banner for this, so a demo can never silently
    # pass off canned output as a real research run.
    is_stub: bool = False


class HealthResponse(BaseModel):
    """What is actually wired right now. Rendered in the UI sidebar."""

    model_config = ConfigDict(extra="allow")

    status: str
    pipeline: str
    research_available: bool
    compose_available: bool
    streaming: str


def normalise_conflict(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept either conflict shape and return the current multi-claim one.

    The fixtures PDF, and any teammate code written from it, uses four flat
    fields: ``claim_a`` / ``source_a`` / ``claim_b`` / ``source_b``.
    ``docs/contracts.md`` replaced that with a ``claims`` list, because the
    pairwise shape cannot represent a three-way disagreement and the archive is
    deliberately built to contain those.

    Converting on the way in means the UI only ever has one shape to render,
    and a teammate who has not migrated yet does not break the demo. Original
    keys are preserved rather than stripped -- if someone is still reading
    ``claim_a`` downstream, it is still there.
    """
    if raw.get("claims"):
        return raw

    converted = dict(raw)
    claims: list[dict[str, Any]] = []
    for suffix in ("a", "b", "c"):
        claim = raw.get(f"claim_{suffix}")
        if claim:
            claims.append({"claim": claim, "source": raw.get(f"source_{suffix}", "unknown")})
    converted["claims"] = claims
    return converted


def normalise_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Prepare an arbitrary upstream payload for :class:`AskResponse`.

    The single funnel every response passes through, whether it came from the
    fixture stub or the real pipeline. Right now it only normalises conflicts;
    it exists as a named seam so future shape drift has one obvious place to be
    absorbed instead of being patched into the UI.
    """
    normalised = dict(raw)
    normalised["conflicts"] = [normalise_conflict(c) for c in raw.get("conflicts") or []]
    return normalised

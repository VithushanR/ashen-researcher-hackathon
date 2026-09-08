"""
Contract 3 — The Research State object.

This is the single structure that flows through every step of the agent
loop. It is produced (in progress, then final) by Person B's research()
function, and consumed by Person C's compose_answer() and Person D's
Streamlit UI.

Design principle: nothing here should ever be inferred implicitly from
an LLM's conversational memory. If a fact matters to the loop's
decisions, it lives here, explicitly, as a field.
"""
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single piece of retrieved evidence, matching Contract 1's chunk shape."""

    chunk_id: str
    document_id: str | None = None
    filename: str
    source_type: str      # "codex" | "wiki" | "novel" | "ephemera" | "image_derived"
    reliability: str      # "T1_authoritative" | "T2_curated" | "T3_narrative" | "T4_unverified"
    page: int | None = None
    section: str | None = None
    content_type: str = "text"   # "text" | "table" | "image" | "vision_description"
    text: str
    entities: list[str] = Field(default_factory=list)


class ClaimSource(BaseModel):
    """One source's claim about a disputed attribute.

    Claim and source are bundled together deliberately, rather than kept
    as two parallel lists (claims=[...], sources=[...]) — parallel lists
    matched by index position are an easy way to silently desync a claim
    from its actual source if either list is ever reordered or filtered.
    """

    claim: str
    source: str            # human-readable: "<filename> (<source_type>, <reliability>)"
    chunk_id: str | None = None


class Conflict(BaseModel):
    """A detected disagreement between two or more evidence items on the same attribute."""

    attribute: str | None = None       # what's actually being disputed, e.g. "year forged"
    claims: list[ClaimSource]          # every competing claim, not just two
    resolution: str | None = None      # explanation of why one was preferred, filled in once resolved
    resolved_value: str | None = None  # the value the system will actually use in the final answer
    # NOTE: resolved_value may deliberately remain None if no justified
    # winner exists (e.g. two T1 sources, equal corroboration, equal
    # specificity) — see conflict.py resolution policy.


class TraceStep(BaseModel):
    """One logged decision point, for the UI's live trace panel and for debugging."""

    step: int
    query: str
    verdict: str           # "insufficient" | "sufficient" | "conflict_detected" | "capped_unresolved"
    missing: str | None = None


class ResearchState(BaseModel):
    """
    The object that flows through the whole loop. Person B produces this,
    fully populated, from research(question) -> ResearchState.

    NOTE: final_answer intentionally removed from this model — it now
    lives in Person C's ComposedAnswer object instead, since
    compose_answer() returns a structured object (answer, status,
    citations, conflicts, iterations_used), not a bare string.
    """

    question: str
    route: str = "unknown"                                   # "simple" | "multihop", derived AFTER the loop runs
    required_claims: list[str] = Field(default_factory=list)  # what must be proven to answer
    iteration: int = 0
    search_history: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    discovered_entities: list[str] = Field(default_factory=list)
    unresolved_claims: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    confidence: int = 0                                      # 0-100, derived from the grounded rubric, never self-rated
    trace: list[TraceStep] = Field(default_factory=list)

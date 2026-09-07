"""Deterministic presentation of upstream conflicts; no source ranking."""

from typing import TYPE_CHECKING
from collections.abc import Callable

from .models import ComposedAnswer
from .coverage_validation import validate_answer_coverage
from .synthesis import CitationClaim
from .semantic_validation import validate_citation_claim

if TYPE_CHECKING:
    from ..agent.state import Evidence, ResearchState


def compose_conflict_answer(
    state: "ResearchState", evidence_by_id: dict[str, "Evidence"],
    *, validate_coverage: Callable[[str], object] | None = None,
    validate_semantics: Callable[[str], object] | None = None,
) -> ComposedAnswer:
    """Preserve all claims and report only resolutions supplied by Person B.

    Missing claim chunk IDs retain source attribution without fabricating a
    citation. Supplied but unknown IDs fail, as in the clean synthesis path.
    No synthesis call is made. Coverage checks the final presentation against
    B's supplied decisions without selecting a winner from the passages.
    """
    paragraphs = []
    conflicts = []
    citations = []
    citation_claims = []
    citable_claims = []
    unresolved = False

    for conflict in state.conflicts:
        if len(conflict.claims) < 2:
            raise ValueError("A conflict must contain at least two competing claims")
        conflicts.append({
            "attribute": conflict.attribute,
            "claims": [
                {"claim": claim.claim, "source": claim.source, "chunk_id": claim.chunk_id}
                for claim in conflict.claims
            ],
            "resolution": conflict.resolution,
            "resolved_value": conflict.resolved_value,
        })
        label = "Source conflict"
        if conflict.attribute is not None:
            label += f" concerning {conflict.attribute}"
        lines = [label + ":"]
        for claim in conflict.claims:
            # Attribute the competing assertion instead of endorsing it as fact.
            attributed_claim = f'{claim.source} reports: "{claim.claim}"'
            lines.append(attributed_claim)
            if claim.chunk_id is None:
                lines.append("No evidence chunk reference was supplied for this claim.")
                continue
            if claim.chunk_id not in evidence_by_id:
                raise ValueError(f"Unknown conflict chunk_id: {claim.chunk_id}")
            citation_claims.append(CitationClaim(claim=attributed_claim, chunk_id=claim.chunk_id))
            citable_claims.append((claim, attributed_claim))

        if conflict.resolved_value is None:
            unresolved = True
            lines.append(
                "The evidence remains contradictory. No preferred result has been "
                "established for this conflict."
            )
        else:
            lines.append(f"The research result prefers: {conflict.resolved_value}.")
            if conflict.resolution is not None:
                lines.append(f"Supplied resolution: {conflict.resolution}")
        paragraphs.append("\n".join(lines))

    if state.unresolved_claims:
        paragraphs.append("Remaining research gaps:\n" + "\n".join(state.unresolved_claims))

    answer = "\n\n".join(paragraphs)
    validate_answer_coverage(
        answer, citation_claims, validate_coverage=validate_coverage,
        unresolved_claims=state.unresolved_claims, conflicts=conflicts,
    )
    if citable_claims and validate_semantics is None:
        raise ValueError("A semantic-validation adapter is required for conflict citations")
    for claim, attributed_claim in citable_claims:
        validate_citation_claim(
            claim.claim, claim.chunk_id, evidence_by_id,
            validate_semantics=validate_semantics, conflict_attribution=True,
        )
        evidence = evidence_by_id[claim.chunk_id]
        citations.append({
            "claim": attributed_claim,
            "filename": evidence.filename,
            "page": evidence.page,
            "section": evidence.section,
            "source_type": evidence.source_type,
        })
    return ComposedAnswer(
        question=state.question,
        answer=answer,
        status=("partial_gap_stated" if unresolved or state.unresolved_claims
                else "complete_with_conflict"),
        confidence=state.confidence,
        citations=citations,
        conflicts=conflicts,
        iterations_used=state.iteration,
    )

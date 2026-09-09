"""Deterministic presentation of upstream conflicts; no source ranking."""

from typing import TYPE_CHECKING
from dataclasses import dataclass


@dataclass
class ConflictCitation:
    raw_claim: str
    attributed_claim: str
    chunk_id: str


@dataclass
class ConflictPresentation:
    answer: str
    conflicts: list[dict]
    citations: list[ConflictCitation]
    has_unresolved: bool


if TYPE_CHECKING:
    from ..agent.state import Evidence, ResearchState


def build_conflict_presentation(
    state: "ResearchState", evidence_by_id: dict[str, "Evidence"],
) -> ConflictPresentation:
    """Build B's report and check references without synthesis or finalization."""
    paragraphs = []
    conflicts = []
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
            citable_claims.append(ConflictCitation(claim.claim, attributed_claim, claim.chunk_id))

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

    return ConflictPresentation("\n\n".join(paragraphs), conflicts, citable_claims, unresolved)

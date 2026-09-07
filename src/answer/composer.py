"""Clean-success composition using Person B's evidence and an injected LLM call."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from .models import ComposedAnswer
from .synthesis import SynthesisResult, build_synthesis_prompt

if TYPE_CHECKING:
    from ..agent.state import ResearchState


def compose_answer(
    state: "ResearchState", *, synthesize: Callable[[str], object]
) -> ComposedAnswer:
    """Compose a successfully finished, nonempty state without conflicts or gaps.

    The caller supplies the team's synthesis adapter, accepting a prompt and
    returning decoded JSON or SynthesisResult. Provider wiring is deliberately
    required; no model or network client is selected here. The caller guarantees
    research has finished successfully because the state has no completion flag.
    """
    if state.conflicts or state.unresolved_claims:
        raise ValueError("Only clean-success states without conflicts or gaps are supported")
    if not state.evidence:
        raise ValueError("Clean-success composition requires evidence")

    evidence_by_id = {}
    for evidence in state.evidence:
        if evidence.chunk_id in evidence_by_id:
            raise ValueError(f"Ambiguous duplicate evidence chunk_id: {evidence.chunk_id}")
        evidence_by_id[evidence.chunk_id] = evidence

    prompt = build_synthesis_prompt(
        state.question,
        [{"chunk_id": item.chunk_id, "text": item.text} for item in state.evidence],
    )
    synthesis = SynthesisResult.model_validate(synthesize(prompt))
    citations = []
    for reference in synthesis.citation_claims:
        if reference.chunk_id not in evidence_by_id:
            raise ValueError(f"Unknown synthesis chunk_id: {reference.chunk_id}")
        evidence = evidence_by_id[reference.chunk_id]
        citations.append({
            "claim": reference.claim,
            "filename": evidence.filename,
            "page": evidence.page,
            "section": evidence.section,
            "source_type": evidence.source_type,
        })

    return ComposedAnswer(
        question=state.question,
        answer=synthesis.answer,
        status="complete",
        confidence=state.confidence,
        citations=citations,
        conflicts=[],
        iterations_used=state.iteration,
    )

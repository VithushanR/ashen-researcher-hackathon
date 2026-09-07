"""Compose clean answers or faithfully present Person B's conflict decisions."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from .models import ComposedAnswer
from .conflicts import compose_conflict_answer
from .coverage_validation import validate_answer_coverage
from .semantic_validation import validate_citation_claim
from .synthesis import PartialSynthesisResult, SynthesisResult, build_synthesis_prompt

if TYPE_CHECKING:
    from ..agent.state import ResearchState


def compose_answer(
    state: "ResearchState", *, synthesize: Callable[[str], object],
    validate_semantics: Callable[[str], object] | None = None,
    validate_coverage: Callable[[str], object] | None = None,
) -> ComposedAnswer:
    """Compose a finished research state, retaining any authoritative gaps.

    The caller supplies the team's synthesis adapter, accepting a prompt and
    returning decoded JSON or SynthesisResult. Provider wiring is deliberately
    required; no model or network client is selected here. The caller guarantees
    research has finished because the state has no completion flag. Conflict
    states use deterministic presentation of Person B's decisions instead of
    synthesis. Non-conflict partial states synthesize only supported facts and
    append Person B's unresolved claims unchanged, without citations for gaps.
    Generated citation claims require an injected semantic validator. Missing
    wiring, rejected claims, and malformed verdicts fail before returning an answer.
    Every final answer also requires injected coverage validation, including
    answers with no generated claims and deterministic conflict/gap reports.
    """
    if not state.evidence:
        if state.unresolved_claims and not state.conflicts:
            answer = "No supporting evidence was retrieved.\n\n" + _format_gaps(state.unresolved_claims)
            validate_answer_coverage(
                answer, [], validate_coverage=validate_coverage,
                unresolved_claims=state.unresolved_claims, no_evidence_returned=True,
            )
            return ComposedAnswer(
                question=state.question,
                answer=answer,
                status="partial_gap_stated", confidence=state.confidence,
                citations=[], conflicts=[], iterations_used=state.iteration,
            )
        raise ValueError("Answer composition requires evidence")

    evidence_by_id = {}
    for evidence in state.evidence:
        if evidence.chunk_id in evidence_by_id:
            raise ValueError(f"Ambiguous duplicate evidence chunk_id: {evidence.chunk_id}")
        evidence_by_id[evidence.chunk_id] = evidence

    if state.conflicts:
        return compose_conflict_answer(state, evidence_by_id, validate_coverage=validate_coverage)

    prompt = build_synthesis_prompt(
        state.question,
        [{"chunk_id": item.chunk_id, "text": item.text} for item in state.evidence],
        unresolved_claims=state.unresolved_claims,
    )
    result_type = PartialSynthesisResult if state.unresolved_claims else SynthesisResult
    raw_result = synthesize(prompt)
    if state.unresolved_claims and isinstance(raw_result, SynthesisResult):
        raw_result = raw_result.model_dump()
    synthesis = result_type.model_validate(raw_result)
    answer = (synthesis.answer + "\n\n" + _format_gaps(state.unresolved_claims)
              if state.unresolved_claims else synthesis.answer)
    validate_answer_coverage(
        answer, synthesis.citation_claims, validate_coverage=validate_coverage,
        unresolved_claims=state.unresolved_claims,
    )
    # Resolve every ID before any semantic calls; never fabricate source metadata.
    for reference in synthesis.citation_claims:
        if reference.chunk_id not in evidence_by_id:
            raise ValueError(f"Unknown synthesis chunk_id: {reference.chunk_id}")
    if synthesis.citation_claims and validate_semantics is None:
        raise ValueError("A semantic-validation adapter is required for generated citation claims")
    citations = []
    for reference in synthesis.citation_claims:
        validate_citation_claim(
            reference.claim, reference.chunk_id, evidence_by_id,
            validate_semantics=validate_semantics,
        )
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
        answer=answer,
        status="partial_gap_stated" if state.unresolved_claims else "complete",
        confidence=state.confidence,
        citations=citations,
        conflicts=[],
        iterations_used=state.iteration,
    )


def _format_gaps(unresolved_claims: list[str]) -> str:
    """Keep B's gap descriptions even if synthesis omits or paraphrases them."""
    return "Unresolved information (not established by research):\n" + "\n".join(
        f"- {gap}" for gap in unresolved_claims
    )

"""Compose clean answers or faithfully present Person B's conflict decisions."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from .models import ComposedAnswer
from .conflicts import build_conflict_presentation
from .coverage_validation import PresentationCompletenessError, validate_answer_coverage
from .semantic_validation import validate_citation_claim
from .synthesis import CitationClaim, OrdinarySectionResult, PartialSynthesisResult, SynthesisResult, build_synthesis_prompt

if TYPE_CHECKING:
    from ..agent.state import ResearchState


MAX_COMPOSITION_REPAIRS = 1


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
    states synthesize independent facts alongside deterministic presentation of
    Person B's decisions. Non-conflict partial states synthesize only supported facts and
    append Person B's unresolved claims unchanged, without citations for gaps.
    Generated citation claims require an injected semantic validator. Missing
    wiring, rejected claims, and malformed verdicts fail before returning an answer.
    Every final answer also requires injected coverage validation, including
    answers with no generated claims and deterministic conflict/gap reports.
    A clear independent presentation omission permits one synthesis repair using
    the same research. Other failures propagate; no research is performed here.
    """
    if not state.evidence:
        if state.unresolved_claims and not state.conflicts:
            answer = "No supporting evidence was retrieved.\n\n" + _format_gaps(state.unresolved_claims)
            validate_answer_coverage(
                answer, [], validate_coverage=validate_coverage,
                unresolved_claims=state.unresolved_claims, no_evidence_returned=True,
                question=state.question, required_claims=state.required_claims, evidence=[],
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

    presentation = build_conflict_presentation(state, evidence_by_id) if state.conflicts else None
    conflicts = presentation.conflicts if presentation else []
    conflict_citations = presentation.citations if presentation else []

    evidence_context = [{"chunk_id": item.chunk_id, "text": item.text} for item in state.evidence]
    result_type = PartialSynthesisResult if state.unresolved_claims else SynthesisResult
    if presentation:
        result_type = OrdinarySectionResult
    repair_feedback = None
    for attempt in range(MAX_COMPOSITION_REPAIRS + 1):
        prompt = build_synthesis_prompt(
            state.question, evidence_context, required_claims=state.required_claims,
            unresolved_claims=state.unresolved_claims, conflicts=conflicts,
            repair_feedback=repair_feedback,
        )
        raw_result = synthesize(prompt)
        # Revalidate instances and subclasses against the schema selected by the state.
        if isinstance(raw_result, (SynthesisResult, OrdinarySectionResult)):
            raw_result = raw_result.model_dump()
        synthesis = result_type.model_validate(raw_result)
        sections = [synthesis.answer] if synthesis.answer is not None else []
        if presentation:
            sections.append(presentation.answer)
        if state.unresolved_claims:
            sections.append(_format_gaps(state.unresolved_claims))
        answer = "\n\n".join(sections)
        combined_claims = list(synthesis.citation_claims) + [
            CitationClaim(claim=item.attributed_claim, chunk_id=item.chunk_id)
            for item in conflict_citations
        ]
        try:
            validate_answer_coverage(
                answer, combined_claims, validate_coverage=validate_coverage,
                unresolved_claims=state.unresolved_claims, conflicts=conflicts,
                ordinary_section=synthesis if presentation else None,
                ordinary_answer=synthesis.answer,
                question=state.question, required_claims=state.required_claims,
                evidence=evidence_context,
            )
        except PresentationCompletenessError as error:
            # An unknown ID is never a reason to repair, including a discarded draft.
            _check_chunk_ids(synthesis.citation_claims, evidence_by_id)
            if not error.repairable or attempt == MAX_COMPOSITION_REPAIRS:
                raise
            repair_feedback = error.omissions
            continue
        _check_chunk_ids(synthesis.citation_claims, evidence_by_id)
        break
    if combined_claims and validate_semantics is None:
        raise ValueError("A semantic-validation adapter is required for generated citation claims")
    for reference in synthesis.citation_claims:
        validate_citation_claim(
            reference.claim, reference.chunk_id, evidence_by_id,
            validate_semantics=validate_semantics,
        )
    for reference in conflict_citations:
        validate_citation_claim(
            reference.raw_claim, reference.chunk_id, evidence_by_id,
            validate_semantics=validate_semantics, conflict_attribution=True,
        )
    citations = []
    for reference in combined_claims:
        evidence = evidence_by_id[reference.chunk_id]
        citations.append({
            "claim": reference.claim,
            "filename": evidence.filename,
            "page": evidence.page,
            "section": evidence.section,
            "source_type": evidence.source_type,
        })
    status = "complete_with_conflict" if presentation else "complete"
    if state.unresolved_claims or (presentation and presentation.has_unresolved):
        status = "partial_gap_stated"

    return ComposedAnswer(
        question=state.question,
        answer=answer,
        status=status,
        confidence=state.confidence,
        citations=citations,
        conflicts=conflicts,
        iterations_used=state.iteration,
    )


def _format_gaps(unresolved_claims: list[str]) -> str:
    """Keep B's gap descriptions even if synthesis omits or paraphrases them."""
    return "Unresolved information (not established by research):\n" + "\n".join(
        f"- {gap}" for gap in unresolved_claims
    )


def _check_chunk_ids(claims: list[CitationClaim], evidence_by_id: dict) -> None:
    """Resolve every ID before semantic calls or permission to repair."""
    for reference in claims:
        if reference.chunk_id not in evidence_by_id:
            raise ValueError(f"Unknown synthesis chunk_id: {reference.chunk_id}")

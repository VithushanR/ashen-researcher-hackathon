"""Deterministic scoring of outputs and recorded validation, without model calls."""

from pydantic import ValidationError

from src.answer.coverage_validation import CoverageVerdict
from src.answer.models import ComposedAnswer
from src.answer.semantic_validation import SemanticVerdict

from .models import EvaluationCase, EvaluationObservation, EvaluationResult, MetricResult


METRICS = (
    "output_contract", "question_match", "answer_status", "answer_content",
    "citation_structure", "citation_coverage", "groundedness", "explicit_absence",
    "conflict_handling", "unresolved_gaps",
    "visual_factual_accuracy", "visual_source_citation",
)


def _metric(passed: bool, *notes: str) -> MetricResult:
    return MetricResult(status="pass" if passed else "fail", notes=list(notes))


def _finish(case, metrics, errors):
    excluded = set()
    if case.visual_expectations is None:
        excluded.update(("visual_factual_accuracy", "visual_source_citation"))
    elif not any(f.expected_image_filenames for f in case.visual_expectations.facts):
        excluded.add("visual_source_citation")
    applicable = [metric for name, metric in metrics.items()
                  if name not in excluded and metric.status != "not_applicable"]
    successes = sum(metric.status == "pass" for metric in applicable)
    return EvaluationResult(
        case_id=case.id, data_origin=case.data_origin,
        passed=not errors and successes == len(applicable),
        score=successes / len(applicable) if applicable else 0.0,
        metrics=metrics, errors=errors,
        notes=["Text expectations are literal regression checks, not semantic accuracy judgments."]
        + ([case.notes] if case.notes else []),
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def evaluate_visual_expectations(expectations, answer: ComposedAnswer):
    """Literal reference checks only: no pixels, evidence text, or model calls."""
    if expectations is None:
        return {name: MetricResult(status="not_evaluated", notes=[
            "No human-verified visual rubric supplied; excluded from scoring."])
            for name in ("visual_factual_accuracy", "visual_source_citation")}
    text = _normalize(answer.answer)
    fact_errors, source_errors = [], []
    for fact in expectations.facts:
        accepted = {_normalize(statement) for statement in fact.accepted_statements}
        if not any(statement in text for statement in accepted):
            fact_errors.append(f"{fact.id}: missing accepted visual assertion.")
        for forbidden in fact.forbidden_statements:
            if _normalize(forbidden) in text:
                fact_errors.append(f"{fact.id}: forbidden assertion: {forbidden}")
        # Exact normalized claim association prevents unrelated citations from counting.
        filenames = {citation["filename"] for citation in answer.citations
                     if _normalize(citation["claim"]) in accepted}
        for filename in fact.expected_image_filenames:
            if filename not in filenames:
                source_errors.append(f"{fact.id}: missing associated image citation: {filename}")
    source_required = any(f.expected_image_filenames for f in expectations.facts)
    return {
        "visual_factual_accuracy": _metric(not fact_errors, *fact_errors),
        "visual_source_citation": (_metric(not source_errors, *source_errors) if source_required
            else MetricResult(status="not_evaluated", notes=[
                "No image filenames required; excluded from scoring."])),
    }


def evaluate_answer(case: EvaluationCase, observation: EvaluationObservation) -> EvaluationResult:
    """Never treat absent, stale, or invalid validation results as successful checks."""
    metrics = {name: MetricResult(status="not_evaluated") for name in METRICS}
    errors = []
    if observation.pipeline_error is not None:
        errors.append(observation.pipeline_error)
    raw = observation.output
    if isinstance(raw, ComposedAnswer):
        raw = raw.model_dump()
    try:
        # Reuse the real field and citation validators, even for existing instances.
        answer = ComposedAnswer.model_validate(raw)
    except ValidationError as exc:
        errors.append(f"Invalid ComposedAnswer: {exc}")
        metrics["output_contract"] = _metric(False, "Output failed production contract validation.")
        if any(error["loc"] and error["loc"][0] == "citations" for error in exc.errors()):
            metrics["citation_structure"] = _metric(False, "Invalid citation structure.")
        return _finish(case, metrics, errors)

    metrics["output_contract"] = _metric(True)
    metrics["citation_structure"] = _metric(True)
    metrics.update(evaluate_visual_expectations(case.visual_expectations, answer))
    metrics["question_match"] = _metric(answer.question == case.question)
    metrics["answer_status"] = _metric(answer.status == case.expected_status,
                                        f"Expected {case.expected_status}; received {answer.status}.")
    if case.expected_claims or case.forbidden_answer_fragments:
        missing = [claim for claim in case.expected_claims if claim not in answer.answer]
        forbidden = [text for text in case.forbidden_answer_fragments if text in answer.answer]
        metrics["answer_content"] = _metric(not missing and not forbidden,
            *[f"Missing required text: {text}" for text in missing],
            *[f"Forbidden text present: {text}" for text in forbidden])
    else:
        metrics["answer_content"] = MetricResult(status="not_applicable",
                                                  notes=["No literal content expectations supplied."])

    coverage = observation.coverage
    if coverage is None:
        metrics["citation_coverage"].notes = ["No recorded coverage assessment supplied."]
    elif coverage.answer != answer.answer or coverage.citations != answer.citations:
        metrics["citation_coverage"] = _metric(False, "Coverage snapshot does not match this output.")
    else:
        try:
            checked = CoverageVerdict.model_validate(coverage.verdict.model_dump())
            metrics["citation_coverage"] = _metric(checked.coverage == "complete",
                                                   checked.reason or checked.coverage)
        except ValidationError as exc:
            metrics["citation_coverage"] = _metric(False, "Malformed coverage verdict.")
            errors.append(str(exc))

    # Index binding prevents a verdict for one claim from scoring a different citation.
    reviews = observation.semantics
    indices = [review.citation_index for review in reviews]
    aligned = (len(indices) == len(set(indices)) and sorted(indices) == list(range(len(answer.citations)))
               and all(review.citation == answer.citations[review.citation_index]
                       for review in reviews if review.citation_index < len(answer.citations)))
    checked_reviews = []
    if not aligned:
        status = "not_evaluated" if not reviews and answer.citations else "fail"
        for name in ("groundedness", "explicit_absence"):
            metrics[name] = MetricResult(status=status, notes=["Semantic assessments missing or mismatched."])
    else:
        try:
            checked_reviews = [SemanticVerdict.model_validate(review.verdict.model_dump()) for review in reviews]
        except ValidationError as exc:
            errors.append(str(exc))
            metrics["groundedness"] = _metric(False, "Malformed semantic verdict.")
            metrics["explicit_absence"] = _metric(False, "Malformed semantic verdict.")
        else:
            if checked_reviews:
                metrics["groundedness"] = _metric(all(v.support == "supported" for v in checked_reviews),
                                                   *[v.reason or v.support for v in checked_reviews])
                metrics["explicit_absence"] = _metric(all(v.explicit_absence == "clear" for v in checked_reviews))
            else:
                metrics["groundedness"] = MetricResult(status="not_applicable", notes=["No factual citations."])
                metrics["explicit_absence"] = MetricResult(status="not_applicable")

    if case.expected_absence_claims:
        citation_claims = [citation["claim"] for citation in answer.citations]
        missing = [claim for claim in case.expected_absence_claims
                   if claim not in answer.answer or claim not in citation_claims]
        if missing:
            metrics["explicit_absence"] = _metric(False,
                *[f"Absence must be stated and cited: {claim}" for claim in missing])

    if case.expected_conflicts or answer.conflicts or case.expected_conflict_statements:
        metrics["conflict_handling"] = _metric(
            answer.conflicts == case.expected_conflicts
            and all(text in answer.answer for text in case.expected_conflict_statements),
            "Checks preserved conflict dictionaries and specified presentation text; does not rank sources.",
        )
    else:
        metrics["conflict_handling"] = MetricResult(status="not_applicable")

    if case.expected_unresolved_claims:
        missing = [gap for gap in case.expected_unresolved_claims if gap not in answer.answer]
        cited_gaps = [gap for gap in case.expected_unresolved_claims
                      if any(citation["claim"] == gap for citation in answer.citations)]
        metrics["unresolved_gaps"] = _metric(not missing and not cited_gaps,
            *[f"Gap missing: {gap}" for gap in missing],
            *[f"Gap incorrectly given an evidence citation: {gap}" for gap in cited_gaps])
    else:
        metrics["unresolved_gaps"] = MetricResult(status="not_applicable")
    return _finish(case, metrics, errors)

"""Run injected offline pipelines and return JSON-serializable comparison inputs."""

from collections.abc import Callable, Iterable

from .models import EvaluationCase, EvaluationObservation, EvaluationReport, EvaluationSummary
from .scoring import METRICS, evaluate_answer


def run_evaluation(
    cases: Iterable[EvaluationCase], pipeline: Callable[[str], EvaluationObservation],
    *, pipeline_name: str,
) -> EvaluationReport:
    """Give the pipeline questions only; never leak expected answers to it."""
    cases = list(cases)
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    origins = {case.data_origin for case in cases}
    if len(origins) > 1:
        raise ValueError("Run synthetic and archive cases in separate reports")
    results = []
    for case in cases:
        try:
            observation = EvaluationObservation.model_validate(pipeline(case.question))
        except Exception as exc:
            # A failed run is a failed case, not a dropped row in the denominator.
            observation = EvaluationObservation(pipeline_error=f"{type(exc).__name__}: {exc}")
        results.append(evaluate_answer(case, observation))
    counts = {name: {status: 0 for status in ("pass", "fail", "not_evaluated", "not_applicable")}
              for name in METRICS}
    for result in results:
        for name, metric in result.metrics.items():
            counts[name][metric.status] += 1
    passed = sum(result.passed for result in results)
    return EvaluationReport(
        pipeline_name=pipeline_name, data_origin=next(iter(origins), None), results=results,
        summary=EvaluationSummary(total=len(results), passed=passed, failed=len(results) - passed,
                                  pass_rate=passed / len(results) if results else 0.0,
                                  metric_counts=counts),
    )

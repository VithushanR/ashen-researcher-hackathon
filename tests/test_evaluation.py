"""Offline evaluation infrastructure tests; all facts and judgments are synthetic."""

import json
from copy import deepcopy

import pytest

from evaluation.models import EvaluationObservation
from evaluation.run_eval import run_evaluation
from evaluation.scoring import evaluate_answer
from src.answer.coverage_validation import CoverageVerdict
from src.answer.semantic_validation import SemanticVerdict
from tests.evaluation_fixtures import (
    ABSENCE, FACT, GAP, answer, case, citation, conflict_example, observation, representative_examples,
)


@pytest.mark.parametrize("reference,record,passed,failed_metric", representative_examples(),
                         ids=[row[0].id for row in representative_examples()])
def test_representative_evaluation_cases(reference, record, passed, failed_metric):
    result = evaluate_answer(reference, record)
    assert result.passed is passed
    assert result.data_origin == "synthetic"
    if failed_metric:
        assert result.metrics[failed_metric].status == "fail"
        assert result.score < 1.0
    else:
        assert result.score == 1.0
    assert json.loads(result.model_dump_json())["passed"] is passed


def test_invalid_citation_reuses_production_validator():
    raw = answer().model_dump()
    del raw["citations"][0]["page"]
    result = evaluate_answer(case(), EvaluationObservation(output=raw))
    assert not result.passed
    assert result.metrics["output_contract"].status == "fail"
    assert result.metrics["citation_structure"].status == "fail"


def test_missing_assessments_are_not_counted_as_passes():
    result = evaluate_answer(case(), EvaluationObservation(output=answer()))
    assert not result.passed
    assert result.metrics["citation_coverage"].status == "not_evaluated"
    assert result.metrics["groundedness"].status == "not_evaluated"


@pytest.mark.parametrize("field", ["answer", "citations"])
def test_stale_coverage_snapshot_fails(field):
    record = observation()
    if field == "answer":
        record.coverage.answer = "Different answer."
    else:
        record.coverage.citations = []
    result = evaluate_answer(case(), record)
    assert result.metrics["citation_coverage"].status == "fail"


@pytest.mark.parametrize("change", ["content", "index", "duplicate"])
def test_mismatched_semantic_assessments_fail(change):
    record = observation()
    if change == "content":
        record.semantics[0].citation["claim"] = "Different claim."
    elif change == "index":
        record.semantics[0].citation_index = 5
    else:
        record.semantics.append(record.semantics[0].model_copy(deep=True))
    result = evaluate_answer(case(), record)
    assert result.metrics["groundedness"].status == "fail"
    assert not result.passed


def test_malformed_verdict_instances_are_revalidated():
    record = observation()
    record.coverage.verdict = CoverageVerdict.model_construct(coverage="invalid")
    record.semantics[0].verdict = SemanticVerdict.model_construct(support="invalid", explicit_absence="clear")
    result = evaluate_answer(case(), record)
    assert result.metrics["citation_coverage"].status == "fail"
    assert result.metrics["groundedness"].status == "fail"
    assert result.errors


def test_explicit_absence_violation_fails_despite_positive_local_support():
    result = evaluate_answer(case(), observation(absence="violated"))
    assert result.metrics["groundedness"].status == "pass"
    assert result.metrics["explicit_absence"].status == "fail"
    assert not result.passed


def test_expected_absence_requires_both_text_and_citation():
    reference = case(expected_claims=[], expected_absence_claims=[ABSENCE])
    result = evaluate_answer(reference, observation(answer(answer=ABSENCE)))
    assert result.metrics["explicit_absence"].status == "fail"


def test_conflict_resolution_and_presentation_are_checked_without_ranking():
    reference, record = conflict_example()
    record.output.conflicts[0]["resolved_value"] = "red"
    result = evaluate_answer(reference, record)
    assert result.metrics["conflict_handling"].status == "fail"
    reference, record = conflict_example()
    reference.expected_conflict_statements.append("Required explanation missing from output.")
    assert evaluate_answer(reference, record).metrics["conflict_handling"].status == "fail"


def test_missing_gap_or_fabricated_gap_citation_fails():
    reference = case(expected_unresolved_claims=[GAP])
    assert evaluate_answer(reference, observation()).metrics["unresolved_gaps"].status == "fail"
    result = answer(answer=f"{FACT} {GAP}", citations=[citation(), citation(GAP)])
    assert evaluate_answer(reference, observation(result)).metrics["unresolved_gaps"].status == "fail"


def test_required_and_forbidden_text_expectations():
    reference = case(expected_claims=["Required sentence."], forbidden_answer_fragments=[FACT])
    result = evaluate_answer(reference, observation())
    assert result.metrics["answer_content"].status == "fail"
    assert len(result.metrics["answer_content"].notes) == 2


def test_empty_citations_need_coverage_but_no_semantic_assessment():
    reference = case(expected_status="partial_gap_stated", expected_claims=[], expected_unresolved_claims=[GAP])
    record = observation(answer(answer=f"Unresolved information: {GAP}", status="partial_gap_stated", citations=[]))
    result = evaluate_answer(reference, record)
    assert result.passed
    assert result.metrics["groundedness"].status == "not_applicable"
    assert result.metrics["citation_coverage"].status == "pass"


def test_runner_summary_includes_failures_and_does_not_leak_expectations():
    first = case(id="synthetic-one")
    second = case(id="synthetic-two", question="Second synthetic question?")
    received = []

    def pipeline(question):
        received.append(question)
        return observation(answer(question=question), support="supported" if question == first.question else "unsupported")

    report = run_evaluation([first, second], pipeline, pipeline_name="synthetic-test-pipeline")
    assert received == [first.question, second.question]
    assert report.summary.total == 2
    assert report.summary.passed == 1
    assert report.summary.failed == 1
    assert report.summary.pass_rate == 0.5
    assert report.summary.metric_counts["groundedness"]["pass"] == 1
    assert report.summary.metric_counts["groundedness"]["fail"] == 1
    assert json.loads(report.model_dump_json())["data_origin"] == "synthetic"


def test_pipeline_failure_is_recorded_and_runner_continues():
    calls = []

    def pipeline(question):
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("Synthetic pipeline failure")
        return observation(answer(question=question))

    report = run_evaluation([case(id="first"), case(id="second")], pipeline, pipeline_name="test")
    assert len(calls) == 2
    assert report.summary.failed == 1
    assert report.summary.passed == 1
    assert any("Synthetic pipeline failure" in error for error in report.results[0].errors)


def test_empty_run_has_zero_counts_without_division_error():
    report = run_evaluation([], lambda question: observation(), pipeline_name="empty")
    assert report.summary.total == 0
    assert report.summary.pass_rate == 0.0
    assert report.data_origin is None


@pytest.mark.parametrize("cases", [
    [case(), case()],
    [case(id="synthetic"), case(id="archive", data_origin="archive")],
])
def test_runner_rejects_duplicate_ids_and_mixed_origins(cases):
    with pytest.raises(ValueError):
        run_evaluation(cases, lambda question: observation(), pipeline_name="invalid")


def test_scoring_does_not_mutate_case_or_observation():
    reference, record = case(), observation()
    before = deepcopy((reference, record))
    evaluate_answer(reference, record)
    assert (reference, record) == before

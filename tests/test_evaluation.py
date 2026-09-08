"""Offline evaluation infrastructure tests; all facts and judgments are synthetic."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from evaluation.models import EvaluationObservation
from evaluation.run_eval import run_evaluation
from evaluation.scoring import evaluate_answer
from src.answer.coverage_validation import CoverageVerdict
from src.answer.semantic_validation import SemanticVerdict
from tests.evaluation_fixtures import (
    ABSENCE, FACT, GAP, DOG, CROW, answer, case, citation, conflict_example, observation,
    representative_examples, visual_case, visual_observation,
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


@pytest.mark.parametrize("text,passed", [(DOG, True), (CROW, False)])
def test_visual_dog_reference_is_independent_of_positive_runtime_judgments(text, passed):
    result = evaluate_answer(visual_case(), visual_observation(text))
    assert result.metrics["groundedness"].status == "pass"
    assert result.metrics["citation_coverage"].status == "pass"
    assert result.metrics["citation_structure"].status == "pass"
    assert result.metrics["visual_factual_accuracy"].status == ("pass" if passed else "fail")
    assert result.metrics["visual_source_citation"].status == ("pass" if passed else "fail")
    assert result.passed is passed


@pytest.mark.parametrize("text", ["No animal is described.", DOG + " " + CROW])
def test_missing_or_forbidden_visual_fact_fails(text):
    result = evaluate_answer(visual_case(), visual_observation(text))
    assert result.metrics["visual_factual_accuracy"].status == "fail"
    assert not result.passed


@pytest.mark.parametrize("citations", [[], [citation(DOG, "wrong.png")],
    [citation("An unrelated event.", "synthetic_banner.png")],
    [citation(DOG, "folder/synthetic_banner.png")]])
def test_visual_filename_requires_exact_original_and_associated_claim(citations):
    result = evaluate_answer(visual_case(), observation(answer(answer=DOG, citations=citations)))
    assert result.metrics["visual_factual_accuracy"].status == "pass"
    assert result.metrics["visual_source_citation"].status == "fail"
    assert not result.passed


def test_visual_normalization_alternatives_and_mixed_text():
    reference = visual_case()
    reference.visual_expectations.facts[0].accepted_statements.append("A dog appears on the banner.")
    text = "A DOG   appears on the\nbanner."
    record = observation(answer(answer=FACT + " " + text,
        citations=[citation(), citation(text, "synthetic_banner.png")]))
    assert evaluate_answer(reference, record).passed


@pytest.mark.parametrize("swapped", [False, True])
def test_multiple_images_and_swapped_citations(swapped):
    reference = visual_case()
    fact_type = type(reference.visual_expectations.facts[0])
    second = "The second banner depicts a horse."
    reference.visual_expectations.facts.append(fact_type(id="second",
        accepted_statements=[second], expected_image_filenames=["second.png"]))
    files = ["synthetic_banner.png", "second.png"]
    if swapped:
        files.reverse()
    record = observation(answer(answer=DOG + " " + second,
        citations=[citation(DOG, files[0]), citation(second, files[1])]))
    result = evaluate_answer(reference, record)
    assert result.metrics["visual_factual_accuracy"].status == "pass"
    assert result.passed is not swapped


def test_all_configured_image_filenames_required_but_extra_citations_allowed():
    reference = visual_case()
    reference.visual_expectations.facts[0].expected_image_filenames.append("second.png")
    assert not evaluate_answer(reference, visual_observation()).passed
    record = observation(answer(answer=DOG, citations=[citation(DOG, name) for name in
        ["synthetic_banner.png", "second.png", "additional.png"]]))
    assert evaluate_answer(reference, record).passed


def test_optional_visual_metrics_preserve_legacy_score_and_outcome():
    for reference, record, passed, _ in representative_examples():
        result = evaluate_answer(reference, record)
        legacy = [m for name, m in result.metrics.items()
                  if not name.startswith("visual_") and m.status != "not_applicable"]
        assert result.score == sum(m.status == "pass" for m in legacy) / len(legacy)
        assert result.passed is passed
        assert result.metrics["visual_factual_accuracy"].status == "not_evaluated"
        assert result.metrics["visual_source_citation"].status == "not_evaluated"


def test_no_filename_requirement_excludes_only_source_metric():
    reference = visual_case()
    reference.visual_expectations.facts[0].expected_image_filenames = []
    result = evaluate_answer(reference, visual_observation())
    assert result.passed and result.score == 1.0
    assert result.metrics["visual_source_citation"].status == "not_evaluated"
    assert not evaluate_answer(reference, EvaluationObservation(output=answer(answer=DOG))).passed


def test_configured_visual_metric_without_valid_output_cannot_pass():
    result = evaluate_answer(visual_case(), EvaluationObservation())
    assert result.metrics["visual_factual_accuracy"].status == "not_evaluated"
    assert result.metrics["visual_source_citation"].status == "not_evaluated"
    assert not result.passed


@pytest.mark.parametrize("change", ["reviewer", "notes", "facts", "id", "accepted",
    "blank_accepted", "blank_forbidden", "blank_filename", "duplicate", "missing_reviewer"])
def test_invalid_visual_expectations_rejected(change):
    raw = visual_case().model_dump()
    visual = raw["visual_expectations"]
    fact = visual["facts"][0]
    if change == "reviewer": visual["verified_by"] = " "
    elif change == "notes": visual["reference_notes"] = ""
    elif change == "facts": visual["facts"] = []
    elif change == "id": fact["id"] = ""
    elif change == "accepted": fact["accepted_statements"] = []
    elif change == "blank_accepted": fact["accepted_statements"] = [" "]
    elif change == "blank_forbidden": fact["forbidden_statements"] = [" "]
    elif change == "blank_filename": fact["expected_image_filenames"] = [" "]
    elif change == "duplicate": visual["facts"].append(deepcopy(fact))
    else: del visual["verified_by"]
    with pytest.raises(ValidationError):
        type(visual_case()).model_validate(raw)


def test_visual_report_json_counts_and_no_mutation():
    references = [visual_case(id="dog"), visual_case(id="crow")]
    records = [visual_observation(), visual_observation(CROW)]
    before = deepcopy((references, records))
    iterator = iter(records)
    report = run_evaluation(references, lambda question: next(iterator), pipeline_name="synthetic")
    assert report.summary.passed == report.summary.failed == 1
    assert report.summary.metric_counts["visual_factual_accuracy"]["pass"] == 1
    assert report.summary.metric_counts["visual_factual_accuracy"]["fail"] == 1
    assert json.loads(report.model_dump_json())["summary"]["pass_rate"] == 0.5
    assert (references, records) == before

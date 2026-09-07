"""Entirely synthetic evaluation examples and injected judgments, not archive truth."""

from copy import deepcopy

from evaluation.models import (
    CitationAssessment, CoverageAssessment, EvaluationCase, EvaluationObservation,
)
from src.answer.coverage_validation import CoverageVerdict
from src.answer.models import ComposedAnswer
from src.answer.semantic_validation import SemanticVerdict


FACT = "The synthetic gate is blue."
GAP = "The gate's construction date is not established."
ABSENCE = "Test Figure's canonical temperament is not established."


def citation(claim=FACT, filename="synthetic.md"):
    return {"claim": claim, "filename": filename, "page": None,
            "section": "Synthetic test section", "source_type": "wiki"}


def answer(**changes):
    fields = dict(question="What color is the synthetic gate?", answer=FACT,
                  status="complete", confidence=80, citations=[citation()],
                  conflicts=[], iterations_used=2)
    fields.update(changes)
    return ComposedAnswer(**fields)


def case(id="synthetic-clean", **changes):
    fields = dict(id=id, question="What color is the synthetic gate?",
                  data_origin="synthetic", expected_status="complete", expected_claims=[FACT])
    fields.update(changes)
    return EvaluationCase(**fields)


def observation(result=None, coverage="complete", support="supported", absence="clear"):
    result = result if result is not None else answer()
    return EvaluationObservation(
        output=result,
        coverage=CoverageAssessment(answer=result.answer, citations=deepcopy(result.citations),
                                    verdict=CoverageVerdict(coverage=coverage)),
        semantics=[CitationAssessment(
            citation_index=i, citation=deepcopy(item),
            verdict=SemanticVerdict(support=support, explicit_absence=absence),
        ) for i, item in enumerate(result.citations)],
    )


def conflict_example(resolved=True):
    left = 'Test source A reports: "The synthetic gate is blue."'
    right = 'Test source B reports: "The synthetic gate is red."'
    conflict = {
        "attribute": "synthetic gate color",
        "claims": [
            {"claim": FACT, "source": "Test source A", "chunk_id": "synthetic_a"},
            {"claim": "The synthetic gate is red.", "source": "Test source B", "chunk_id": "synthetic_b"},
        ],
        "resolution": "Supplied test resolution." if resolved else None,
        "resolved_value": "blue" if resolved else None,
    }
    ending = ("Preferred value: blue. Supplied test resolution." if resolved
              else "The evidence remains contradictory; no winner is established.")
    status = "complete_with_conflict" if resolved else "partial_gap_stated"
    text = f"Sources disagree. {left} {right} {ending}"
    result = answer(answer=text, status=status, conflicts=[conflict],
                    citations=[citation(left), citation(right, "synthetic_b.md")])
    reference = case(
        id="synthetic-resolved" if resolved else "synthetic-unresolved",
        expected_status=status, expected_claims=[left, right], expected_conflicts=deepcopy([conflict]),
        expected_conflict_statements=["Sources disagree.", ending],
    )
    return reference, observation(result)


def representative_examples():
    partial = answer(answer=f"{FACT}\nUnresolved information: {GAP}", status="partial_gap_stated")
    absent = answer(question="What is Test Figure's canonical temperament?", answer=ABSENCE,
                    citations=[citation(ABSENCE)])
    return [
        (case(), observation(), True, None),
        (*conflict_example(True), True, None),
        (*conflict_example(False), True, None),
        (case(id="synthetic-partial", expected_status="partial_gap_stated",
              expected_unresolved_claims=[GAP]), observation(partial), True, None),
        (case(id="synthetic-absence", question=absent.question, expected_claims=[ABSENCE],
              expected_absence_claims=[ABSENCE], forbidden_answer_fragments=["Test Figure is brave."]),
         observation(absent), True, None),
        (case(id="synthetic-unsupported"), observation(support="unsupported"), False, "groundedness"),
        (case(id="synthetic-uncovered"), observation(coverage="incomplete"), False, "citation_coverage"),
        (case(id="synthetic-wrong-status"), observation(answer(status="partial_gap_stated")),
         False, "answer_status"),
    ]

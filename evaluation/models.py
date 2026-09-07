"""Evaluation records reuse Person C's output and validator result contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.answer.coverage_validation import CoverageVerdict
from src.answer.models import ComposedAnswer
from src.answer.semantic_validation import SemanticVerdict


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationCase(EvaluationModel):
    """Reference expectations, not instructions passed to the evaluated pipeline."""

    id: str = Field(min_length=1)
    question: str
    data_origin: Literal["synthetic", "archive"]
    expected_status: str
    expected_claims: list[str] = Field(default_factory=list)
    forbidden_answer_fragments: list[str] = Field(default_factory=list)
    expected_absence_claims: list[str] = Field(default_factory=list)
    expected_conflicts: list[dict] = Field(default_factory=list)
    expected_conflict_statements: list[str] = Field(default_factory=list)
    expected_unresolved_claims: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("expected_status")
    @classmethod
    def validate_expected_status(cls, value: str) -> str:
        return ComposedAnswer.validate_status(value)


class CoverageAssessment(EvaluationModel):
    """Bind a recorded coverage verdict to the exact evaluated answer/citations."""

    answer: str
    citations: list[dict]
    verdict: CoverageVerdict


class CitationAssessment(EvaluationModel):
    """Bind a recorded semantic verdict to a citation's position and contents."""

    citation_index: int = Field(ge=0)
    citation: dict
    verdict: SemanticVerdict


class EvaluationObservation(EvaluationModel):
    """Pipeline output plus recorded checks; raw dictionaries allow invalid-output tests."""

    output: ComposedAnswer | dict | None = None
    coverage: CoverageAssessment | None = None
    semantics: list[CitationAssessment] = Field(default_factory=list)
    pipeline_error: str | None = None


class MetricResult(EvaluationModel):
    status: Literal["pass", "fail", "not_evaluated", "not_applicable"]
    notes: list[str] = Field(default_factory=list)


class EvaluationResult(EvaluationModel):
    case_id: str
    data_origin: Literal["synthetic", "archive"]
    passed: bool
    score: float
    metrics: dict[str, MetricResult]
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvaluationSummary(EvaluationModel):
    total: int
    passed: int
    failed: int
    pass_rate: float
    metric_counts: dict[str, dict[str, int]]


class EvaluationReport(EvaluationModel):
    pipeline_name: str
    data_origin: Literal["synthetic", "archive"] | None
    results: list[EvaluationResult]
    summary: EvaluationSummary

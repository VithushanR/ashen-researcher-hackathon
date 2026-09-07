"""Person C's structured answer contract for Person D's API and UI."""

from pydantic import BaseModel, field_validator


class ComposedAnswer(BaseModel):
    """Carry an answer and supplied metadata without resolving conflicts."""

    question: str
    answer: str
    status: str
    confidence: int
    citations: list[dict]
    conflicts: list[dict]
    iterations_used: int

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        """Accept only the team's agreed output statuses."""
        if value not in {"complete", "complete_with_conflict", "partial_gap_stated"}:
            raise ValueError(
                "status must be complete, complete_with_conflict, or partial_gap_stated"
            )
        return value

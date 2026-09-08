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

    @field_validator("citations")
    @classmethod
    def validate_citations(cls, value: list[dict]) -> list[dict]:
        """Check citation structure without changing values or judging evidence."""
        for index, citation in enumerate(value):
            for field in ("claim", "filename", "source_type", "page"):
                if field not in citation:
                    raise ValueError(f"citations[{index}].{field} is required")
            for field in ("claim", "filename", "source_type"):
                if not isinstance(citation[field], str):
                    raise ValueError(f"citations[{index}].{field} must be a string")
            page = citation["page"]
            # bool is an int subclass in Python, but is not a page number.
            if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
                raise ValueError(f"citations[{index}].page must be an int or None")
            section = citation.get("section")
            if section is not None and not isinstance(section, str):
                raise ValueError(f"citations[{index}].section must be a string or None")
        return value

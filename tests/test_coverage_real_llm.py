"""Genuine Gemini call through the coverage-validation adapter.

Every other coverage test mocks validate_coverage, so none of them can catch a
real model deviating from the expected output shape (e.g. echoing an input
field like required_claims back into its JSON, which trips CoverageVerdict's
extra="forbid"). This test exercises the real prompt against the real FAST
model with populated required_claims -- the exact path that was blind to that
bug -- and asserts the full round trip succeeds without extra_forbidden.

Skipped when no GEMINI_API_KEY is configured.
"""

import pytest

from src.answer.coverage_validation import PresentationCoverageVerdict, validate_answer_coverage
from src.answer.synthesis import CitationClaim
from src.api import llm

pytestmark = pytest.mark.skipif(not llm.llm_available(), reason="requires OPENROUTER_API_KEY")


def test_real_coverage_call_with_populated_required_claims():
    fact = "The Gloamreach archive was founded in the year 214 of the Age of Shadows."
    question = "In what year was the Gloamreach archive founded?"
    required_claims = [question]
    evidence = [{"chunk_id": "archive-1", "text": fact}]
    citation_claims = [CitationClaim(claim=fact, chunk_id="archive-1")]

    verdict = validate_answer_coverage(
        fact, citation_claims,
        validate_coverage=llm.validate_coverage,
        unresolved_claims=[], conflicts=None, no_evidence_returned=False,
        question=question, required_claims=required_claims, evidence=evidence,
        ordinary_answer=fact,
    )

    assert isinstance(verdict, PresentationCoverageVerdict)
    assert verdict.coverage == "complete"
    assert [item.requirement for item in verdict.presentation] == required_claims

"""
The sufficiency checker — the control point of the whole loop.

We deliberately do NOT ask the model "are you confident? yes/no" —
self-rated confidence is a known weak signal (spec §2.3). Instead, the
checker scores current evidence against a fixed rubric and must produce
a specific missing_info string describing what's still needed. That
string becomes the next search query (see planner.plan_next_query).

The LLM call itself goes through a separate call_llm() wrapper so
retry/backoff and caching live in one shared place, not duplicated
across every file that calls the LLM.
"""
import json

from pydantic import BaseModel, ValidationError

from agent.state import ResearchState
from agent.llm_client import call_llm

SUFFICIENCY_PROMPT_TEMPLATE = """You are checking whether enough evidence has been gathered to answer a question.

Question: {question}

Evidence collected so far:
{evidence_text}

Evaluate the evidence against this rubric:
1. coverage: does the evidence address every sub-part of the question? (yes / partial / no)
2. agreement: do the sources agree with each other, or do any contradict on the same fact? (yes / no / conflict)
3. unresolved: is there a specific claim still unsupported by any evidence? If yes, describe exactly what is missing in one sentence. If no, this must be null.

Some evidence items may be IMAGE chunks whose text is only a thin caption
or label — the fact being asked about may live in the image's pixels, not
in any text you can read here. Such items are marked "[IMAGE — pixels not
readable as text]" below. Judge coverage based only on text you can
actually read: if the only relevant evidence is one of those image items
and its caption does not answer the question, coverage is "no" and the
verdict is "insufficient", with missing_info naming the visual fact still
needed. The loop handles looking at the image itself separately.

Return ONLY a JSON object, no other text, in this exact shape:
{{"coverage": "...", "agreement": "...", "missing_info": "<specific string or null>", "verdict": "sufficient|insufficient|conflict_detected"}}

verdict rules:
- "sufficient" only if coverage is "yes" and agreement is "yes"
- "conflict_detected" if agreement is "conflict"
- otherwise "insufficient"
"""


class SufficiencyVerdict(BaseModel):
    """
    Structured output of one sufficiency check. Not part of ResearchState
    itself — this is a transient per-iteration object; loop.py reads it
    and appends the relevant parts (verdict, missing_info) to state.trace.
    """

    coverage: str        # "yes" | "partial" | "no"
    agreement: str       # "yes" | "no" | "conflict"
    missing_info: str | None
    verdict: str         # "sufficient" | "insufficient" | "conflict_detected"


def _format_evidence(state: ResearchState) -> str:
    """Render collected evidence as plain text for the prompt.

    Image chunks (content_type == "image") are tagged so the checker
    knows their visual content is NOT readable from the text shown —
    this is what lets it return "visual_evidence_only" instead of
    treating a thin caption as if it were the full fact.
    """
    if not state.evidence:
        return "(no evidence collected yet)"
    lines = []
    for item in state.evidence:
        marker = " [IMAGE — pixels not readable as text]" if item.content_type == "image" else ""
        lines.append(
            f"- [{item.filename}, p.{item.page}] ({item.reliability}){marker}: {item.text}"
        )
    return "\n".join(lines)


def check_sufficiency(state: ResearchState) -> SufficiencyVerdict:
    """
    Score the current state's evidence against the grounded rubric and
    return a structured verdict.

    Raises ValueError if the LLM's response can't be parsed or fails
    internal consistency checks — this is intentional. A malformed
    verdict should fail loudly here, not silently propagate a broken
    ResearchState downstream to Person C.
    """
    prompt = SUFFICIENCY_PROMPT_TEMPLATE.format(
        question=state.question,
        evidence_text=_format_evidence(state),
    )

    raw_response = call_llm(prompt)  # cheap/fast model per spec's model tiering

    try:
        parsed = json.loads(raw_response)
        verdict = SufficiencyVerdict(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Sufficiency checker returned malformed output: {raw_response!r}") from e

    _validate_verdict_consistency(verdict)
    return verdict


def _validate_verdict_consistency(verdict: SufficiencyVerdict) -> None:
    """
    Enforce the rubric's own rules in code, not just in the prompt — an
    LLM can still return an inconsistent combination even when told not
    to. Catch that here rather than let it corrupt the loop silently.
    """
    if verdict.verdict == "insufficient" and not verdict.missing_info:
        raise ValueError(
            "Verdict is 'insufficient' but missing_info is empty — "
            "the checker must always state what's missing when insufficient."
        )
    if verdict.verdict == "sufficient" and verdict.agreement == "conflict":
        raise ValueError(
            "Verdict is 'sufficient' but agreement is 'conflict' — "
            "a conflict must route through conflict_detected, not sufficient."
        )

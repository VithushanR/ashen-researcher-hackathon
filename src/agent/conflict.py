"""
Contradiction detection and resolution — the flagship differentiator
feature (spec §2.7).

Detection: LLM-based, not a heuristic pre-filter. We don't yet know the
real corpus well enough to safely narrow which evidence pairs might
conflict (e.g. via shared entity names) without risking missed
conflicts where the same fact is described differently. Once the real
corpus is available and conflict patterns are better understood, a
cheap pre-filter could be added to cut LLM calls — deliberately not
built now, noted in docs/decisions.md.

Resolution: NOT delegated to the LLM. This is a documented, ordered
policy implemented in code — source tier -> corroboration count ->
specificity -> directness -> one targeted verification search if still
unresolved (spec §2.7, Step 3). Keeping this in code, not a prompt, is
what makes it something the team can point to and defend verbally
rather than "the model decided."

IMPORTANT: if tier, corroboration, and specificity all tie with no
meaningful signal to break the tie, resolved_value is deliberately left
as None rather than forcing a guess. Person C's compose_answer() must
handle this as a valid terminal state (present both claims, favor
neither) — not treat it as "still pending."
"""
import json

from pydantic import BaseModel, ValidationError

from agent.state import ClaimSource, Conflict, Evidence
from agent.llm_client import call_llm

RELIABILITY_RANK = {
    "T1_authoritative": 4,
    "T2_curated": 3,
    "T3_narrative": 2,
    "T4_unverified": 1,
}

CONFLICT_DETECTION_PROMPT_TEMPLATE = """You are checking a set of evidence passages for factual contradictions.

Evidence:
{evidence_text}

Do any of these passages make incompatible claims about the same attribute
(e.g. a year, a count, a name, a location, an outcome)? Passages that are
merely about different topics are NOT a conflict.

Return ONLY a JSON object, no other text, in this exact shape:
{{
  "conflicts_found": true or false,
  "conflicts": [
    {{
      "attribute": "short description of what's disputed, e.g. 'year forged'",
      "claims": [
        {{"claim": "...", "chunk_id": "..."}},
        {{"claim": "...", "chunk_id": "..."}}
      ]
    }}
  ]
}}

If conflicts_found is false, "conflicts" must be an empty list.
"""


class _RawClaim(BaseModel):
    claim: str
    chunk_id: str


class _RawConflict(BaseModel):
    attribute: str
    claims: list[_RawClaim]


class _ConflictDetectionResult(BaseModel):
    conflicts_found: bool
    conflicts: list[_RawConflict]


def _format_evidence_for_detection(evidence: list[Evidence]) -> str:
    lines = []
    for item in evidence:
        lines.append(f"- chunk_id={item.chunk_id}: {item.text}")
    return "\n".join(lines)


def _evidence_by_chunk_id(evidence: list[Evidence]) -> dict[str, Evidence]:
    return {item.chunk_id: item for item in evidence}


def detect_conflicts(evidence: list[Evidence]) -> list[Conflict]:
    """
    Ask the LLM to identify any conflicting claims across all evidence
    collected so far. Returns Conflict objects with claims populated
    but resolution/resolved_value left as None — resolve_conflicts()
    fills those in.
    """
    if len(evidence) < 2:
        return []  # can't have a conflict with fewer than two sources

    prompt = CONFLICT_DETECTION_PROMPT_TEMPLATE.format(
        evidence_text=_format_evidence_for_detection(evidence)
    )
    raw_response = call_llm(prompt)

    try:
        parsed = json.loads(raw_response)
        result = _ConflictDetectionResult(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Conflict detector returned malformed output: {raw_response!r}") from e

    if not result.conflicts_found:
        return []

    chunk_lookup = _evidence_by_chunk_id(evidence)
    conflicts: list[Conflict] = []
    for raw_conflict in result.conflicts:
        claim_sources = []
        for raw_claim in raw_conflict.claims:
            source_evidence = chunk_lookup.get(raw_claim.chunk_id)
            source_label = (
                f"{source_evidence.filename} ({source_evidence.source_type}, {source_evidence.reliability})"
                if source_evidence
                else raw_claim.chunk_id  # fallback if LLM hallucinated a chunk_id
            )
            claim_sources.append(
                ClaimSource(
                    claim=raw_claim.claim,
                    source=source_label,
                    chunk_id=raw_claim.chunk_id,
                )
            )
        conflicts.append(Conflict(attribute=raw_conflict.attribute, claims=claim_sources))

    return conflicts


def resolve_conflicts(conflicts: list[Conflict], evidence: list[Evidence]) -> list[Conflict]:
    """
    Apply the ordered resolution policy to each conflict, filling in
    resolution and resolved_value. This is deterministic code, not an
    LLM call — the policy itself is the differentiator, and it needs to
    be something the team can explain and defend, not a black box.
    """
    chunk_lookup = _evidence_by_chunk_id(evidence)
    resolved: list[Conflict] = []
    for conflict in conflicts:
        winner = _apply_resolution_policy(conflict, chunk_lookup)
        resolved.append(winner)
    return resolved


def _apply_resolution_policy(conflict: Conflict, chunk_lookup: dict[str, Evidence]) -> Conflict:
    """
    Step 1 — source tier: prefer the claim from the highest reliability tier.
    Step 2 — corroboration: prefer the claim supported by more independent sources.
    Step 3 — specificity: prefer the more precise claim, but only if the
             difference is meaningful (not a crude "longer string wins").
    Step 4 — directness: not automatable without deeper source metadata
             (e.g. primary vs. secondary account) — known gap, noted below.
    Step 5 — targeted verification search: NOT done here — requires
             re-entering the loop to issue another search, handled in
             loop.py, not this function.

    If tier, corroboration, and specificity all tie with no meaningful
    signal, resolved_value is left as None — an honest "no justified
    winner" rather than an arbitrary guess.
    """
    # Step 1: source tier
    ranked_by_tier = sorted(
        conflict.claims,
        key=lambda c: RELIABILITY_RANK.get(
            chunk_lookup[c.chunk_id].reliability if c.chunk_id in chunk_lookup else "", 0
        ),
        reverse=True,
    )
    top_tier = RELIABILITY_RANK.get(
        chunk_lookup[ranked_by_tier[0].chunk_id].reliability
        if ranked_by_tier[0].chunk_id in chunk_lookup else "", 0
    )
    tied_at_top = [
        c for c in ranked_by_tier
        if c.chunk_id in chunk_lookup
        and RELIABILITY_RANK.get(chunk_lookup[c.chunk_id].reliability, 0) == top_tier
    ]

    if len(tied_at_top) == 1:
        winner = tied_at_top[0]
        conflict.resolved_value = winner.claim
        conflict.resolution = f"Preferred '{winner.source}' — highest reliability tier among competing claims."
        return conflict

    # Step 2: corroboration count (only relevant if tier was tied)
    claim_counts: dict[str, int] = {}
    for c in tied_at_top:
        claim_counts[c.claim] = claim_counts.get(c.claim, 0) + 1
    max_count = max(claim_counts.values())
    corroborated = [c for c in tied_at_top if claim_counts[c.claim] == max_count]
    if max_count > 1 and len(corroborated) < len(tied_at_top):
        winner = corroborated[0]
        conflict.resolved_value = winner.claim
        conflict.resolution = f"Preferred '{winner.claim}' — corroborated by {max_count} sources at the same reliability tier."
        return conflict

    # Step 3: specificity — only a tiebreaker if meaningfully more detailed
    lengths = [len(c.claim) for c in tied_at_top]
    if max(lengths) >= 2 * min(lengths):
        most_specific = max(tied_at_top, key=lambda c: len(c.claim))
        conflict.resolved_value = most_specific.claim
        conflict.resolution = f"Preferred '{most_specific.source}' — meaningfully more specific claim."
        return conflict

    # No justified winner — stay honest rather than guess.
    conflict.resolved_value = None
    conflict.resolution = (
        "Unresolved: competing claims are equally reliable, equally "
        "corroborated, and equally specific. No justified basis to prefer one."
    )
    return conflict

# Contracts

These are the agreed shared contracts for Persons B, C, and D. The Python
definitions below document the interfaces; they are not an implementation.
`BaseModel` refers to Pydantic. `Evidence` and `TraceStep` refer to the shared
types used by Person B; their fields are not defined by this update.

## ClaimSource and Conflict

```python
class ClaimSource(BaseModel):
    claim: str
    source: str
    chunk_id: str | None = None


class Conflict(BaseModel):
    attribute: str | None = None
    claims: list[ClaimSource]
    resolution: str | None = None
    resolved_value: str | None = None
```

`claims` replaces the outdated pairwise fields `claim_a`, `source_a`,
`claim_b`, and `source_b`. Consumers must iterate through `conflict.claims`
and support two or more competing claims.

Person B owns conflict detection and conflict-resolution logic, including
deciding which source wins. Person C receives and presents that result.
Person C must:

- Preserve each competing claim and its source.
- Use `resolution` and `resolved_value` when Person B provides them.
- Never invent a winning claim when a conflict is unresolved.
- Explain unresolved disagreements honestly in the final answer.
- Include the multi-claim conflict details in the structured output for Person D.

## ResearchState: Person B to Person C

```python
class ResearchState(BaseModel):
    question: str
    route: str
    required_claims: list[str]
    iteration: int
    search_history: list[str]
    evidence: list[Evidence]
    discovered_entities: list[str]
    unresolved_claims: list[str]
    conflicts: list[Conflict]
    confidence: int
    trace: list[TraceStep]
```

Person B returns this state from `research(question)`. `required_claims` is
part of the final contract. `final_answer` is no longer a ResearchState field:
Person C owns final answer composition and returns it through `ComposedAnswer`.

## ComposedAnswer: Person C to Person D

The agreed return contract is `compose_answer(state: ResearchState) -> ComposedAnswer`.
The earlier string-only return contract is outdated.

```python
class ComposedAnswer(BaseModel):
    question: str
    answer: str
    status: str
    confidence: int
    citations: list[dict]
    conflicts: list[dict]
    iterations_used: int
```

Supported status values are:

- `complete`
- `complete_with_conflict`
- `partial_gap_stated`

This structured result gives Person D's API and UI the answer text, status,
confidence, structured citations, conflict details, and iterations used.
The citations support clickable source references; the conflicts support
conflict callouts and preserve the multi-claim structure above as dictionaries.
The exact citation dictionary fields and rules for selecting a status are not
defined by this update.

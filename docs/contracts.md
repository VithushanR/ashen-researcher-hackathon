# Contracts

These are the agreed shared contracts for Persons B, C, and D. The Python
definitions below document the interfaces; they are not an implementation.
`BaseModel` and `Field` refer to Pydantic. `Evidence` and `TraceStep` are shared
types owned by Person B. The confirmed Evidence shape is documented below;
TraceStep's fields are not defined here.

## Evidence: Person B's normalized input to Person C

```python
class Evidence(BaseModel):
    chunk_id: str
    document_id: str | None = None
    filename: str
    source_type: str
    reliability: str
    page: int | None = None
    section: str | None = None
    content_type: str = "text"
    text: str
    entities: list[str] = Field(default_factory=list)
```

Confirmed content types are `text`, `table`, `image`, and `vision_description`.
`image_derived` is an additional source type. Person B creates a new Evidence
item for a vision description; it does not replace or mutate the original image
chunk. Person C does not implement a second copy of this production model.

Person C uses the supplied text for synthesis and semantic support checks.
For a vision-derived claim, citation filename remains the original image filename,
source_type remains `image_derived`, and page/section are copied exactly. The
existing page -> section -> filename location fallback applies. A raw image's
caption supports only facts stated in that text, not unseen visual attributes.
Provenance comes from Evidence metadata, never from the chunk-ID naming pattern;
the vision model itself is not the cited source. No new answer status or visual
confidence field is introduced.

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
Rules for selecting a status are not defined by this update.

### Citation dictionaries

The public contract remains `citations: list[dict]`. Each dictionary has:

| Field | Type | Required? |
| --- | --- | --- |
| `claim` | `str` | Yes |
| `filename` | `str` | Yes |
| `page` | `int \| None` | Yes, including when its value is `None` |
| `section` | `str \| None` | No; may be absent or `None` |
| `source_type` | `str` | Yes |

Citation location follows this priority:

1. If `page` is not `None`, use filename + page.
2. If `page` is `None` and a section is available, use filename + section.
3. If neither is available, fall back to filename.

Structural validation preserves supplied values and extra keys unchanged. It
does not normalize text, add an absent section, or check whether a source
supports a claim. Page values are integers or `None`, not booleans, numeric
strings, or floats. Citation rendering is separate from structural validation.

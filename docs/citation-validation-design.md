# Person C semantic citation validation design

## Current safeguard and its limits

Synthesis instructions require explicit absence statements to be answered
directly and cited. They prohibit reconstructing an explicitly unestablished
attribute from behavior, occupation, activity, roles, or events. The instruction
applies to all supplied evidence, including adjacent chunks about the same entity.

An explicit statement such as "No canonical temperament is established" is
positive evidence about what is established in the archive. It is different from
retrieving no temperament information. An absence answer may be `complete` when
Person B returns no gaps or conflicts. Person C does not change confidence,
sufficiency, conflict decisions, or the existing status rules for this safeguard.

The executable pipeline checks synthesis structure, validates coverage of the
final assembled answer, resolves every chunk ID, then calls an injected
semantic-validation adapter for every generated atomic claim before constructing
its citation. Both coverage and semantic checks must pass before returning the
answer. No provider, SDK, or model is selected here.

### Runtime coverage gate

`compose_answer(..., validate_coverage=adapter)` requires a coverage adapter for
every accepted answer, including zero-claim partial answers, deterministic gap-only
answers, and conflict presentations. The internal Pydantic result is:

```text
coverage: complete | incomplete | uncertain
uncovered_claims: list[string] (optional, defaults to [])
reason: string | null (optional)
```

Only `complete` with no uncovered claims passes. Incomplete or uncertain results
raise `CitationCoverageError` carrying the original answer and verdict. Missing
wiring, wrong types, unknown values, extra fields, or an internally inconsistent
complete verdict fail safely. Adapter exceptions propagate. No answer text is
silently rewritten, deleted, or repaired.

The prompt compares the exact final answer against the structured claim/chunk-ID
pairs. It requires every atomic factual idea, including extra clauses and
inferences within a sentence, to be represented. Faithful paraphrases count;
word overlap alone does not. Transitions and non-factual wording need no claim,
but hedged factual assertions still do. Explicit archive absence is factual
content and must have a declared absence claim.

Partial-answer coverage sees the appended gaps as well as synthesis text. Trusted
research context, passed by code rather than synthesis, permits faithful reports
of B's unresolved claims without fabricated citations. It does not permit a
positive answer to a gap. The deterministic empty-evidence report supplies that
state fact explicitly. Conflict coverage receives the preserved Conflict objects
and attributed claim references, allowing faithful reports of B's decisions and
source attributions (including those without chunk IDs). It must not reinterpret
or re-resolve them. No source ranking or new B-side fields are introduced.

Validation order for synthesized answers:

```text
evidence-index sanity checks
-> synthesis + structured parsing
-> assemble final text, including B's gaps
-> citation coverage
-> resolve every generated chunk ID
-> semantic support + explicit-absence compatibility (same adapter call)
-> trusted citation construction
-> ComposedAnswer
```

Conflict states first build deterministic B conflict presentation and resolve all
supplied conflict chunk IDs. Synthesis uses all evidence for an optional independent
ordinary section; whole conflict chunks are not removed. OrdinarySectionResult
permits only null answer with no claims, or nonempty text with cited claims.
Instances/subclasses are revalidated without weakening clean/partial contracts.

The composer combines ordinary text, deterministic conflicts, and one B gap section.
One coverage call sees the exact final answer and a separate ordinary_section.
It must reject ordinary restatement, paraphrase, resolution, or strengthening of
disputed claims, and repeated gap reporting; ambiguous separation is uncertain.
Independent facts from the same entity or chunk remain allowed. This is a
presentation check, not winner selection or a general answer-completeness check.

After coverage, all ordinary IDs are checked, ordinary claims use normal semantic
validation, and raw competing claims use conflict-attribution validation. Only then
are trusted citations constructed. Shared chunk IDs do not cause deduplication.
Disagreement alone does not invalidate a directly supported conflict attribution;
scope, uncertainty, strengthening, and absence inference checks remain required.
B's claims, resolution, and resolved_value are unchanged. Any unresolved conflict
or B gap yields partial_gap_stated; otherwise conflicts yield complete_with_conflict.
Missing conflict references retain their notice without a citation or semantic call.
Gap-only answers without evidence retain their existing coverage-only path.

The adapter receives a prompt containing the atomic claim, the exact referenced
passage, and the full evidence set (chunk IDs, text, document IDs, sections, and
entities). Context metadata comes only from B's supplied Evidence objects. It is
used to bind entity/attribute scope, not to rank sources. Support must come from
the cited passage; another passage cannot rescue an incorrect citation.

### Runtime verdict and failure policy

`compose_answer(..., validate_semantics=adapter)` uses this internal result:

```text
support: supported | unsupported | contradicted | uncertain
explicit_absence: clear | violated | uncertain
reason: string | null (optional)
absence_findings: [{chunk_id: string, passage: string}] (optional)
```

Both verdict fields are required. Pydantic rejects wrong types, unknown values,
and extra fields. A finding must name a real chunk and quote an exact nonempty
passage from it. A violation requires at least one finding. The two checks are
independent: even `support=supported` fails if `explicit_absence=violated`.

Only `supported` plus `clear` passes. All other verdicts raise
`CitationSupportError`, carrying the original claim, chunk ID, and structured
verdict. Malformed output, fabricated findings, adapter failures, or missing
adapter wiring also fail before returning an answer. The composer does not
silently drop, rewrite, or repair a rejected claim and adds no public statuses.

This applies to generated claim/chunk-ID pairs in clean and partial synthesis
and to every conflict claim with a supplied chunk reference.
Deterministic conflict presentation remains B-owned in substance and is not
sent for re-resolution. Gap-only answers without generated factual claims do
not require a semantic call. B's gap statements receive no fabricated citations.

The shared semantic adapter is still pending. Runtime gating is implemented;
semantic judgment accuracy depends on that adapter and requires later real-model
validation. Tests inject verdicts and verify the payloads, strict parsing, and
rejection behavior without network access. Existing composition regressions use
an explicitly named test-only supported-verdict double.

## Semantic checking rules

The following rules guide the adapters. Whole-answer coverage, per-pair runtime
gating, and the explicit-absence prompt are implemented; this is not a new
Person B contract:

1. Check every factual assertion in the final answer, including assertions that
   lack a corresponding citation claim. Resolve each supplied chunk ID before
   checking support; a real ID is necessary but never sufficient.
2. Inspect all evidence supplied in the research state for explicit absence
   statements relevant to the assertion's entity and attribute. Looking only at
   the cited chunk misses an absence stated in a neighboring chunk.
3. Bind each absence to its entity, attribute, and scope using the surrounding
   text. Keep its exact supporting passage and chunk ID for the validation
   finding. Runtime code verifies the quote and ID; their semantic relevance is
   the adapter's responsibility. Do not infer entity identity merely from proximity. Ambiguous scope
   needs review; it must not automatically block another entity's attributes.
4. Apply an additional anti-inference check to claims touching that attribute,
   including paraphrases such as "brave" for temperament or "tall" for physical
   appearance. Check whether the claim faithfully reports the absence or instead
   reconstructs the attribute from an action, role, occupation, or event.
5. A faithful absence claim must retain qualifications such as "canonical" or
   "recorded" and cite an absence-supporting passage. "Unrecorded" does not mean
   "never happened"; "not established" does not mean "does not exist". Silence
   alone does not support an explicit absence assertion.
6. Flag an inferred attribute even when its cited chunk exists and supports the
   neighboring action or occupation. Keep this finding separate from an unknown
   chunk-ID error. Runtime code rejects the entire answer on a failed verdict;
   review or regeneration is left to the caller. No new public response statuses
   are added.
7. Validate direct event claims independently. An unestablished temperament does
   not invalidate a documented rescue; unestablished appearance does not
   invalidate a documented occupation. Do not reject the entire chunk or entity.
8. If direct attribute evidence and an explicit absence genuinely disagree,
   surface the discrepancy for Person B's conflict process. Person C must not
   rank sources, select a winner, alter B's resolution, or rewrite unresolved
   claims to make the answer appear complete.

These checks require semantic interpretation and cannot be replaced by a
keyword blacklist. Implementation must use the team's agreed validation boundary
and failure policy; no provider or model is selected in this design.

## Visual evidence compatibility and limits

Vision-derived evidence is validated for internal consistency between the final claim and the vision-model description, but the vision description is not independently re-verified against the original image pixels.

`vision_description` follows the existing text-based synthesis, coverage, and
semantic support path. It is already-normalized Evidence from Person B, not an
instruction to load an image or call a vision model. No production code change
was needed for this compatibility: the generic path consumes Evidence.text and
copies citation metadata directly. `text` and `table` retain the same behavior.

For image-derived claims, the citation uses the original image filename,
`source_type="image_derived"`, and the exact supplied page and section. Provenance
is taken from Evidence metadata, not parsed from chunk_id. The vision model is
not cited as a source. B's new vision item and original image item remain separate.

Reliability on image_derived evidence currently represents the inherited
authority tier of the original source, not independent confidence in the vision
interpretation. Person C neither re-ranks that source nor recalculates confidence.

Person C must not infer visual facts from a thin raw image caption. A relevant
image name is not support for an unstated appearance, count, or symbol. A raw
caption may support a fact literally stated in its text. A claim citing a raw
caption cannot borrow support from another chunk's vision description. Uncertain
vision text such as "appears to show a raven" cannot support "definitely shows a
raven"; the existing semantic gate rejects an unsupported verdict, while a
faithfully hedged claim may pass. The semantic adapter remains responsible for
that judgment; deterministic regression tests inject verdicts and verify the
actual referenced text and rejection behavior without testing image pixels.

Visual-answer accuracy should be included in later real/hand-verified evaluation,
with reviewers inspecting original images as well as answers and descriptions.
The current evaluation infrastructure does not measure image-grounded accuracy.
Vision calls, targeted vision prompts, repeated-call prevention, sufficiency,
conflict decisions, and any shared provider wiring remain outside Person C.

## Acceptance cases

All examples below are synthetic, not archive facts.

| Evidence | Answer claim | Expected semantic finding |
| --- | --- | --- |
| No canonical physical features are established for Test Figure. | Test Figure's canonical appearance is not established. | Supported absence, cite the absence passage. |
| No canonical temperament is established for Test Figure. | Test Figure's canonical temperament is not established. | Supported absence; may fully answer the question. |
| Same temperament absence; another passage records a rescue. | Test Figure is brave. | Unsupported attribute inference, even with the rescue chunk's real ID. |
| Same appearance absence; another passage records work as a smith. | Test Figure is muscular. | Unsupported attribute inference, even with the occupation chunk's real ID. |
| Same temperament absence; another passage records a rescue. | Test Figure rescued a traveler. | Evaluate against the event passage; the absence does not invalidate it. |
| No recorded appearance for Test Figure; a passage describes Other Figure. | Other Figure has red hair. | Do not transfer Test Figure's absence; check Other Figure's cited evidence. |
| No canonical temperament is established. | Test Figure has no temperament. | Unsupported strengthening of the absence itself. |

Runtime tests now feed incorrect claims and failing semantic verdicts into the
composer and verify rejection, including a locally supported claim blocked by
cross-chunk explicit absence. They also exercise valid absence/event claims,
malformed output, unknown IDs, and fabricated quotations. The tests establish
orchestration behavior, not real-model semantic accuracy.

Coverage tests inject incomplete verdicts for an omitted sentence, an extra
inference within a covered sentence, and factual text with no declared claims.
They verify rejection before semantic checking, strict result parsing, required
wiring, and continued failure on unknown IDs or unsupported declared claims.
Existing regression tests use an explicit test-only complete-coverage adapter.

Remaining integration dependencies: shared synthesis, coverage, and semantic
adapters. Deterministic tests verify orchestration and fail-safe handling of
their results; actual semantic/coverage judgment accuracy still needs testing
with the team's selected models and real archive examples. There is no production
fallback that assumes coverage or semantic support when an adapter is missing.

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

The executable pipeline now checks synthesis structure, resolves every chunk ID,
then calls an injected semantic-validation adapter for every generated atomic
claim before constructing its citation. Only a passing semantic verdict permits
the final answer to be returned. No provider, SDK, or model is selected here.

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

This applies to generated claim/chunk-ID pairs in clean and partial synthesis.
Deterministic conflict presentation remains B-owned in substance and is not
sent for re-resolution. Gap-only answers without generated factual claims do
not require a semantic call. B's gap statements receive no fabricated citations.

The shared semantic adapter is still pending. Runtime gating is implemented;
semantic judgment accuracy depends on that adapter and requires later real-model
validation. Tests inject verdicts and verify the payloads, strict parsing, and
rejection behavior without network access. Existing composition regressions use
an explicitly named test-only supported-verdict double.

## Semantic checking rules and remaining coverage work

The following rules guide the adapter. Per-pair runtime gating and the
explicit-absence prompt are implemented. Whole-answer coverage in step 1 remains
future work; this is not a new Person B contract:

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

Remaining coverage work: compare all factual assertions in `answer` with the
generated citation claims, including answers with empty citation-claim lists.
The current runtime gate checks every supplied pair; it does not yet detect an
additional unsupported sentence omitted from `citation_claims`. The synthesis
prompt prohibits such omissions, but a prompt is not a runtime coverage check.

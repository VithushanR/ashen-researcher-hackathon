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

The executable checks currently validate synthesis structure, citation metadata
types, and exact chunk-ID lookup. These checks do **not** establish entailment.
A fabricated personality claim citing a real behavioral passage can still pass
those structural checks. The prompt is a generation safeguard, not a semantic
validation guarantee. Deterministic tests use injected synthesis responses and
verify the prompt rules, expected output, citation provenance, and status.

## Additional checking required in the semantic validator

The following is a design for the next validation milestone, not implemented
semantic checking or a new Person B contract:

1. Check every factual assertion in the final answer, including assertions that
   lack a corresponding citation claim. Resolve each supplied chunk ID before
   checking support; a real ID is necessary but never sufficient.
2. Inspect all evidence supplied in the research state for explicit absence
   statements relevant to the assertion's entity and attribute. Looking only at
   the cited chunk misses an absence stated in a neighboring chunk.
3. Bind each absence to its entity, attribute, and scope using the surrounding
   text. Keep its exact supporting passage and chunk ID for the validation
   finding. Do not infer entity identity merely from proximity. Ambiguous scope
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
   chunk-ID error. Do not release that claim as semantically validated. The next
   milestone must agree how findings trigger rejection, review, or regeneration;
   do not add new public response statuses implicitly.
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

The future validator needs adversarial tests that feed the **incorrect** claims
above directly into validation and verify rejection or flagging. Current mocked
synthesis tests demonstrate the intended generation contract, not those future
semantic rejection guarantees.

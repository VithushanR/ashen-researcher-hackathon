# Technical Decisions

Real decisions made across the build, in each owner's own reasoning. Each entry: what
was chosen, what was considered instead, and why the alternative was rejected.

---

## Ingestion & Retrieval (Person A)

### 1. Sufficiency-driven routing over a keyword pre-classifier (shared with B)
See Agent Loop section — this decision spans both components' original planning.

### 2. Local ChromaDB instead of a cloud vector database
**Chose:** Persisted ChromaDB on disk. **Rejected:** MongoDB Atlas / Pinecone. **Why:**
no network dependency during the live demo; reproducible from a clean checkout via
`build_corpus.py`; the corpus (1892 chunks post-dedup) is nowhere near the scale that
justifies a distributed store; avoids a fifth service account on top of Voyage,
OpenRouter, and GitHub.

### 3. Deduplicating `.docx`/`.pdf` pairs in chronicles/codex, never in ephemera
**Chose:** When a chronicles/codex document exists as both formats with the same base
filename (no `.scan.` marker), only `.docx` is ingested. **Why:** manual retrieval
testing proved these pairs are byte-for-byte duplicate content — before the fix, 2 of
5 top results for a real query were the same paragraph twice. Chunk count dropped
3217 → 1892; retrieval diversity measurably improved. **Why ephemera is excluded:**
its clean/`.scan.` pairs were confirmed to hold *different* transcripts of the same
event (e.g. two distinct interrogation-record questioning lines for the same subject)
— deduplicating would destroy real, distinct corroboration evidence.

### 4. Vision fallback built as a required feature
**Why:** direct inspection of real wiki pages confirmed visual facts are deliberately
never captioned in text. 11 of 20 real sample questions require information contained
in image pixels alone.

### 5. Entity extraction via a heuristic, not a full NER model
**Chose:** regex — runs of 2+ Title-Case words, permitting lowercase connectors mid-
phrase. **Why:** no NER library was in the toolchain; adding one purely for this would
be disproportionate. Verified against real text (correctly extracts "Sabelle
Mournvale the Twice-Crowned"). Known weaknesses (ballad-formatted false positives,
possessive suffixes) are documented rather than chased to perfection.

---

## Agent Loop (Person B)

### 6. Sufficiency checking uses a grounded rubric, not self-rated confidence
**Chose:** score evidence against a fixed rubric (coverage / agreement / unresolved),
output a specific `missing_info` string. **Rejected:** asking the model "are you
confident? yes/no." **Why:** self-rated confidence is a known weak signal — models
report high confidence even when wrong. A rubric forces the model to check each
sub-part, and `missing_info` becomes the next search query, which is what drives
multi-hop behavior.

### 7. Route is derived after the loop runs, not predicted before it
**Chose:** if sufficient after one iteration, label "simple" afterward; otherwise
"multihop." **Rejected:** a keyword pre-classifier (`MULTIHOP_MARKERS` list). **Why:**
a fixed keyword list is guessed from a handful of known examples and won't generalize
to the hidden judging set's different phrasing. Deriving route from actual behavior
works on any question by construction.

### 8. Conflict resolution is deterministic code, never delegated to the LLM
**Chose:** an ordered policy — source tier → corroboration count → specificity — coded
in Python. **Rejected:** asking the LLM "which of these is right?" **Why:** the
resolution must be explainable and defensible to a judge in the final round. "The
model decided" has no good answer under questioning; "T1 codex outranks T4 ballad
because we tier sources at ingestion" does.

### 9. Never force a winner on a genuine tie
**Chose:** when tier, corroboration, and specificity all tie, `resolved_value` stays
`None` and the conflict is reported honestly, not guessed. **Why:** forcing a pick
between equally-reliable sources produces a confident answer that's secretly a
coin-flip — exactly the failure mode the archive was built to expose.

### 10. One shared LLM client with per-call model selection
**Chose:** a single `call_llm(prompt, model=...)`, model IDs centralized in
`models.py`, both B and C use it. **Rejected:** each component maintaining its own
provider client. **Why:** duplicated clients mean duplicated retry/backoff, scattered
API-key handling, and no single place to swap models under free-tier rotation.

### 11. `required_claims`: designed as the shared, stable checklist across B and C —
**partially implemented, deliberately not yet fully wired**

A full proposal exists (see below) to make `required_claims` the single, stable
definition of what a complete answer must contain, shared identically between B's
sufficiency checking and C's coverage validation, rather than each component
independently re-deriving its own understanding of the question from raw text on
every call.

**The proposed design, in brief:**
- B decomposes the question into `required_claims` **once**, via one LLM call, at the
  start of `research()` — using an LLM (not Python string-splitting) because
  semantic decomposition must correctly handle cases like "compare the codex **and**
  wiki accounts" (where "and" joins sources, not separate questions) or "who ruled
  before **and** after the rebellion" (shared subject, two time periods) — patterns a
  keyword rule cannot reliably distinguish.
- B's sufficiency checker evaluates evidence against each claim explicitly, tracking
  per-requirement status (supported / missing), rather than judging the whole question
  as one undifferentiated block.
- The same `required_claims` list is passed to C, so C's synthesis and coverage
  validation check the final answer against the *identical* checklist B used — instead
  of C independently re-reading the raw question and potentially reaching a different
  boundary of what "complete" means.
- If C's synthesis omits a requirement B had already established, that is a
  composition error, not a research gap — C performs one controlled repair using only
  B's existing evidence (no new research, no invented facts), then revalidates. A
  second failure fails closed rather than looping indefinitely.
- If decomposition itself fails, `required_claims = []`; research continues, and C
  falls back to its original question-based completeness check rather than generating
  a second, competing checklist of its own.

**Current, actual state:** `required_claims` exists on `ResearchState` and is
populated via the decomposition call. Neither B's sufficiency checker nor C's
coverage validator consume it yet — both currently operate exactly as they did before
this field existed. We deliberately deferred full wiring: it touches core loop and
composition logic, and correctly implementing the repair-vs-reject question for C
(surfaced only once the proposal was written out in full) deserves proper design
rather than a rushed change in the final day before submission. The field is
populated and available for a scoped, two-stage follow-up (B's consumption, then C's,
as separate reviewable commits), rather than left as an unexplained placeholder.

---

## Answer Composition & Evaluation (Person C)

### 12. Citation metadata comes from Evidence, never from the LLM
**Chose:** the LLM identifies which `chunk_id` supports a claim; Python resolves that
ID against `ResearchState.evidence` and copies the real filename/page/section/type.
**Rejected:** asking the LLM to generate complete citations directly. **Why:**
LLM-generated metadata can be hallucinated even when the answer text itself is
correct. Resolving programmatically from known Evidence means the system cannot
invent a page number or filename that was never actually retrieved.

### 13. Validate claims after generation, rather than trusting synthesis alone
**Chose:** separate answer generation from verification — after synthesis, check
whether each claim is semantically supported by its cited evidence, and whether the
question's important parts were covered. **Why:** good retrieval doesn't guarantee a
grounded final answer; the model can still exaggerate, combine facts incorrectly, or
cite a source that doesn't actually support the exact claim made.

### 14. Person C never overrides Person B's conflict decisions
**Chose:** B owns conflict detection and resolution; C only presents the result.
**Why:** allowing C to also weigh conflicting sources while writing prose risks
inconsistent decisions — B might correctly mark two same-tier sources unresolved, but
C could quietly pick one anyway while trying to sound confident.

### 15. Explicit partial/conflict statuses instead of always returning confidence
**Chose:** `complete`, `complete_with_conflict`, `partial_gap_stated` as distinct,
visible outcomes. **Why:** the challenge is about searching until there is enough
evidence, not producing fluent text regardless. An incomplete answer should say so
explicitly, not hide the gap behind confident phrasing.

### 16. Vision-derived evidence is treated like any other evidence, once created
**Chose:** C runs no second, independent vision-verification call on
`vision_description` evidence — it's validated for internal consistency (does the
final claim match what the vision model said) exactly like any text evidence.
**Rejected:** a second vision call inside C to re-check the image. **Why:** would
duplicate B's responsibility, double API cost/latency, and likely just ask a similar
model to verify its own prior visual read — not genuinely independent verification.
Visual-answer accuracy against the real image is instead a manually-scored evaluation
metric, not an automated runtime check.

---

## Interface & Integration (Person D)

### 17. Everything routes through one seam module, never direct imports
**Chose:** the API and UI call `pipeline.py`, which resolves each teammate's entry
point from a configurable `module:function` string at call time, falling back to
fixtures when one is missing. **Rejected:** importing `research()` and
`compose_answer()` directly at the top of the API. **Why:** with four people building
in parallel against a hard deadline, direct imports mean the interface can't exist
until everyone else finishes, concentrating all UI risk into the final two days, and
one teammate's broken import would take the whole demo down. With the seam, the
interface was built and fully tested on day one against a fake contract; integration
became a configuration change, not a rewrite.

### 18. System status is four independent flags, not one boolean
See architecture.md's "Honest reporting" section — chosen after a single boolean was
wrong three separate times while every visible signal stayed green.

### 19. Build the UI against a written contract before the dependency exists
**Chose:** on day one, wrote a JSON fixture with three canned answer states (clean,
conflicting, partial) and built the entire interface against it. **Why:** the three
answer states are the actual point of this sub-track — an honest partial answer and a
surfaced conflict are what the system is judged on, not the happy path. Building
against whatever the first real end-to-end run emits would mean the conflict and
partial paths get designed and tested in the last 48 hours, under pressure, since a
clean answer is what comes out first in any working pipeline.

### 20. Permissive about what's accepted, strict about what's sent
**Chose:** response models allow unknown fields to pass through untouched; the request
model forbids them. **Why:** the asymmetry maps onto who owns the mistake. If a
teammate improves their component and adds a field, that's normal progress and must
not break the UI. If D's own code sends a misspelled field name, being strict there
means the bug is caught immediately rather than silently succeeding with a feature
quietly disabled.

### 21. The model adapter layer adds capability, never a second API client
**Chose:** `llm.py` adds caching, adapter-level retry, model tiering, and JSON
extraction on top of Person B's existing OpenRouter client, making no HTTP call of its
own — enforced structurally by a test that fails if the file contains `openrouter.ai`
or `import requests`. **Rejected:** a second, independent OpenRouter integration.
**Why:** two clients with two independent retry policies is indefensible under
judging, and doubles the free-tier rate-limit surface — two layers each retrying four
times is sixteen attempts for one logical call, a real path to a mid-demo quota ban.
# Known Limitations

Honest, specific gaps — what doesn't work perfectly, discovered by testing rather than
assumed, and why each was an acceptable trade-off given the time available.

---

## Ingestion & Retrieval

- **Entity extraction produces false positives on ballad-formatted ephemera.**
  Poetry-style documents capitalize the first word of every line, so the heuristic
  occasionally captures phrases like "For Crookgate Keep" as an entity, including the
  leading preposition. The real name still exists as a substring, so exact-name
  search is unaffected.
- **Possessive suffixes are not stripped** ("House Morvain's" tracked separately from
  "House Morvain").
- **Retrieval quality was spot-checked with three real queries**, confirming the
  pipeline functions and that deduplication measurably improved diversity — exhaustive
  validation across all 20 sample questions is the evaluation harness's job, not
  re-verified independently at the retrieval-component level.
- **OCR quality reflects this corpus's own design**, not a hardened pipeline —
  Tesseract performs well because the `.scan.pdf` files are clean, digitally-rendered
  simulated scans, not genuinely degraded photographs; no claims are made about
  real-world scan noise beyond what this specific corpus contains.

## Agent Loop

- **`required_claims` is computed but not yet consumed.** The field is populated via a
  real LLM call, but neither the sufficiency checker nor the coverage validator read
  it yet — both still independently interpret the raw question on every call. A full
  design for consuming it exists (see `decisions.md` #11) but was deliberately scoped
  out of this submission given the risk of changing core loop and composition logic
  in the final day.
- **Conflict resolution only fully implements the tier step.** Corroboration count and
  specificity are coded but simple; "directness" (primary vs. secondary source) isn't
  automated, since it needs source metadata not reliably available in the current
  chunk contract.
- **The loop has only ever run against fake data and mocked LLM calls.** All tests
  mock the LLM and use fake chunks — it has not been run end-to-end against the real
  retrieval index or a live OpenRouter model, so behavior against messy real evidence
  and occasionally malformed model JSON is unverified.
- **Model IDs were unverified placeholders** until locked via `models.py` — no real
  OpenRouter run was possible before that point.
- **`capped_unresolved` never appears as an explicit trace verdict.** When the loop
  hits its iteration cap, the signal lands in `unresolved_claims`, not as a distinct
  trace entry — a minor inconsistency between the documented verdict set and what's
  actually written, later worked around by Person D deriving the same signal from "a
  finished trace whose last round is still `insufficient`."
- **False-premise questions are not specially detected.** The system falls back to
  its honest partial-answer path — correct in outcome, reached less directly than a
  system with explicit premise-checking would achieve. Scoped out as solving a
  problem with no confirmed presence in the actual hidden question set.

## Answer Composition & Evaluation

- **C cannot independently verify upstream evidence is factually correct**, only that
  the final answer is faithful to the evidence it was given. If B's vision step
  misreads an image and produces an incorrect `vision_description`, C confirms
  internal consistency, not pixel-level accuracy — a second, independent vision check
  was deliberately not added (see `decisions.md` #16).
- **Semantic validation is LLM-based, not a formal proof system** — subtle wording
  differences or ambiguous claims may still be judged incorrectly on occasion.
- **Citation quality depends entirely on upstream `chunk_id` stability** — if
  retrieval or reranking ever regenerated IDs, citation resolution would break. This
  is why chunk ID stability is a hard requirement in Contract 1, not a nice-to-have.
- **Citations are intentionally lightweight** (page/section/filename fallback only) —
  no APA or bibliography-style formatting, since the task requires traceable
  provenance, not academic citation formatting.
- **An environment/dependency mismatch temporarily blocked running the newest 21
  adapter tests** (`numpy==2.5.3` pinned in `requirements.txt`, incompatible with a
  Python 3.11 environment) — flagged and being resolved rather than silently ignored.
- **Coverage validation's requirement-mapping check is exact-match by design, and
  that strictness occasionally rejects a genuinely correct answer.**
  `_validate_requirement_mappings` (`coverage_validation.py`) requires a cited claim's
  text to appear, near-verbatim, in both the synthesized answer and the coverage
  model's own quoted excerpt of it — three strings from three separate LLM
  generations. Confirmed empirically against the real pipeline: the same question,
  same code, run repeatedly, failed with `"Mapped atomic claim must appear in
  ordinary answer and excerpt"` on roughly 1 in 3 attempts purely from model sampling
  variance (e.g. "Gloamreach was founded in 246 AS." vs. "The true founding of
  Gloamreach is marked by 246 AS." — same fact, different wording). This is a
  deliberate trade-off, not a bug: the alternative (fuzzy/normalized-substring or
  token-overlap matching) was identified but deferred past submission, since loosening
  it touches the correctness guarantee the whole coverage gate exists to provide, and
  that is not a change to make under a deadline. The mitigation shipped instead is a
  retry, not a loosened check: the API layer (`_compose_with_retry` in
  `src/api/pipeline.py`) catches this exact error message (nothing broader — a real
  bug or a quota error still fails immediately) and retries only the synthesis +
  coverage step, up to 3 attempts total, without repeating the expensive retrieval
  loop. Verified against the real pipeline: 5/5 questions returned 200, with one run's
  server log showing attempt 1 fail and attempt 2 pass — 6 compose calls across 5
  questions, not 15. If all 3 attempts fail, the existing error response is returned
  unchanged (no crash, no silently degraded stub answer).
- **A naming mismatch exists between shared API health code** (expects `STRONG_MODEL`,
  `llm_available`) **and the current role-based model configuration**
  (`FAST_MODEL`/`SYNTHESIS_MODEL`/`VISION_MODEL`) — deliberately not patched inside
  Person C's own files, since that would blur ownership between answer composition and
  shared infrastructure; flagged for the infrastructure owner instead.

## Interface & Integration

- **The live research trace currently replays a finished run rather than streaming
  each round as it completes.** The planned live version needs a small callback added
  to the agent loop that hadn't landed as of this writing. `/health` honestly reports
  this as `"replayed"` rather than `"live"`, and the UI is designed to switch itself
  to the live path automatically the moment the loop exposes the needed callback — no
  change required on the interface side when that happens.
- **`retrieval_available` currently means "the module imports successfully," not
  "the index is actually built and queryable."** On a machine with packages installed
  but indexes not yet built, status would incorrectly report available. Identified as
  a smaller instance of the same "looks fine, quietly isn't" failure mode the whole
  four-flag health system was built to eliminate; the fix (probe index health, not
  importability) is small and outstanding.
- **A full four-component run (real retrieval + real loop + real composition, in the
  actual UI) had not executed even once as of this writing**, since Person C's
  composer was not yet merged. Every adjacent pair of components was individually
  driven and validated against the wire contract, and the integration seam is
  designed so nothing above it changes once the composer lands — but "designed to
  work" is explicitly distinguished from "observed to work" and is not claimed as
  equivalent to a judge.
- **Deliberate non-goals, stated plainly rather than left implicit:** no
  authentication and open CORS (this is a locally-run judged demo, not a deployed
  service); caching is keyed on the question with no invalidation (correct for a
  fixed two-day corpus, wrong for anything long-lived — bypassable via an environment
  flag, used specifically when recording the untested-question demo segment so that
  run is provably live); streaming uses one worker thread per request (sufficient for
  a single judge at a keyboard, not a concurrency design — deliberately not solved by
  pushing async complexity into the agent loop, the component actually being judged);
  the baseline returns no citations (an accurate reflection of a genuine one-shot RAG
  system that cannot know which sentence came from which source — the contrast with
  the full system's citations is the intended point).

## Known bugs found and fixed during development (kept as evidence of real testing)

- An import-path mismatch meant the real agent loop was unreachable under the actual
  server process (though reachable under the test runner), causing the system to
  silently serve fixture answers while every test still passed and `/health` reported
  status truthfully — the bug was that nothing asserted the real component was
  actually *reachable*, only that its own reporting was honest. Verified via mutation
  testing: reverting the fix left 108 of 109 tests still passing.
- A misconfigured module path for the baseline RAG combined with an undocumented
  return-type mismatch (`str` vs. an assumed dict) meant a one-line fix to the path
  alone would have converted a silent stub into a live 500 error on the baseline
  comparison toggle — caught only by testing the fix with real input rather than
  assuming it was complete.
- A merge conflict resolution regenerated `requirements.txt` from a single teammate's
  virtual environment, silently dropping another component's dependencies
  (`fastapi`, `streamlit`, `diskcache`, `pytest`) while unrelated packages survived,
  making the result look plausible. No test could catch this, since a test suite
  cannot notice a dependency it is currently running under. The fix parses actual
  source imports and compares them against the pinned list, rather than trusting any
  single environment's `pip freeze`.
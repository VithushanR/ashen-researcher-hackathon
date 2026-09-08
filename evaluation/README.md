# Offline answer evaluation

This package scores recorded `ComposedAnswer` outputs and validator results.
It does not call models, perform retrieval, or contain archive ground truth.
No official sample questions, real reference answers, or baseline pipeline are
available in this checkout. Synthetic examples live only in
`tests/evaluation_fixtures.py`; their supplied verdicts are test judgments, not
measured archive accuracy.

## Inputs

`EvaluationCase` identifies the question, explicit `data_origin` (`synthetic` or
`archive`), expected status, and optional reference expectations:

- `expected_claims`: literal answer fragments that must appear.
- `forbidden_answer_fragments`: literal text that must not appear.
- `expected_absence_claims`: absence statements required in both answer and citations.
- `expected_conflicts`: exact shared multi-claim dictionaries to preserve.
- `expected_conflict_statements`: required literal conflict presentation text.
- `expected_unresolved_claims`: B's gap descriptions to state without citing them
  as factual evidence claims.

Text comparisons are case-sensitive substring checks, with exact citation-claim
matching for expected absences and gaps. They are reproducible regression checks,
not a semantic answer-accuracy grader. Supply explicit presentation expectations
for conflicts; dictionary preservation alone cannot judge explanatory prose.

`EvaluationObservation` supplies an output (model or raw dictionary), an optional
pipeline error, and recorded validation assessments:

- `CoverageAssessment`: exact answer/citation snapshots and the existing
  `CoverageVerdict`.
- `CitationAssessment`: citation index, exact citation snapshot, and the existing
  `SemanticVerdict`, one per citation.

The evaluator reuses production `ComposedAnswer` validation and the existing
verdict models. Snapshots prevent accidentally scoring a different answer or
citation with stale verdicts. The evaluator trusts recorded validator judgments;
it does not independently verify passages, chunk IDs, or absence quotations
without the production evidence context. It does not invent successful verdicts.
Pipeline adapters must capture the actual validation results or supply explicitly
identified offline review judgments. Conflict citations undergo runtime semantic validation in attribution mode.
Their actual recorded assessments are still required for evaluation; missing
assessments remain not evaluated.

## Metrics and scoring

The machine-readable metrics are `output_contract`, `question_match`,
`answer_status`, `answer_content`, `citation_structure`, `citation_coverage`,
`groundedness`, `explicit_absence`, `conflict_handling`, and `unresolved_gaps`.

Each reports `pass`, `fail`, `not_evaluated`, or `not_applicable`, plus notes.
Coverage requires a matching snapshot and a complete verdict. Groundedness
requires aligned assessments with `support=supported`; absence compatibility
requires `explicit_absence=clear`, plus any explicit absence expectations.
These metrics are separate: local support can pass while absence compatibility
fails. Status correctness compares the expected status; it does not recalculate
B's sufficiency, confidence, or conflict resolution.

Overall `passed` requires all applicable metrics to pass and no errors.
`score` is the fraction of applicable metrics passing; `not_evaluated` remains
in its denominator, while `not_applicable` is excluded. A pipeline error always
fails the case even if other metrics pass. This score is an infrastructure health
summary, not the competition rubric or hand-marked answer accuracy.

## Runner and future comparison

`run_evaluation(cases, pipeline, pipeline_name=...)` calls the injected pipeline
with each **question only**, never its expected answers. It returns an
`EvaluationReport` with results and summary counts: total, passed, failed,
pass_rate, and per-metric status counts. Use `report.model_dump()` for Streamlit
or `report.model_dump_json(indent=2)` for JSON output. The runner does not write
files automatically. Empty runs report zero counts. Pipeline exceptions become
failed rows and do not stop later cases or disappear from the denominator.

Case IDs must be unique. Run synthetic and archive cases separately; mixed-origin
reports are rejected. For baseline versus full comparison, run the same versioned
case list twice with separate named pipeline adapters and join results by case ID.
Neither real adapter nor live model wiring is implemented in this milestone.

Remaining work includes the official 20 questions, archive-grounded reference
answers/evidence, team-written hard questions, real validation-result capture,
baseline/full adapters, and hand-marked Correct / Partly Correct / Wrong accuracy.
Retrieval recall/precision, failure-category analysis, and measured limitations
remain later evaluation work. Iteration counts already remain in the pipeline's
`ComposedAnswer`; this initial summary does not aggregate retrieval/runtime data.


## Offline visual factual accuracy

Runtime vision output is evidence, never evaluation ground truth. Optional
`EvaluationCase.visual_expectations` contains `verified_by`, `reference_notes`,
and a nonempty `facts` list. Each fact has a unique `id`, nonempty human-approved
`accepted_statements`, optional `forbidden_statements`, and optional
`expected_image_filenames`. Reviewer/reference fields must be nonblank. They
record review provenance but cannot prove that a human actually reviewed an image.
Use complete, image-specific assertions, not bare keywords. Actual archive
expectations must be supplied and verified by people; the test reviewer metadata
and dog/crow examples are explicitly synthetic, not archive annotations.

`visual_factual_accuracy` case-folds text and collapses whitespace. Every fact
requires an accepted assertion as an answer substring, and any configured
forbidden assertion fails the metric. `visual_source_citation` requires each
configured filename to occur on a citation whose normalized claim exactly matches
an accepted assertion for that fact. Filename comparison is exact: no path,
chunk-ID, or model-name inference. Lists are all-required; additional citations
are allowed. A correct filename attached only to an unrelated claim cannot pass.

Without visual expectations both metrics are `not_evaluated` and excluded from
score/pass calculation. With expectations, factual accuracy is required; source
citation is required only when filenames are configured. A required failed or
unevaluated visual check prevents overall success. Other missing assessments
retain their existing fail-closed behavior. Successful legacy cases were NOT
automatically visually evaluated. Reports expose both new metric status counts.

These are deterministic literal rubric checks, not unrestricted semantic accuracy.
Case and whitespace vary safely; paraphrases need explicit accepted alternatives.
Negation, quotes, and contradictory prose can fool substring matching; forbidden
assertions only catch configured errors. Unlisted facts are not evaluated, and
filenames do not authenticate image bytes. No pixel inspection, runtime vision
verification, provider, or model call is performed. Human review remains necessary
for free-form answers and reference quality. The public ComposedAnswer is unchanged.

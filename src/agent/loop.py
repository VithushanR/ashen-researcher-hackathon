"""
The agent loop — wires router, planner, evidence analysis, conflict
detection/resolution, and the sufficiency checker together into the
single research(question) -> ResearchState function Person C depends on.

Loop shape (architecture diagram):
    search -> evidence analyzer -> conflict detector -> sufficiency check
    -> if insufficient: next-search planner -> repeat
    -> if sufficient or cap hit: return

Stop conditions, both enforced in code, not merely requested in a
prompt (spec §2.2):
  - hard iteration cap (max_iter)
  - early stop if two consecutive rounds add no new evidence

route is NOT predicted before the loop runs — it's derived afterward
from how many iterations it actually took (see router.derive_route),
since a sufficiency-driven loop naturally distinguishes simple from
multi-hop questions without needing to guess in advance.

IMPORT PATH NOTE: fixtures/ lives at the repo root, a sibling of src/,
not nested inside src/agent/ — import accordingly.
"""
from agent.state import ResearchState, Evidence, TraceStep
from agent.router import derive_route
from agent.planner import derive_required_claims, plan_first_query, plan_next_query
from agent.evidence import analyze_evidence
from agent.sufficiency import check_sufficiency
from agent.conflict import detect_conflicts, resolve_conflicts
from agent.vision import describe_image
from fixtures.fake_hybrid_search import hybrid_search as fake_hybrid_search


def research(
    question: str,
    search_fn=fake_hybrid_search,
    max_iter: int = 5,
) -> ResearchState:
    """
    Run the full research loop for a single question and return the
    final ResearchState.

    search_fn defaults to the fake stand-in — swap this for the real
    Person A hybrid_search() the moment it's handed off. Because the
    contract (Contract 2) hasn't changed, this should be close to a
    one-line edit at the call site, not a change to this function.
    """
    state = ResearchState(question=question)
    state.required_claims = derive_required_claims(question)
    query = plan_first_query(question)
    stale_rounds = 0

    while state.iteration < max_iter:
        if query in state.search_history:
            # Would repeat an already-issued query — nothing new to gain.
            state.unresolved_claims.append(
                "Stopped early: next query would repeat a previous search."
            )
            break

        raw_chunks = search_fn(query, k=8)
        new_evidence = analyze_evidence(raw_chunks, state)
        added = len(new_evidence)
        state.evidence.extend(new_evidence)
        state.search_history.append(query)
        state.iteration += 1

        # Conflict detection re-scans all evidence collected so far each
        # round, so we replace (not append to) state.conflicts each time
        # rather than risk duplicate entries for the same attribute.
        raw_conflicts = detect_conflicts(state.evidence)
        if raw_conflicts:
            state.conflicts = resolve_conflicts(raw_conflicts, state.evidence)

        verdict = check_sufficiency(state)
        state.trace.append(
            TraceStep(step=state.iteration, query=query, verdict=verdict.verdict, missing=verdict.missing_info)
        )

        if verdict.verdict == "sufficient":
            state.confidence = _compute_confidence(state, capped=False)
            state.route = derive_route(state.iteration)
            return state

        # Vision fallback (spec §2.15). Per the finalized team contract,
        # this is INTERNAL loop behavior, NOT a sufficiency verdict — the
        # checker returns an ordinary "insufficient", and the loop itself
        # decides whether the gap depends on an unreadable image.
        #
        # Trigger: the checker said insufficient AND the evidence contains
        # a relevant content_type == "image" chunk not yet described. If
        # so, look at the image directly, append its description as a new
        # Evidence item, and continue the loop normally — the next
        # sufficiency check sees it like any other evidence. Each image is
        # described at most once (tracked via a synthetic search_history
        # marker) so a persistently-unreadable image can't spin forever.
        if verdict.verdict == "insufficient" and _has_undescribed_image(state):
            described = _run_vision_fallback(state)
            added = added + described  # count vision evidence as progress
            if described > 0:
                stale_rounds = 0
                # Re-search on the same gap next round; the freshly added
                # visual evidence will now be in the state for the checker.
                query = plan_next_query(
                    verdict.missing_info or state.question,
                    state.discovered_entities,
                    state.search_history,
                )
                continue
            # described == 0 → fall through to the normal stale/cap logic
            # below, exactly like any other round that added nothing.

        # Early stop: two consecutive rounds with no new evidence.
        stale_rounds = stale_rounds + 1 if added == 0 else 0
        if stale_rounds >= 2:
            if verdict.missing_info:
                state.unresolved_claims.append(verdict.missing_info)
            state.confidence = _compute_confidence(state, capped=True)
            state.route = derive_route(state.iteration)
            _ensure_terminal_gap(state)
            return state

        # Next query: if a conflict was just flagged, prioritise a
        # targeted verification search for it (resolution policy Step 5);
        # otherwise search for whatever the sufficiency checker said is
        # missing.
        if verdict.verdict == "conflict_detected" and state.conflicts:
            unresolved = next(
                (c for c in state.conflicts if c.resolved_value is None), None
            )
            gap = f"verification of disputed {unresolved.attribute}" if unresolved else verdict.missing_info
            query = plan_next_query(
                gap or verdict.missing_info or "",
                state.discovered_entities,
                state.search_history,
            )
        else:
            query = plan_next_query(
                verdict.missing_info or "",
                state.discovered_entities,
                state.search_history,
            )

    # Cap reached without a "sufficient" or early-stop break.
    if state.trace and state.trace[-1].missing:
        state.unresolved_claims.append(state.trace[-1].missing)
    state.confidence = _compute_confidence(state, capped=True)
    state.route = derive_route(state.iteration)
    _ensure_terminal_gap(state)
    return state


def _ensure_terminal_gap(state: ResearchState) -> None:
    """A non-success exit must remain visibly incomplete to downstream consumers."""
    if not state.unresolved_claims and not any(c.resolved_value is None for c in state.conflicts):
        state.unresolved_claims.append("Research stopped before sufficiency was established.")


def _has_undescribed_image(state: ResearchState) -> bool:
    """
    True if the evidence contains an image chunk (content_type == "image")
    that hasn't yet been sent through the vision fallback this run.

    "Undescribed" is tracked by the same synthetic search_history marker
    _run_vision_fallback writes, so an image is only ever a trigger once —
    after it's been looked at, its description (or its "nothing relevant")
    is already in the state and it won't re-fire.
    """
    for item in state.evidence:
        if item.content_type == "image":
            marker = f"__vision__:{item.chunk_id}"
            if marker not in state.search_history:
                return True
    return False


def _run_vision_fallback(state: ResearchState, vision_fn=None) -> int:
    """
    Find image chunks in the current evidence that haven't yet been
    described visually, send each to the vision model, and append the
    description back into state.evidence as a new Evidence item that
    conforms to the finalized team contract (docs: consolidated Evidence
    contract + vision implementation guide).

    Returns the number of new visual-evidence items added. A synthetic
    marker is written into search_history per image so the same image is
    never described twice — that's what stops a stubbornly-unreadable
    image from looping forever.

    vision_fn is injectable so tests can substitute a fake describer and
    never make a real API call. Resolved at call time (not bound as a
    default) so monkeypatching agent.loop.describe_image is honoured.
    """
    if vision_fn is None:
        vision_fn = describe_image
    added = 0
    for item in list(state.evidence):
        if item.content_type != "image":
            continue
        marker = f"__vision__:{item.chunk_id}"
        if marker in state.search_history:
            continue  # already looked at this image

        state.search_history.append(marker)
        description = vision_fn(item.filename, state.question)

        if not description or description.strip().upper() == "NOTHING RELEVANT":
            continue  # image had nothing useful — don't add empty evidence

        # Wrap the vision output as a new Evidence item per the finalized
        # contract. Preserve document_id, filename, page, section, entities,
        # reliability from the source image; change only chunk_id,
        # source_type, content_type, and text. filename stays the ORIGINAL
        # image so Person C's citation points at the real artifact, never
        # at the vision model.
        state.evidence.append(Evidence(
            chunk_id=f"{item.chunk_id}_vision_{state.iteration}",
            document_id=item.document_id,
            filename=item.filename,
            source_type="image_derived",
            reliability=item.reliability,
            page=item.page,
            section=item.section,
            content_type="vision_description",
            text=description.strip(),
            entities=item.entities,
        ))
        added += 1

    return added


def _compute_confidence(state: ResearchState, capped: bool) -> int:
    """
    Placeholder confidence scoring — deliberately simple for now.

    NOT self-rated by the LLM (spec §2.3) — derived from the loop's own
    outcome: whether it terminated on "sufficient" vs. hit the cap, and
    whether any conflicts were left unresolved. Revisit this once real
    evidence is available to see whether a richer rubric-based score is
    worth the added complexity.
    """
    if capped:
        base = 40
    else:
        base = 90

    unresolved_conflicts = sum(1 for c in state.conflicts if c.resolved_value is None)
    base -= unresolved_conflicts * 10

    return max(0, min(100, base))

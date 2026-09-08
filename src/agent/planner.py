"""
Query planning — turns a question (and later, a sufficiency gap) into
an actual search string.

Two responsibilities:
  1. plan_first_query — the very first search, built from the raw question
  2. plan_next_query  — every subsequent search, built from what the
                        sufficiency checker said was missing, plus any
                        newly discovered entities not already covered
                        by that missing_info string

This is what produces genuine multi-hop behavior (spec §2.5): each new
query is built from newly discovered information, not a rephrasing of
the original question.

Note on scope (deliberately NOT handled here):
  - Descriptive queries ("the commander of the western chapter") are left
    as-is and passed straight through to hybrid_search(). Matching a
    description to the right entity is a retrieval-quality problem
    (vector search + reranking), not a query-planning problem — handled
    by Person A's hybrid_search(), not here.
  - False-premise questions (e.g. asking why someone became king when
    they never did) are not specially detected. The loop will just run
    to its iteration cap and produce an honest partial answer, per the
    existing capped/unresolved path. See docs/limitations.md.

KNOWN GAP: this file does not yet populate ResearchState.required_claims
(spec §3.6 step 3 — decomposing the question into required claims before
the first query). To be added: either plan_first_query returns a tuple
of (query, required_claims), or a separate small function derives
required_claims from the question. Not yet implemented as of this
compilation.
"""
from agent.aliases import get_aliases


def plan_first_query(question: str) -> str:
    """
    Build the first search query from the raw question.

    Expanded with any known aliases for names/titles mentioned in the
    question, so a query for "Isolde Mournvale" also benefits if the
    corpus refers to her elsewhere as "the Hollow Blade".

    get_aliases() is currently a stub returning [] until Person A's
    alias glossary is ready — this function doesn't need to change when
    that happens, only aliases.py does.

    Known limitation: alias lookup is done word-by-word, not phrase-by-
    phrase, so multi-word names (e.g. "Isolde Mournvale") won't match
    a glossary entry keyed on the full name until this is revisited
    once the real glossary format is confirmed with Person A.
    """
    expansions: list[str] = []
    for word in question.replace("?", "").split():
        aliases = get_aliases(word)
        expansions.extend(aliases)

    if expansions:
        return question + " " + " ".join(expansions)
    return question


def plan_next_query(
    missing_info: str,
    discovered_entities: list[str],
    search_history: list[str] | None = None,
) -> str:
    """
    Build the next search query from the sufficiency checker's stated
    gap, optionally expanded with a newly discovered entity that we have
    NOT already searched on.

    The gap description (missing_info) is the primary driver — that's
    what the sufficiency checker said is still missing, so it's what the
    next search should target.

    BUG FIX (multi-hop chain): the previous version appended *every*
    discovered entity that wasn't literally spelled out in missing_info.
    That re-injected entities we had already searched and already have
    evidence for (e.g. re-adding "Isolde Mournvale" after we'd found her
    page), which made the retriever return the SAME chunk again — the
    loop collected one piece of evidence and stalled instead of chaining
    to the next hop. A multi-hop query must be built from what we still
    NEED, not from what we already have.

    The fix: only consider entities we have not already issued a search
    that mentioned them (tracked via search_history), and add at most
    one — the most recently discovered such entity — as a lead to
    follow. If every discovered entity has already been searched, the
    gap description stands on its own.

    Example:
        missing_info = "Which war the Ashen Order won."
        discovered_entities = ["Isolde Mournvale", "Ashen Order"]
        search_history = ["...Isolde Mournvale..."]
        -> "Which war the Ashen Order won."
           ("Ashen Order" is already in the gap text; "Isolde Mournvale"
            was already searched — so nothing stale is re-added.)
    """
    if not missing_info:
        # Defensive fallback — the sufficiency checker should never
        # return "insufficient" with no missing_info, but if it does,
        # search on discovered entities alone rather than crashing or
        # blindly repeating the previous query.
        return " ".join(discovered_entities) if discovered_entities else ""

    history_text = " ".join(search_history or []).lower()

    # A candidate entity is worth adding only if it is genuinely new: not
    # already named in the gap text, AND not already the subject of a
    # previous search. Otherwise adding it just re-fetches old evidence.
    fresh_entities = [
        entity for entity in discovered_entities
        if entity.lower() not in missing_info.lower()
        and entity.lower() not in history_text
    ]

    if fresh_entities:
        # Add only the most recently discovered fresh entity — the newest
        # thread to pull — rather than piling on every past entity.
        return missing_info + " " + fresh_entities[-1]
    return missing_info

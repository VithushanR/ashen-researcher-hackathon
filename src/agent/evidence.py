"""
Evidence analysis — takes raw chunks back from hybrid_search() and turns
them into Evidence objects added to the ResearchState, deduped against
what's already been collected, with newly seen entities tracked.

This sits between "search returned some chunks" and "sufficiency checker
looks at the state" in the loop (architecture diagram's "EVIDENCE
ANALYZER" box).
"""
from agent.state import Evidence, ResearchState


def analyze_evidence(raw_chunks: list[dict], state: ResearchState) -> list[Evidence]:
    """
    Convert raw chunk dicts (Contract 1 format) into Evidence objects,
    skipping any chunk already present in state.evidence (deduped by
    chunk_id), and update state.discovered_entities in place with any
    new entity names found on the new chunks.

    Malformed chunks missing a chunk_id are skipped rather than let
    through with a null ID — chunk_id is mandatory under Contract 1,
    so a missing one means the search layer returned something broken,
    not something this function should silently tolerate.

    Returns only the newly added Evidence objects (not the full
    accumulated list) — the caller (loop.py) is responsible for
    extending state.evidence with the return value.
    """
    existing_ids = {item.chunk_id for item in state.evidence}
    new_evidence: list[Evidence] = []

    for chunk in raw_chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue  # malformed chunk (missing chunk_id) — skip, don't corrupt state
        if chunk_id in existing_ids:
            continue  # already have this chunk, skip it

        evidence_item = Evidence(
            chunk_id=chunk_id,
            document_id=chunk.get("document_id"),
            filename=chunk["filename"],
            source_type=chunk["source_type"],
            reliability=chunk["reliability"],
            page=chunk.get("page"),
            section=chunk.get("section"),
            content_type=chunk.get("content_type", "text"),
            text=chunk["text"],
            entities=chunk.get("entities", []),
        )
        new_evidence.append(evidence_item)
        existing_ids.add(chunk_id)

        for entity in evidence_item.entities:
            if entity not in state.discovered_entities:
                state.discovered_entities.append(entity)

    return new_evidence

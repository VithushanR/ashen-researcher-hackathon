"""
Tests for the agent loop, exercised against the three fixture scenarios
described in fake_research_states.json: a clean multi-hop success, a
conflict that gets resolved, and a run that hits the iteration cap with
an honest partial answer.

call_llm is monkeypatched wherever it's imported (sufficiency.py and
conflict.py each import it directly), so these tests never make a real
API call — they're fast, deterministic, and don't burn API quota.

Requires a pytest.ini at the repo root with:

    [pytest]
    pythonpath = src

so that "agent.*" imports resolve correctly when tests are run from
the repo root.
"""
import json

import pytest

from agent.loop import research


def _sufficiency_response(coverage, agreement, missing_info, verdict):
    return json.dumps(
        {
            "coverage": coverage,
            "agreement": agreement,
            "missing_info": missing_info,
            "verdict": verdict,
        }
    )


def _no_conflicts_response():
    return json.dumps({"conflicts_found": False, "conflicts": []})


# ---------------------------------------------------------------------
# Case 1 — clean multi-hop success (mirrors fake_research_states.json case 1)
# ---------------------------------------------------------------------
def test_clean_multihop_success(monkeypatch):
    def fake_search(query, k=8):
        if "Isolde" in query:
            return [{
                "chunk_id": "wiki_014_s2",
                "document_id": "wiki_014",
                "filename": "isolde_mournvale.md",
                "source_type": "wiki",
                "reliability": "T2_curated",
                "page": None,
                "section": "Affiliations",
                "text": "Isolde Mournvale has been a sworn member of the Ashen Order since 341 AS.",
                "entities": ["Isolde Mournvale", "Ashen Order"],
            }]
        return [{
            "chunk_id": "codex_02_p47_c3",
            "document_id": "codex_02",
            "filename": "annals_of_the_ashen_era_vol2.pdf",
            "source_type": "codex",
            "reliability": "T1_authoritative",
            "page": 47,
            "section": "Military Campaigns",
            "text": "The Ashen Order emerged victorious in the War of Drowned Light in 358 AS.",
            "entities": ["Ashen Order", "War of Drowned Light"],
        }]

    sufficiency_calls = iter([
        _sufficiency_response("partial", "yes", "Which war the Ashen Order won.", "insufficient"),
        _sufficiency_response("yes", "yes", None, "sufficient"),
    ])
    monkeypatch.setattr("agent.sufficiency.call_llm", lambda prompt: next(sufficiency_calls))
    monkeypatch.setattr("agent.conflict.call_llm", lambda prompt: _no_conflicts_response())

    state = research(
        "Which war was won by the organization that included Isolde Mournvale as one of its members?",
        search_fn=fake_search,
    )

    assert state.iteration == 2
    assert state.route == "multihop"
    assert len(state.evidence) == 2
    assert state.conflicts == []
    assert state.confidence > 0
    assert state.trace[-1].verdict == "sufficient"


# ---------------------------------------------------------------------
# Case 2 — conflict detected and resolved (mirrors fixture case 2)
# ---------------------------------------------------------------------
def test_conflict_detected_and_resolved(monkeypatch):
    def fake_search(query, k=8):
        return [
            {
                "chunk_id": "ephemera_091_p2",
                "document_id": "ephemera_091",
                "filename": "letter_gravemaw_sighting.txt",
                "source_type": "ephemera",
                "reliability": "T4_unverified",
                "page": 2,
                "section": None,
                "text": "They say the Gauntlet of Sorrowfell was forged in the winter of 312 AS.",
                "entities": ["Gauntlet of Sorrowfell"],
            },
            {
                "chunk_id": "codex_01_p112_table4",
                "document_id": "codex_01",
                "filename": "annals_of_the_ashen_era_vol1.pdf",
                "source_type": "codex",
                "reliability": "T1_authoritative",
                "page": 112,
                "section": "Figure Plate 4",
                "text": "Artifact: Gauntlet of Sorrowfell | Forged: 341 AS | Forgemaster: Kellan Ashgrove",
                "entities": ["Gauntlet of Sorrowfell", "Kellan Ashgrove"],
            },
        ]

    conflict_response = json.dumps({
        "conflicts_found": True,
        "conflicts": [{
            "attribute": "year forged",
            "claims": [
                {"claim": "Forged in 312 AS (approx.)", "chunk_id": "ephemera_091_p2"},
                {"claim": "Forged in 341 AS", "chunk_id": "codex_01_p112_table4"},
            ],
        }],
    })

    sufficiency_calls = iter([
        _sufficiency_response("no", "conflict", "Resolve disagreement on forging year.", "conflict_detected"),
        _sufficiency_response("yes", "yes", None, "sufficient"),
    ])
    monkeypatch.setattr("agent.sufficiency.call_llm", lambda prompt: next(sufficiency_calls))
    monkeypatch.setattr("agent.conflict.call_llm", lambda prompt: conflict_response)

    state = research(
        "In which year was the Gauntlet of Sorrowfell actually forged?",
        search_fn=fake_search,
    )

    assert len(state.conflicts) == 1
    conflict = state.conflicts[0]
    assert conflict.resolved_value == "Forged in 341 AS"  # T1 codex should win over T4 ephemera
    assert "reliability tier" in conflict.resolution
    assert state.trace[0].verdict == "conflict_detected"


# ---------------------------------------------------------------------
# Case 3 — iteration cap reached, honest partial answer (mirrors fixture case 3)
# ---------------------------------------------------------------------
def test_capped_partial_answer(monkeypatch):
    def fake_search(query, k=8):
        return [{
            "chunk_id": "ephemera_091_p2",
            "document_id": "ephemera_091",
            "filename": "letter_gravemaw_sighting.txt",
            "source_type": "ephemera",
            "reliability": "T4_unverified",
            "page": 2,
            "section": None,
            "text": "The Gravemaw Wyrm was last sighted deep in the Hollow Vale.",
            "entities": ["Hollow Vale"],
        }]

    always_insufficient = lambda prompt: _sufficiency_response(
        "no", "yes", "Which house holds dominion over the Hollow Vale.", "insufficient"
    )
    monkeypatch.setattr("agent.sufficiency.call_llm", always_insufficient)
    monkeypatch.setattr("agent.conflict.call_llm", lambda prompt: _no_conflicts_response())

    state = research(
        "Whose dominion encompasses the lair of the Gravemaw Wyrm?",
        search_fn=fake_search,
        max_iter=5,
    )

    # Same chunk_id every time -> analyze_evidence dedupes it after round 1,
    # so "added" evidence is 0 from round 2 onward -> early stop after 2
    # stale rounds, well before the max_iter=5 cap.
    assert state.iteration <= 5
    assert len(state.unresolved_claims) >= 1
    assert state.confidence < 90  # capped path should score lower than a clean success
    assert state.evidence[0].reliability == "T4_unverified"


# ---------------------------------------------------------------------
# Case 4 — vision fallback (spec §2.15): retrieval surfaces only an
# unreadable image chunk, loop looks at the image, feeds description back
# ---------------------------------------------------------------------
def test_vision_fallback_on_image_only_evidence(monkeypatch):
    def fake_search(query, k=8):
        return [{
            "chunk_id": "img_042",
            "document_id": "images",
            "filename": "banner_house_sorrowfell.png",
            "source_type": "wiki",
            "reliability": "T2_curated",
            "page": None,
            "section": None,
            "content_type": "image",
            "text": "Figure plate: banner of an unnamed house.",  # thin caption, no visual fact
            "entities": ["House Sorrowfell"],
        }]

    # Round 1 -> only an image chunk, caption doesn't answer -> insufficient
    #   (the loop itself detects the undescribed image and runs vision)
    # Round 2 -> vision description now in evidence -> sufficient
    sufficiency_calls = iter([
        _sufficiency_response("no", "yes", "What emblem is on the banner.", "insufficient"),
        _sufficiency_response("yes", "yes", None, "sufficient"),
    ])
    monkeypatch.setattr("agent.sufficiency.call_llm", lambda prompt: next(sufficiency_calls))
    monkeypatch.setattr("agent.conflict.call_llm", lambda prompt: _no_conflicts_response())

    # Fake vision: never touches the network or the filesystem.
    def fake_vision(filename, question):
        assert filename == "banner_house_sorrowfell.png"
        return "A black raven clutching a broken sword on a grey field."
    monkeypatch.setattr("agent.loop.describe_image", fake_vision)

    state = research(
        "What emblem is shown on the banner of House Sorrowfell?",
        search_fn=fake_search,
    )

    # The visual description should be a new evidence item conforming to
    # the finalized contract: source_type image_derived, content_type
    # vision_description, chunk_id suffixed _vision_<iteration>, and the
    # ORIGINAL image filename preserved for citation.
    visual_items = [e for e in state.evidence if e.source_type == "image_derived"]
    assert len(visual_items) == 1
    v = visual_items[0]
    assert "black raven" in v.text
    assert v.content_type == "vision_description"
    assert v.chunk_id.startswith("img_042_vision_")
    assert v.filename == "banner_house_sorrowfell.png"  # cites the real image, not the model
    assert v.reliability == "T2_curated"                # inherited from source image
    assert state.trace[-1].verdict == "sufficient"


def test_vision_fallback_stops_when_image_has_nothing(monkeypatch):
    """An image the vision model finds irrelevant must not loop forever."""
    def fake_search(query, k=8):
        return [{
            "chunk_id": "img_099",
            "document_id": "images",
            "filename": "blank_plate.png",
            "source_type": "ephemera",
            "reliability": "T4_unverified",
            "page": None,
            "section": None,
            "content_type": "image",
            "text": "Damaged plate, mostly illegible.",
            "entities": [],
        }]

    monkeypatch.setattr(
        "agent.sufficiency.call_llm",
        lambda prompt: _sufficiency_response("no", "yes", "The emblem.", "insufficient"),
    )
    monkeypatch.setattr("agent.conflict.call_llm", lambda prompt: _no_conflicts_response())
    monkeypatch.setattr("agent.loop.describe_image", lambda f, q: "NOTHING RELEVANT")

    state = research("What is the emblem?", search_fn=fake_search, max_iter=5)

    # No visual evidence added, loop terminates via stale-round early stop.
    assert not any(e.source_type == "image_derived" for e in state.evidence)
    assert state.iteration <= 5
    assert state.confidence < 90

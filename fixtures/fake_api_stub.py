"""FAKE STAND-IN /ask endpoint for Person D.

Per the Execution & Collaboration Plan, nobody waits for anyone. Person D's
Streamlit UI is built against this stub from Day 1 and swaps to the real
endpoint once Person B's ``research()`` and Person C's ``compose_answer()`` are
handed off. Because the response shape does not change at that point, the swap
touches the API wiring only -- not the UI.

Keep this file in the repo even after the real pipeline is wired in. It is
useful for pytest later (fixed, known inputs), and it is honest evidence in the
commit history that the team built against contracts and stubs from day one
rather than sequentially.

Run it standalone:

    uvicorn fixtures.fake_api_stub:app --reload --port 8001

Then:

    curl -X POST http://127.0.0.1:8001/ask ^
         -H "Content-Type: application/json" ^
         -d "{\"question\": \"anything\"}"
"""

import json
import random
from pathlib import Path

from fastapi import FastAPI

app = FastAPI(
    title="Ashen Researcher - fake /ask stub",
    description="Contract-first stand-in for the real pipeline. Returns canned answers.",
)

_FIXTURE_PATH = Path(__file__).parent / "fake_api_response.json"
_FAKES = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

# Order matters only for readability; the endpoint picks at random.
_EXAMPLES = [
    _FAKES["clean_answer_example"],
    _FAKES["conflict_answer_example"],
    _FAKES["partial_answer_example"],
]


@app.post("/ask")
def ask(payload: dict) -> dict:
    """Return one of the three example responses at random.

    Deliberately ignores the question that was actually asked. Randomising the
    case is the point: it forces the UI to be built for all three states
    (clean / conflict / partial) instead of only the one that happens to be
    convenient. A UI that has never rendered a conflict callout will not grow
    one on integration day.

    The real version, once Persons B and C hand off, becomes:

        question = payload["question"]
        state = research(question)          # Person B
        result = compose_answer(state)      # Person C
        return result
    """
    example = dict(random.choice(_EXAMPLES))
    # Echo back the real question so the UI header is not obviously canned.
    example["question"] = payload.get("question", example["question"])
    return example


@app.get("/health")
def health() -> dict:
    """Let the UI tell that it is talking to the stub, not the real pipeline."""
    return {"status": "ok", "pipeline": "stub", "cases": len(_EXAMPLES)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)

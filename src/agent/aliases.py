"""
Alias lookup — stub version.

Real implementation and the underlying glossary are Person A's
responsibility (Execution Plan, Person A, Phase 3, step 10:
build_alias_glossary()). This stub exists so planner.py can be built
and tested today without waiting on Person A's ingestion pipeline.

Swap point: once Person A hands off the real alias glossary, replace
the body of get_aliases() to read from it. The function signature
should not need to change — confirm this with Person A before they
build the real version.
"""


def get_aliases(name: str) -> list[str]:
    """
    Fake stand-in. Always returns an empty list until Person A's real
    alias glossary is ready.

    Real version (once handed off) will look something like:

        _GLOSSARY = json.loads(Path("alias_glossary.json").read_text())

        def get_aliases(name: str) -> list[str]:
            return _GLOSSARY.get(name, [])
    """
    return []

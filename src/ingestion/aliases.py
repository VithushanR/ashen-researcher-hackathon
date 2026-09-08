"""
Builds an alias glossary ({canonical name: [aliases]}) from wiki chunks.

Two independent signals feed the glossary, both scoped per wiki document
(chunks sharing a document_id, i.e. the same source .md file):

1. Explicit pattern match: "also known as X" / "also called X" in the
   prose. Since parse_markdown/chunk_text never strip wikilink or heading
   markup out of chunk text, this regex runs directly against the final
   chunk "text" field.

2. Wikilink frequency: [[Name]] tokens (chunk text still carries the raw
   [[...]] syntax, same reason as above) that recur often within a single
   document are treated as alias candidates for that document's subject —
   the intuition being that a page repeatedly wikilinking one term is
   very likely using it as another name for itself, not just referencing
   an unrelated page once or twice.

Neither ingestion nor chunking performs entity/NER extraction (out of
scope per the task list), so there's no reliable "this document's subject
is X" field to read. The wiki corpus is assumed to name each file after
its subject (e.g. isolde_mournvale.md -> "Isolde Mournvale"), so the
canonical name is derived by humanizing document_id. This is a
heuristic, flagged here rather than silently assumed.
"""

import re
from collections import Counter, defaultdict

_ALIAS_PATTERN_RE = re.compile(r"also (?:known|called) as ([^.,;\n]+)", re.IGNORECASE)
_WIKILINK_TOKEN_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")

WIKILINK_ALIAS_MIN_COUNT = 3  # recurrences within one document to count as an alias signal


def _humanize(document_id: str) -> str:
    return document_id.replace("_", " ").replace("-", " ").strip().title()


def build_alias_glossary(chunks: list[dict]) -> dict[str, list[str]]:
    """
    Build a {canonical_name: [aliases]} glossary from wiki text chunks.

    Args:
        chunks: The full list of final locked-contract chunks (only
            source_type == "wiki" text chunks are used).

    Returns:
        Dict mapping each wiki document's canonical subject name to a
        sorted list of alias strings. Documents with no detected aliases
        are omitted.
    """
    chunks_by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("source_type") != "wiki" or chunk.get("content_type") != "text":
            continue
        document_id = chunk.get("document_id")
        if document_id:
            chunks_by_document[document_id].append(chunk)

    glossary: dict[str, set[str]] = defaultdict(set)

    for document_id, document_chunks in chunks_by_document.items():
        subject = _humanize(document_id)

        for chunk in document_chunks:
            for match in _ALIAS_PATTERN_RE.finditer(chunk["text"]):
                alias = match.group(1).strip().rstrip(".,;")
                if alias and alias.lower() != subject.lower():
                    glossary[subject].add(alias)

        wikilink_counts: Counter = Counter()
        for chunk in document_chunks:
            for token in _WIKILINK_TOKEN_RE.findall(chunk["text"]):
                wikilink_counts[token.strip()] += 1

        for token, count in wikilink_counts.items():
            if count >= WIKILINK_ALIAS_MIN_COUNT and token.lower() != subject.lower():
                glossary[subject].add(token)

    return {subject: sorted(aliases) for subject, aliases in glossary.items() if aliases}

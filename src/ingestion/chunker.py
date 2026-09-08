"""
Splits prose text into word-bounded chunks for embedding/retrieval.

Sentence boundaries are the hard constraint (a chunk boundary must never
land mid-sentence); paragraph boundaries are a soft preference used to
pick *where* within the [min_words, max_words] window a chunk ends, once
that window has been reached.

Tables and images are never passed through this module — they're already
atomic single-chunk units (see parse_codex_tables.py) and go straight to
metadata.attach_metadata unchanged, regardless of size.
"""

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n+")


def _split_into_sentences_with_paragraph_flags(text: str) -> list[tuple[str, bool]]:
    """
    Split text into sentences, flagging the last sentence of each
    paragraph. Paragraphs are delimited by one or more newlines, which
    covers both parse_docx's one-paragraph-per-line output and
    parse_markdown's blank-line-separated prose.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    sentences: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        flattened = re.sub(r"\s+", " ", paragraph).strip()
        paragraph_sentences = [
            s.strip() for s in _SENTENCE_SPLIT_RE.split(flattened) if s.strip()
        ]
        if not paragraph_sentences:
            continue
        for sentence in paragraph_sentences[:-1]:
            sentences.append((sentence, False))
        sentences.append((paragraph_sentences[-1], True))

    return sentences


def chunk_text(
    text: str,
    min_words: int = 400,
    max_words: int = 600,
    overlap_words: int = 50,
) -> list[str]:
    """
    Split text into chunks of roughly min_words-max_words, overlapping by
    overlap_words. Never splits mid-sentence; prefers ending a chunk at a
    paragraph boundary once min_words has been reached.

    Args:
        text: The text to chunk.
        min_words: Minimum words per chunk (soft floor before a
            paragraph-boundary stop is taken).
        max_words: Maximum words per chunk (hard ceiling; a chunk is only
            allowed past this if a single sentence alone exceeds it).
        overlap_words: Approximate number of trailing words from one
            chunk to repeat at the start of the next.

    Returns:
        List of chunk strings, in order. Empty list for empty/whitespace
        -only input.
    """
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences_with_paragraph_flags(text)
    if not sentences:
        return []

    word_counts = [len(sentence.split()) for sentence, _ in sentences]
    n = len(sentences)

    chunks: list[str] = []
    start = 0
    while start < n:
        word_count = 0
        end = start
        while end < n:
            sentence_words = word_counts[end]
            if word_count + sentence_words > max_words and word_count >= min_words:
                break
            word_count += sentence_words
            is_paragraph_end = sentences[end][1]
            end += 1
            if is_paragraph_end and word_count >= min_words:
                break
            if word_count >= max_words:
                break

        if end == start:
            # A single sentence alone exceeds max_words; keep it whole
            # rather than cutting mid-sentence.
            end = start + 1

        chunk_sentences = [sentence for sentence, _ in sentences[start:end]]
        chunks.append(" ".join(chunk_sentences))

        if end >= n:
            break

        # Walk backward from the chunk's end to find where overlap_words
        # worth of trailing sentences begins.
        overlap_word_total = 0
        overlap_start = end
        while overlap_start > start:
            overlap_word_total += word_counts[overlap_start - 1]
            if overlap_word_total >= overlap_words:
                break
            overlap_start -= 1

        # If the whole chunk would be needed for overlap, don't repeat it
        # entirely — that would stall progress and duplicate the chunk.
        start = overlap_start if overlap_start > start else end

    return chunks

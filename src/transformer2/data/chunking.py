"""
Corpus-agnostic chunking for language-model training data.

Splits and packs text so that every chunk fits a token budget without ending
mid-word. Shared by the Reddit and TinyStories postprocessors; the corpus-specific
cleaning lives alongside each of those.

Every function takes a `lengths_fn` that maps a list of strings to their token
counts, so the caller controls which tokenizer defines "a token".
"""

import re

SEPARATOR = '\n\n'
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_oversized(text, lengths_fn, max_tokens):
    """
    Break text that exceeds max_tokens into pieces that fit.

    Falls back through sentence boundaries to word boundaries, so a piece never ends
    mid-word.

    Args:
        text (str): Text longer than max_tokens
        lengths_fn (callable): Maps a list of strings to their token counts
        max_tokens (int): Budget per piece

    Returns:
        list of str: Pieces
    """
    units = [u for u in SENTENCE_SPLIT_RE.split(text) if u]
    if len(units) <= 1:
        units = [u for u in text.split(' ') if u]
    if not units:
        return [text]

    unit_lengths = lengths_fn(units)
    # Joining costs roughly a token; reserving it keeps the estimate conservative
    join_cost = 1

    pieces, current, current_len = [], [], 0
    for unit, length in zip(units, unit_lengths):
        cost = length + (join_cost if current else 0)
        if current and current_len + cost > max_tokens:
            pieces.append(' '.join(current))
            current, current_len = [unit], length
        else:
            current.append(unit)
            current_len += cost
    if current:
        pieces.append(' '.join(current))
    return pieces


def enforce_budget(chunks, lengths_fn, max_tokens, max_passes=3):
    """
    Guarantee that no chunk exceeds max_tokens.

    Packing sums the token counts of the parts, but token counts are not additive
    across concatenation: the separator costs tokens and BPE can merge across the
    join. Measured chunks therefore run slightly over budget and have to be split
    again, using the measured length rather than the estimate.

    Args:
        chunks (list of str): Candidate chunks
        lengths_fn (callable): Maps a list of strings to their token counts
        max_tokens (int): Budget per chunk
        max_passes (int): Give up re-splitting after this many passes

    Returns:
        list of str: Chunks, none exceeding max_tokens
    """
    chunks = list(chunks)
    target = max_tokens
    for _ in range(max_passes):
        lengths = lengths_fn(chunks)
        if all(length <= max_tokens for length in lengths):
            return chunks

        # Shrink the packing target each pass. Re-splitting at the same target can
        # reproduce the same over-budget piece; a smaller one always makes progress.
        target = max(1, int(target * 0.9))

        rebuilt = []
        for chunk, length in zip(chunks, lengths):
            if length <= max_tokens:
                rebuilt.append(chunk)
            else:
                rebuilt.extend(split_oversized(chunk, lengths_fn, target))

        if rebuilt == chunks:
            break
        chunks = rebuilt

    # Boundary-aware splitting gets almost everything under budget, but a handful of
    # chunks survive because the estimate is not exact. Trim those from the end so the
    # guarantee is absolute rather than statistical; the dataset would otherwise
    # truncate them anyway, silently.
    lengths = lengths_fn(chunks)
    for index, (chunk, length) in enumerate(zip(chunks, lengths)):
        while length > max_tokens and chunk:
            words = chunk.split(' ')
            if len(words) > 1:
                keep = min(len(words) - 1, max(1, int(len(words) * max_tokens / length)))
                chunk = ' '.join(words[:keep])
            else:
                # No word boundary to cut on. Scraped corpora carry pasted minified
                # code and base64 blobs that tokenize into hundreds of tokens with no
                # whitespace, so fall back to characters.
                keep = min(len(chunk) - 1, max(1, int(len(chunk) * max_tokens / length)))
                chunk = chunk[:keep]
            length = lengths_fn([chunk])[0]
        chunks[index] = chunk
    return chunks


def chunk_segments(segments, lengths_fn, max_tokens, min_tokens):
    """
    Pack related segments into chunks of at most max_tokens.

    Callers decide what may share a chunk by choosing what to pass in together: one
    post's comments belong together, two unrelated documents do not.

    Args:
        segments (list of str): Related text segments
        lengths_fn (callable): Maps a list of strings to their token counts
        max_tokens (int): Maximum tokens per chunk
        min_tokens (int): Discard chunks below this length

    Returns:
        list of str: Chunks
    """
    paragraphs = []
    for segment in segments:
        paragraphs.extend(p.strip() for p in segment.split('\n\n') if p.strip())
    if not paragraphs:
        return []

    lengths = lengths_fn(paragraphs)
    separator_cost = lengths_fn([SEPARATOR])[0]

    # Anything longer than a whole chunk has to be broken up first
    units = []
    for paragraph, length in zip(paragraphs, lengths):
        if length <= max_tokens:
            units.append((paragraph, length))
        else:
            pieces = split_oversized(paragraph, lengths_fn, max_tokens)
            units.extend(zip(pieces, lengths_fn(pieces)))

    chunks, current, current_len = [], [], 0
    for unit, length in units:
        cost = length + (separator_cost if current else 0)
        if current and current_len + cost > max_tokens:
            chunks.append(SEPARATOR.join(current))
            current, current_len = [unit], length
        else:
            current.append(unit)
            current_len += cost
    if current:
        chunks.append(SEPARATOR.join(current))

    chunks = enforce_budget(chunks, lengths_fn, max_tokens)

    if min_tokens <= 0:
        return chunks
    return [c for c, n in zip(chunks, lengths_fn(chunks)) if n >= min_tokens]

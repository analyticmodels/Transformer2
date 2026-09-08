"""
Turn collected Reddit posts into a plain-text training corpus.

`data_collection.py` writes one row per post, with a `combined_text` field that
concatenates every top-level comment behind `Title:` / `Content:` / `Comment:`
labels. That format is unusable for language modelling: the labels are scaffolding
the model would learn to reproduce, and a single popular thread can run to tens of
thousands of tokens, so any affordable `max_length` discards most of it.

This module strips the scaffolding and rewrites the corpus as uniform chunks:

    t2-postprocess reddit --max_tokens 512

Chunks are packed from segments belonging to the *same* post, so a chunk reads as a
continuous discussion rather than unrelated fragments glued together, while still
filling close to `max_tokens` and wasting little of each sequence on padding.
Segments are split on paragraph, then sentence, then word boundaries, so a chunk
never ends mid-word.
"""

import argparse
import html
import os
import re

import pandas as pd
from tqdm import tqdm

from transformer2.data.chunking import chunk_segments, enforce_budget, split_oversized  # noqa: F401
from transformer2.tokenization.bpe import create_tokenizer
from transformer2.paths import data_dir

INPUT_NAME = "reddit_posts.parquet"
OUTPUT_NAME = "reddit_text.parquet"

# Reddit placeholders for content that no longer exists
DELETED_MARKERS = {"[deleted]", "[removed]", "[deleted by user]", ""}

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]*\)')      # keep the anchor text
USER_SUB_RE = re.compile(r'(?<!\w)/?[ur]/[A-Za-z0-9_\-]+')   # /u/name, r/sub
QUOTE_PREFIX_RE = re.compile(r'^\s*&gt;.*$|^\s*>.*$', re.MULTILINE)
BOLD_ITALIC_RE = re.compile(r'(\*{1,3}|_{1,3}|~~|`+)')
WHITESPACE_RE = re.compile(r'[ \t]+')
BLANK_LINES_RE = re.compile(r'\n{3,}')


def clean_text(text):
    """
    Reduce one Reddit segment to plain prose.

    Removes the artefacts that are structure rather than language: HTML entities,
    URLs, markdown emphasis, quoted parent comments, and /u/ and /r/ references.

    Args:
        text (str): Raw selftext, comment body or title

    Returns:
        str: Cleaned text, or an empty string if nothing meaningful remains
    """
    if not isinstance(text, str):
        return ""

    if text.strip().lower() in DELETED_MARKERS:
        return ""

    text = html.unescape(text)
    text = QUOTE_PREFIX_RE.sub('', text)       # drop quoted parent text
    text = MARKDOWN_LINK_RE.sub(r'\1', text)   # [anchor](url) -> anchor
    text = URL_RE.sub('', text)
    text = USER_SUB_RE.sub('', text)
    text = BOLD_ITALIC_RE.sub('', text)
    text = WHITESPACE_RE.sub(' ', text)
    text = BLANK_LINES_RE.sub('\n\n', text)

    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines).strip()

    if text.strip().lower() in DELETED_MARKERS:
        return ""
    return text


def post_segments(row, keep_titles=False):
    """
    Extract the prose segments of a single post, in reading order.

    Args:
        row (pandas.Series): One row of the collected Reddit DataFrame
        keep_titles (bool): Include the post title. Titles are natural language, but
            they are also metadata, so they are excluded by default.

    Returns:
        list of str: Cleaned, non-empty segments
    """
    segments = []
    if keep_titles:
        segments.append(row.get('title', ''))
    segments.append(row.get('selftext', ''))

    comments = row.get('comments')
    if comments is not None:
        # parquet round-trips the list column as a numpy array
        segments.extend(list(comments))

    return [cleaned for cleaned in (clean_text(s) for s in segments) if cleaned]


def postprocess_reddit_data(input_path=None, output_path=None,
                            max_tokens=512, min_tokens=32, keep_titles=False,
                            tokenizer=None, drop_duplicates=True):
    """
    Rewrite collected Reddit posts as a chunked plain-text corpus.

    Args:
        input_path (str): Parquet written by data_collection.py
        output_path (str): Destination parquet with a single `text` column
        max_tokens (int): Maximum tokens per chunk; match the training --max_length
        min_tokens (int): Discard chunks shorter than this
        keep_titles (bool): Include post titles as text
        tokenizer: Tokenizer wrapper used to measure chunk lengths. Created if None.
        drop_duplicates (bool): Remove identical chunks, which are common on Reddit

    Returns:
        pandas.DataFrame: The written corpus, with one `text` column

    Raises:
        FileNotFoundError: If input_path does not exist
    """
    input_path = str(input_path or data_dir() / INPUT_NAME)
    output_path = str(output_path or data_dir() / OUTPUT_NAME)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Collect Reddit data first:\n"
            f"    t2-collect --dataset reddit"
        )

    if tokenizer is None:
        tokenizer = create_tokenizer("reddit")

    def lengths_fn(texts):
        texts = list(texts)
        if not texts:
            return []
        return [len(e['input_ids']) for e in tokenizer.encode_batch(texts)]

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} Reddit posts from {input_path}")

    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking", unit="post"):
        segments = post_segments(row, keep_titles=keep_titles)
        chunks.extend(chunk_segments(segments, lengths_fn, max_tokens, min_tokens))

    out = pd.DataFrame({'text': chunks})
    before = len(out)
    if drop_duplicates:
        out = out.drop_duplicates(subset='text').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_parquet(output_path, index=False)

    token_counts = lengths_fn(out['text'].tolist()) if len(out) else []
    print(f"\nWrote {len(out):,} chunks to {output_path}")
    if drop_duplicates:
        print(f"  removed {before - len(out):,} duplicate chunks")
    if token_counts:
        mean_tokens = sum(token_counts) / len(token_counts)
        print(f"  tokens per chunk: mean {mean_tokens:.0f}, "
              f"min {min(token_counts)}, max {max(token_counts)}")
        print(f"  fill at max_length={max_tokens}: {100 * mean_tokens / max_tokens:.0f}% "
              f"(the rest is padding)")
        print(f"  total tokens: {sum(token_counts):,}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Strip metadata from collected Reddit posts and chunk them for training'
    )
    parser.add_argument('--input', default=None,
                        help=f'Collected posts parquet (default: <data>/{INPUT_NAME})')
    parser.add_argument('--output', default=None,
                        help=f'Destination parquet (default: <data>/{OUTPUT_NAME})')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum tokens per chunk; set this to your training --max_length')
    parser.add_argument('--min_tokens', type=int, default=32,
                        help='Discard chunks shorter than this (default: 32)')
    parser.add_argument('--keep_titles', action='store_true',
                        help='Include post titles as text (excluded by default as metadata)')
    parser.add_argument('--keep_duplicates', action='store_true',
                        help='Keep identical chunks instead of deduplicating')
    args = parser.parse_args()

    postprocess_reddit_data(
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        keep_titles=args.keep_titles,
        drop_duplicates=not args.keep_duplicates,
    )


if __name__ == "__main__":
    main()

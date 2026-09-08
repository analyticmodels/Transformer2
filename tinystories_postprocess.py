"""
Prepare the TinyStories corpus for training.

TinyStories arrives as clean prose, so unlike the Reddit collection there is no
scaffolding to strip. What it does need is normalisation, deduplication, and a
guarantee that no story exceeds the training sequence length:

    python tinystories_postprocess.py --max_tokens 320

Each story is a self-contained document, so stories are never packed together: a
chunk that ran from the end of one story into the start of an unrelated one would
teach the model that stories restart mid-sequence. The rare over-long story is split
on sentence boundaries instead.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from text_chunking import chunk_segments
from tokenizer import create_tokenizer
from paths import data_dir

INPUT_NAME = "tinystories.parquet"
OUTPUT_NAME = "tinystories_text.parquet"

WHITESPACE_RE = re.compile(r'[ \t]+')
BLANK_LINES_RE = re.compile(r'\n{3,}')


def clean_story(text):
    """
    Normalise a single story.

    TinyStories needs no artefact removal, only whitespace normalisation: the raw
    rows carry ragged indentation and runs of blank lines that would otherwise become
    tokens the model has to learn.

    Args:
        text (str): Raw story text

    Returns:
        str: Normalised text, or an empty string if nothing remains
    """
    if not isinstance(text, str):
        return ""
    text = WHITESPACE_RE.sub(' ', text)
    text = BLANK_LINES_RE.sub('\n\n', text)
    return '\n'.join(line.strip() for line in text.split('\n')).strip()


def postprocess_tinystories(input_path=None, output_path=None,
                            max_tokens=320, min_tokens=32, tokenizer=None,
                            drop_duplicates=True, batch_size=20000):
    """
    Normalise, deduplicate and length-bound the TinyStories corpus.

    Args:
        input_path (str): Parquet written by data_collection.py
        output_path (str): Destination parquet with a single `text` column
        max_tokens (int): Maximum tokens per row; match the training --max_length
        min_tokens (int): Discard rows shorter than this
        tokenizer: Tokenizer wrapper used to measure lengths. Created if None.
        drop_duplicates (bool): Remove identical stories
        batch_size (int): Stories measured per tokenizer batch

    Returns:
        pandas.DataFrame: The written corpus, with one `text` column

    Raises:
        FileNotFoundError: If input_path does not exist
    """
    input_path = str(input_path or data_dir() / INPUT_NAME)
    output_path = str(output_path or data_dir() / OUTPUT_NAME)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Download it first:\n"
            f"    python data_collection.py --dataset tinystories"
        )

    if tokenizer is None:
        tokenizer = create_tokenizer("tinystories")

    def lengths_fn(texts):
        texts = list(texts)
        if not texts:
            return []
        return [len(e['input_ids']) for e in tokenizer.encode_batch(texts)]

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} stories from {input_path}")

    stories = [clean_story(t) for t in df['text'].tolist()]
    stories = [s for s in stories if s]

    kept = []
    over_long = 0
    for start in tqdm(range(0, len(stories), batch_size), desc="Measuring", unit="batch"):
        batch = stories[start:start + batch_size]
        for story, length in zip(batch, lengths_fn(batch)):
            if length > max_tokens:
                # Split rather than truncate, so the tail of the story is kept
                over_long += 1
                kept.extend(chunk_segments([story], lengths_fn, max_tokens, min_tokens))
            elif length >= min_tokens:
                kept.append(story)

    out = pd.DataFrame({'text': kept})
    before = len(out)
    if drop_duplicates:
        out = out.drop_duplicates(subset='text').reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out.to_parquet(output_path, index=False)

    counts = np.array([
        n for start in range(0, len(out), batch_size)
        for n in lengths_fn(out['text'].tolist()[start:start + batch_size])
    ]) if len(out) else np.array([])

    print(f"\nWrote {len(out):,} rows to {output_path}")
    print(f"  stories split for exceeding {max_tokens} tokens: {over_long:,}")
    if drop_duplicates:
        print(f"  removed {before - len(out):,} duplicates")
    if counts.size:
        print(f"  tokens per row: mean {counts.mean():.0f}, min {counts.min()}, "
              f"max {counts.max()}, p95 {int(np.percentile(counts, 95))}")
        print(f"  fill at max_length={max_tokens}: {100 * counts.mean() / max_tokens:.0f}%")
        print(f"  total tokens: {counts.sum():,}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Prepare TinyStories for training')
    parser.add_argument('--input', default=None,
                        help=f'Downloaded stories parquet (default: <data>/{INPUT_NAME})')
    parser.add_argument('--output', default=None,
                        help=f'Destination parquet (default: <data>/{OUTPUT_NAME})')
    parser.add_argument('--max_tokens', type=int, default=320,
                        help='Maximum tokens per row; set this to your training --max_length')
    parser.add_argument('--min_tokens', type=int, default=32,
                        help='Discard rows shorter than this (default: 32)')
    parser.add_argument('--keep_duplicates', action='store_true',
                        help='Keep identical stories instead of deduplicating')
    args = parser.parse_args()

    postprocess_tinystories(
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        drop_duplicates=not args.keep_duplicates,
    )


if __name__ == "__main__":
    main()

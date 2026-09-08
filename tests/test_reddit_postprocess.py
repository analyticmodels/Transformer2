"""
Tests for the Reddit postprocessing step.

The collected corpus is one row per post with every comment concatenated behind
`Title:`/`Content:`/`Comment:` labels. These tests pin down that the labels and
other non-prose artefacts are removed, and that the resulting chunks fit the
training budget without ending mid-word.
"""

import numpy as np
import pandas as pd
import pytest

from transformer2.data.reddit import (
    enforce_budget,
    chunk_segments,
    clean_text,
    post_segments,
    postprocess_reddit_data,
    split_oversized,
)


def word_lengths(texts):
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return [len(t.split()) for t in texts]


@pytest.fixture
def raw_posts():
    return pd.DataFrame([
        {
            'id': 'a1', 'title': 'Why is the sky blue?',
            'selftext': 'I have always wondered about this. Can someone explain?',
            'subreddit': 'askscience', 'score': 42, 'num_comments': 2,
            'permalink': '/r/askscience/a1', 'url': 'https://reddit.com/a1',
            'created_utc': 0,
            'comments': ['Rayleigh scattering is the reason. ' * 20,
                         'Shorter wavelengths scatter more strongly. ' * 20],
        },
        {
            'id': 'b2', 'title': 'Deleted thread',
            'selftext': '[deleted]', 'subreddit': 'python', 'score': 1,
            'num_comments': 0, 'permalink': '/r/python/b2',
            'url': 'https://reddit.com/b2', 'created_utc': 0,
            'comments': ['[removed]'],
        },
    ])


class TestCleanText:
    def test_strips_urls(self):
        assert 'http' not in clean_text("see https://example.com/x?y=1 for more")

    def test_keeps_markdown_link_text(self):
        assert clean_text("read [the docs](https://x.com) now") == "read the docs now"

    def test_strips_user_and_subreddit_references(self):
        cleaned = clean_text("thanks /u/someone over at /r/python and r/science")
        assert 'u/someone' not in cleaned
        assert 'r/python' not in cleaned
        assert 'r/science' not in cleaned

    def test_unescapes_html_entities(self):
        assert clean_text("Tom &amp; Jerry&#39;s") == "Tom & Jerry's"

    def test_drops_quoted_parent_text(self):
        cleaned = clean_text("&gt; the quoted parent\nmy actual reply")
        assert 'quoted parent' not in cleaned
        assert 'my actual reply' in cleaned

    def test_removes_markdown_emphasis(self):
        assert clean_text("this is **bold** and *italic* and `code`") == \
            "this is bold and italic and code"

    @pytest.mark.parametrize("marker", ['[deleted]', '[removed]', '', '   '])
    def test_deleted_content_becomes_empty(self, marker):
        assert clean_text(marker) == ""

    def test_non_string_input_is_safe(self):
        assert clean_text(None) == ""
        assert clean_text(float('nan')) == ""

    def test_collapses_excess_blank_lines(self):
        assert '\n\n\n' not in clean_text("a\n\n\n\n\nb")


class TestPostSegments:
    def test_title_excluded_by_default(self, raw_posts):
        segments = post_segments(raw_posts.iloc[0])

        assert not any('Why is the sky blue' in s for s in segments)

    def test_title_included_when_requested(self, raw_posts):
        segments = post_segments(raw_posts.iloc[0], keep_titles=True)

        assert any('Why is the sky blue' in s for s in segments)

    def test_body_and_comments_are_separate_segments(self, raw_posts):
        segments = post_segments(raw_posts.iloc[0])

        assert len(segments) == 3  # selftext + 2 comments

    def test_no_scaffolding_labels_survive(self, raw_posts):
        """`Title:`/`Content:`/`Comment:` are structure the model must not learn."""
        segments = post_segments(raw_posts.iloc[0], keep_titles=True)

        joined = "\n".join(segments)
        for label in ('Title:', 'Content:', 'Comment:'):
            assert label not in joined

    def test_deleted_post_yields_nothing(self, raw_posts):
        assert post_segments(raw_posts.iloc[1]) == []

    def test_handles_numpy_array_comments(self):
        """parquet round-trips the list column as a numpy array."""
        row = pd.Series({'title': 't', 'selftext': 'body text',
                         'comments': np.array(['first comment', 'second comment'])})

        assert len(post_segments(row)) == 3


class TestChunking:
    def test_chunks_stay_within_budget(self):
        segments = ["word " * 400]

        chunks = chunk_segments(segments, word_lengths, max_tokens=100, min_tokens=0)

        assert all(len(c.split()) <= 100 for c in chunks)

    def test_no_content_is_lost(self):
        segments = ["alpha beta gamma", "delta epsilon zeta"]

        chunks = chunk_segments(segments, word_lengths, max_tokens=100, min_tokens=0)

        joined = " ".join(chunks).split()
        assert joined == "alpha beta gamma delta epsilon zeta".split()

    def test_short_segments_are_packed_together(self):
        """Packing keeps sequences full; otherwise most of each chunk is padding."""
        segments = ["short one", "short two", "short three"]

        chunks = chunk_segments(segments, word_lengths, max_tokens=100, min_tokens=0)

        assert len(chunks) == 1

    def test_chunks_below_min_tokens_are_dropped(self):
        segments = ["tiny"]

        assert chunk_segments(segments, word_lengths, max_tokens=100, min_tokens=10) == []

    def test_empty_input_yields_no_chunks(self):
        assert chunk_segments([], word_lengths, max_tokens=100, min_tokens=0) == []

    def test_oversized_paragraph_is_split_on_sentences(self):
        paragraph = "First sentence here. Second sentence here. Third sentence here."

        pieces = split_oversized(paragraph, word_lengths, max_tokens=4)

        assert len(pieces) > 1
        assert all(len(p.split()) <= 4 for p in pieces)

    def test_split_never_breaks_a_word(self):
        pieces = split_oversized("alpha beta gamma delta", word_lengths, max_tokens=2)

        rejoined = " ".join(pieces).split()
        assert rejoined == ["alpha", "beta", "gamma", "delta"]

    def test_packing_does_not_cross_posts(self, raw_posts, tmp_path):
        """
        Each chunk must come from a single post. Packing unrelated threads together
        would teach the model that arbitrary topics follow one another.
        """
        first = chunk_segments(post_segments(raw_posts.iloc[0]), word_lengths,
                               max_tokens=1000, min_tokens=0)

        assert all('Rayleigh' in c or 'wavelength' in c or 'wondered' in c for c in first)


class TestEndToEnd:
    def _write(self, df, tmp_path):
        path = tmp_path / 'reddit_posts.parquet'
        df.to_parquet(path, index=False)
        return path

    def test_writes_single_text_column(self, raw_posts, tmp_path):
        src = self._write(raw_posts, tmp_path)
        out = tmp_path / 'reddit_text.parquet'

        postprocess_reddit_data(str(src), str(out), max_tokens=64, min_tokens=5,
                                tokenizer=_WordTokenizer())

        result = pd.read_parquet(out)
        assert list(result.columns) == ['text']
        assert len(result) > 0

    def test_output_contains_no_metadata(self, raw_posts, tmp_path):
        src = self._write(raw_posts, tmp_path)
        out = tmp_path / 'reddit_text.parquet'

        postprocess_reddit_data(str(src), str(out), max_tokens=64, min_tokens=5,
                                tokenizer=_WordTokenizer())

        blob = "\n".join(pd.read_parquet(out)['text'])
        for artefact in ('Title:', 'Content:', 'Comment:', 'askscience',
                         'permalink', 'https://'):
            assert artefact not in blob

    def test_chunks_respect_max_tokens(self, raw_posts, tmp_path):
        src = self._write(raw_posts, tmp_path)
        out = tmp_path / 'reddit_text.parquet'

        postprocess_reddit_data(str(src), str(out), max_tokens=30, min_tokens=1,
                                tokenizer=_WordTokenizer())

        assert all(len(t.split()) <= 30 for t in pd.read_parquet(out)['text'])

    def test_duplicate_chunks_are_removed(self, tmp_path):
        df = pd.DataFrame([
            {'title': 't', 'selftext': 'identical body text here',
             'comments': ['identical body text here']},
            {'title': 't', 'selftext': 'identical body text here', 'comments': []},
        ])
        src = self._write(df, tmp_path)
        out = tmp_path / 'reddit_text.parquet'

        postprocess_reddit_data(str(src), str(out), max_tokens=64, min_tokens=1,
                                tokenizer=_WordTokenizer())

        texts = pd.read_parquet(out)['text'].tolist()
        assert len(texts) == len(set(texts))

    def test_missing_input_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data_collection.py"):
            postprocess_reddit_data(str(tmp_path / 'absent.parquet'),
                                    str(tmp_path / 'out.parquet'),
                                    tokenizer=_WordTokenizer())


class TestBudgetWithRealTokenizer:
    """
    A whitespace tokenizer is additive across concatenation, so it cannot catch
    over-budget chunks. Real BPE is not additive: the separator costs tokens and
    merges happen across joins. These tests use the real tokenizer.
    """

    @pytest.fixture(scope='class')
    def tokenizer(self):
        from transformer2.tokenization.bpe import create_tokenizer
        return create_tokenizer("bluesky")

    @pytest.fixture(scope='class')
    def lengths_fn(self, tokenizer):
        def _lengths(texts):
            texts = list(texts)
            return [len(e['input_ids']) for e in tokenizer.encode_batch(texts)] if texts else []
        return _lengths

    @pytest.mark.parametrize("max_tokens", [64, 128, 512])
    def test_no_chunk_exceeds_budget(self, lengths_fn, max_tokens):
        prose = ("The reason this happens is subtle and worth unpacking carefully. "
                 "Most explanations skip the important part entirely. ")
        segments = [prose * 30, prose * 5, "a short reply", prose * 12]

        chunks = chunk_segments(segments, lengths_fn, max_tokens, min_tokens=0)

        over = [n for n in lengths_fn(chunks) if n > max_tokens]
        assert not over, f"chunks exceeding {max_tokens}: {over}"

    def test_enforce_budget_resplits_oversized_chunks(self, lengths_fn):
        prose = "Sentence number one here. Sentence number two here. " * 40

        result = enforce_budget([prose], lengths_fn, max_tokens=100)

        assert all(n <= 100 for n in lengths_fn(result))

    def test_enforce_budget_terminates_on_unsplittable_input(self, lengths_fn):
        """A single token longer than the budget cannot be split; must not loop."""
        result = enforce_budget(["x" * 5000], lengths_fn, max_tokens=4)

        assert isinstance(result, list) and result

    def test_content_is_preserved_through_budget_enforcement(self, lengths_fn):
        prose = "Alpha beta gamma delta epsilon zeta eta theta. " * 20

        chunks = chunk_segments([prose], lengths_fn, max_tokens=64, min_tokens=0)

        assert " ".join(chunks).split() == prose.split()


class _WordTokenizer:
    """Whitespace tokenizer with the wrapper's encode_batch contract."""

    def encode_batch(self, texts):
        return [{'input_ids': t.split(), 'attention_mask': [1] * len(t.split())}
                for t in texts]

    def encode(self, text):
        return {'input_ids': text.split(), 'attention_mask': [1] * len(text.split())}

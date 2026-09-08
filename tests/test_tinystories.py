"""
Tests for TinyStories collection and postprocessing.

TinyStories arrives clean, so the work is normalisation, deduplication and the
guarantee that no row exceeds the training sequence length. Stories are separate
documents and must never be packed together.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from tinystories_postprocess import clean_story, postprocess_tinystories


class WordTokenizer:
    """Whitespace tokenizer with the wrapper's encode_batch contract."""

    def encode_batch(self, texts):
        return [{'input_ids': t.split(), 'attention_mask': [1] * len(t.split())}
                for t in texts]

    def encode(self, text):
        return {'input_ids': text.split(), 'attention_mask': [1] * len(text.split())}


@pytest.fixture
def stories():
    return pd.DataFrame({'text': [
        "One day, a little girl named Lily found a needle in her room.",
        "Once  upon   a time,\t there was a car named Beep.",
        "One day, a little girl named Lily found a needle in her room.",  # duplicate
        " ".join(f"word{i}" for i in range(200)),                         # over-long
        "tiny",                                                           # too short
    ]})


class TestCleanStory:
    def test_collapses_runs_of_spaces_and_tabs(self):
        assert clean_story("a  b\t\tc") == "a b c"

    def test_collapses_excess_blank_lines(self):
        assert '\n\n\n' not in clean_story("a\n\n\n\n\nb")

    def test_strips_line_indentation(self):
        assert clean_story("   hello\n   world") == "hello\nworld"

    def test_non_string_is_safe(self):
        assert clean_story(None) == ""
        assert clean_story(float('nan')) == ""

    def test_preserves_prose(self):
        text = "Lily found a needle. She was happy!"
        assert clean_story(text) == text


class TestPostprocess:
    def _run(self, df, tmp_path, **kwargs):
        src = tmp_path / 'tinystories.parquet'
        out = tmp_path / 'tinystories_text.parquet'
        df.to_parquet(src, index=False)
        kwargs.setdefault('max_tokens', 50)
        kwargs.setdefault('min_tokens', 5)
        postprocess_tinystories(str(src), str(out), tokenizer=WordTokenizer(), **kwargs)
        return pd.read_parquet(out)

    def test_writes_single_text_column(self, stories, tmp_path):
        assert list(self._run(stories, tmp_path).columns) == ['text']

    def test_no_row_exceeds_max_tokens(self, stories, tmp_path):
        result = self._run(stories, tmp_path, max_tokens=50)

        assert all(len(t.split()) <= 50 for t in result['text'])

    def test_over_long_stories_are_split_not_dropped(self, stories, tmp_path):
        """The tail of a long story is training data too; truncating discards it."""
        result = self._run(stories, tmp_path, max_tokens=50, min_tokens=1)

        recovered = sum(len(t.split()) for t in result['text'] if t.startswith('word'))
        assert recovered >= 190, f"only {recovered} of 200 words survived splitting"

    def test_short_stories_are_dropped(self, stories, tmp_path):
        result = self._run(stories, tmp_path, min_tokens=5)

        assert 'tiny' not in result['text'].tolist()

    def test_duplicates_are_removed(self, stories, tmp_path):
        texts = self._run(stories, tmp_path)['text'].tolist()

        assert len(texts) == len(set(texts))

    def test_duplicates_kept_when_requested(self, stories, tmp_path):
        result = self._run(stories, tmp_path, drop_duplicates=False)

        assert len(result) > len(set(result['text']))

    def test_stories_are_never_packed_together(self, tmp_path):
        """
        Each story is its own document. Merging two would teach the model that a
        narrative restarts mid-sequence.
        """
        df = pd.DataFrame({'text': ["First story about a cat and a hat.",
                                    "Second story about a dog and a log."]})

        result = self._run(df, tmp_path, max_tokens=500, min_tokens=1)

        for text in result['text']:
            assert not ('First story' in text and 'Second story' in text)

    def test_whitespace_is_normalised_in_output(self, stories, tmp_path):
        blob = "\n".join(self._run(stories, tmp_path)['text'])

        assert '\t' not in blob
        assert '  ' not in blob

    def test_missing_input_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data_collection.py"):
            postprocess_tinystories(str(tmp_path / 'absent.parquet'),
                                    str(tmp_path / 'out.parquet'),
                                    tokenizer=WordTokenizer())


class TestCollector:
    def test_writes_text_column_and_respects_max_samples(self, tmp_path):
        from data_collection import collect_tinystories

        fake = [{'text': f'story {i}'} for i in range(10)]

        class FakeDataset:
            def __init__(self, rows): self.rows = rows
            def __len__(self): return len(self.rows)
            def select(self, idx): return FakeDataset([self.rows[i] for i in idx])
            def __getitem__(self, key): return [r[key] for r in self.rows]

        out = tmp_path / 'tinystories.parquet'
        with patch('data_collection.load_dataset', return_value=FakeDataset(fake)):
            df = collect_tinystories(str(out), max_samples=4)

        assert list(df.columns) == ['text']
        assert len(df) == 4
        assert pd.read_parquet(out)['text'].iloc[0] == 'story 0'

    def test_requests_the_named_split(self, tmp_path):
        from data_collection import collect_tinystories

        class FakeDataset:
            def __len__(self): return 1
            def __getitem__(self, key): return ['a story']

        with patch('data_collection.load_dataset', return_value=FakeDataset()) as loader:
            collect_tinystories(str(tmp_path / 'x.parquet'), split='validation')

        assert loader.call_args.kwargs['split'] == 'validation'

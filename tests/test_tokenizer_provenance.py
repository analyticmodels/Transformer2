"""
Tests for per-corpus tokenizers and their provenance.

Every tokenizer this project trains has exactly 50000 entries, so vocabulary size
cannot distinguish a Bluesky tokenizer from a Reddit one. Standing up per-corpus
directories without a fingerprint would therefore increase the chance of silently
training or generating against the wrong vocabulary, not reduce it.
"""

import json
import hashlib

import pytest
import torch

import transformer2.inference.generate as inference
import transformer2.paths as paths
from transformer2.model.transformer import SmallTransformer
from transformer2.tokenization.bpe import (
    create_tokenizer,
    read_tokenizer_meta,
    vocab_fingerprint,
    write_tokenizer_meta,
)


@pytest.fixture
def fake_tokenizer_dir(tmp_path, monkeypatch):
    """A directory shaped like a saved tokenizer, without training one."""
    monkeypatch.setenv(paths.TOKENIZER_ENV, str(tmp_path))

    def _make(dataset, vocab_body='{"a": 0}'):
        d = tmp_path / dataset
        d.mkdir(parents=True, exist_ok=True)
        (d / "vocab.json").write_text(vocab_body)
        (d / "merges.txt").write_text("#version: 0.2\n")
        return d

    return _make


class TestFingerprint:
    def test_hashes_the_vocabulary(self, fake_tokenizer_dir):
        d = fake_tokenizer_dir('bluesky', '{"hello": 1}')

        assert vocab_fingerprint(d) == hashlib.sha256(b'{"hello": 1}').hexdigest()

    def test_different_corpora_hash_differently(self, fake_tokenizer_dir):
        a = fake_tokenizer_dir('bluesky', '{"a": 0}')
        b = fake_tokenizer_dir('reddit', '{"b": 0}')

        assert vocab_fingerprint(a) != vocab_fingerprint(b)

    def test_same_size_vocabularies_still_differ(self, fake_tokenizer_dir):
        """The whole point: equal vocab_size, different tokenizer."""
        a = fake_tokenizer_dir('bluesky', '{"x": 0, "y": 1}')
        b = fake_tokenizer_dir('reddit', '{"p": 0, "q": 1}')

        assert vocab_fingerprint(a) != vocab_fingerprint(b)


class TestMeta:
    def test_round_trips(self, fake_tokenizer_dir):
        d = fake_tokenizer_dir('reddit')

        written = write_tokenizer_meta(d, 'reddit', 50000, 2, source='data/reddit.parquet')
        read = read_tokenizer_meta(d)

        assert read == written
        assert read['dataset'] == 'reddit'
        assert read['vocab_sha256'] == vocab_fingerprint(d)

    def test_absent_meta_reads_as_none(self, tmp_path):
        assert read_tokenizer_meta(tmp_path) is None

    def test_corrupt_meta_reads_as_none(self, fake_tokenizer_dir):
        d = fake_tokenizer_dir('bluesky')
        (d / "meta.json").write_text("{ not json")

        assert read_tokenizer_meta(d) is None


class TestCreateTokenizer:
    def test_dataset_type_is_required(self):
        """
        It used to default to 'wikitext' and be ignored for path resolution, so five
        of six call sites never passed it and every corpus shared one vocabulary.
        """
        with pytest.raises(TypeError):
            create_tokenizer()

    def test_missing_tokenizer_raises_instead_of_falling_back(self, tmp_path, monkeypatch):
        """
        The GPT-2 fallback silently swapped in a 50257-token vocabulary, so training
        continued against the wrong one with only a print() to show for it.
        """
        monkeypatch.setenv(paths.TOKENIZER_ENV, str(tmp_path))

        with pytest.raises(FileNotFoundError, match="No tokenizer for 'reddit'"):
            create_tokenizer('reddit')

    def test_error_names_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.setenv(paths.TOKENIZER_ENV, str(tmp_path))

        with pytest.raises(FileNotFoundError, match="--dataset_type reddit"):
            create_tokenizer('reddit')

    def test_corpora_resolve_to_separate_directories(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.TOKENIZER_ENV, str(tmp_path))

        assert paths.tokenizer_dir('bluesky') != paths.tokenizer_dir('reddit')


class TestLoadModelVerifiesProvenance:
    def _checkpoint(self, tmp_path, config):
        model = SmallTransformer(vocab_size=64, d_model=32, num_heads=4, num_layers=1,
                                 d_ff=32, max_seq_len=16, dropout=0.0)
        path = tmp_path / 'ckpt.pt'
        torch.save({'model_state_dict': model.state_dict(), 'config': config}, path)
        return str(path)

    class _Tok:
        def get_vocab_size(self): return 64

    def test_mismatched_fingerprint_raises(self, tmp_path, fake_tokenizer_dir):
        d = fake_tokenizer_dir('bluesky')
        write_tokenizer_meta(d, 'bluesky', 64, 2)
        path = self._checkpoint(tmp_path, {
            'num_heads': 4,
            'tokenizer_dataset': 'tinystories',
            'tokenizer_vocab_sha256': 'f' * 64,
        })

        with pytest.raises(RuntimeError, match="trained with the 'tinystories' tokenizer"):
            inference.load_model(path, tokenizer=self._Tok(), dataset_type='bluesky')

    def test_matching_fingerprint_loads(self, tmp_path, fake_tokenizer_dir):
        d = fake_tokenizer_dir('bluesky')
        meta = write_tokenizer_meta(d, 'bluesky', 64, 2)
        path = self._checkpoint(tmp_path, {
            'num_heads': 4,
            'tokenizer_dataset': 'bluesky',
            'tokenizer_vocab_sha256': meta['vocab_sha256'],
        })

        model, _ = inference.load_model(path, tokenizer=self._Tok(), dataset_type='bluesky')

        assert model is not None

    def test_checkpoint_without_a_fingerprint_still_loads(self, tmp_path, fake_tokenizer_dir):
        """Checkpoints predating provenance tracking must keep working."""
        fake_tokenizer_dir('bluesky')
        path = self._checkpoint(tmp_path, {'num_heads': 4})

        model, _ = inference.load_model(path, tokenizer=self._Tok(), dataset_type='bluesky')

        assert model is not None

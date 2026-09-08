"""
Tests for inference.load_model.

A load failure used to be swallowed and replaced with a randomly initialised model,
while the caller printed "Model loaded successfully". Every failure mode here must
raise instead.
"""

import pytest
import torch

from coreTransformer import SmallTransformer
from conftest import StubTokenizer

from inference import load_model


@pytest.fixture
def trained_model():
    return SmallTransformer(
        vocab_size=64, d_model=32, num_heads=4, num_layers=2,
        d_ff=48, max_seq_len=16, dropout=0.0
    )


@pytest.fixture
def config():
    return {'num_heads': 4}


class TestSuccessfulLoad:
    def test_reconstructs_the_original_architecture(self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)

        loaded, _ = load_model(path, tokenizer=StubTokenizer(vocab_size=64))

        assert (loaded.d_model, loaded.num_heads, loaded.num_layers) == (32, 4, 2)
        assert (loaded.d_ff, loaded.max_seq_len) == (48, 16)

    def test_weights_actually_transfer(self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)

        loaded, _ = load_model(path, tokenizer=StubTokenizer(vocab_size=64))

        expected = trained_model.state_dict()['embedding.weight']
        assert torch.allclose(loaded.state_dict()['embedding.weight'].cpu(), expected)

    def test_returns_the_tokenizer_it_was_given(self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)
        tokenizer = StubTokenizer(vocab_size=64)

        _, returned = load_model(path, tokenizer=tokenizer)

        assert returned is tokenizer

    def test_model_is_in_eval_mode(self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)

        loaded, _ = load_model(path, tokenizer=StubTokenizer(vocab_size=64))

        assert not loaded.training

    def test_num_heads_override_is_honoured(self, trained_model, write_checkpoint):
        """A checkpoint whose recorded head count is wrong can still be loaded."""
        path = write_checkpoint(trained_model, {'num_heads': 4})

        loaded, _ = load_model(path, tokenizer=StubTokenizer(vocab_size=64), num_heads=2)

        assert loaded.num_heads == 2


class TestFailuresRaise:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Could not read checkpoint"):
            load_model(str(tmp_path / 'nope.pt'), tokenizer=StubTokenizer())

    def test_checkpoint_without_state_dict_raises(self, tmp_path):
        path = tmp_path / 'bad.pt'
        torch.save({'epoch': 1}, path)

        with pytest.raises(RuntimeError, match="model_state_dict"):
            load_model(str(path), tokenizer=StubTokenizer())

    def test_configless_checkpoint_raises_rather_than_guessing(self, trained_model, write_checkpoint):
        path = write_checkpoint(trained_model, config=None)

        with pytest.raises(ValueError, match="num_heads"):
            load_model(path, tokenizer=StubTokenizer(vocab_size=64))

    def test_tokenizer_vocab_mismatch_raises_a_specific_error(self, trained_model, config, write_checkpoint):
        """
        create_tokenizer silently falls back to GPT-2 (50257) when tokenizer/vocab.json
        is missing. Under a strict load that must be a named error, not a shape dump.
        """
        path = write_checkpoint(trained_model, config)

        with pytest.raises(RuntimeError, match="does not match checkpoint"):
            load_model(path, tokenizer=StubTokenizer(vocab_size=50257))

    def test_incompatible_weights_raise(self, trained_model, config, write_checkpoint):
        corrupted = dict(trained_model.state_dict())
        corrupted['transformer_blocks.0.ff.0.weight'] = torch.zeros(999, 32)
        path = write_checkpoint(trained_model, config, state_dict=corrupted)

        with pytest.raises(RuntimeError):
            load_model(path, tokenizer=StubTokenizer(vocab_size=64))

    def test_never_returns_an_untrained_substitute(self, tmp_path):
        """
        The regression guard: on failure the function must not hand back a usable
        model, because callers report success on whatever they receive.
        """
        with pytest.raises(Exception):
            load_model(str(tmp_path / 'absent.pt'), tokenizer=StubTokenizer())


class TestCallerSuppliedModel:
    def test_uses_the_given_model_instead_of_building_one(self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)
        target = SmallTransformer(vocab_size=64, d_model=32, num_heads=4, num_layers=2,
                                  d_ff=48, max_seq_len=16, dropout=0.0)

        loaded, _ = load_model(path, model=target, tokenizer=StubTokenizer(vocab_size=64))

        assert loaded is target

    def test_mismatched_supplied_model_raises_instead_of_loading_partially(
            self, trained_model, config, write_checkpoint):
        path = write_checkpoint(trained_model, config)
        wrong_shape = SmallTransformer(vocab_size=64, d_model=64, num_heads=4, num_layers=2,
                                       d_ff=48, max_seq_len=16, dropout=0.0)

        with pytest.raises(RuntimeError, match="does not fit"):
            load_model(path, model=wrong_shape, tokenizer=StubTokenizer(vocab_size=64))

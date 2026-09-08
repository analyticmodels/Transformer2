"""
Tests for inference.infer_architecture.

The central hazard: Q/K/V projections are d_model x d_model for every head count, so a
wrong num_heads passes load_state_dict cleanly and only shows up as garbage output.
These tests pin down that the function never invents a head count.
"""

import pytest
import torch

from coreTransformer import SmallTransformer
from inference import infer_architecture


@pytest.fixture
def state_dict():
    model = SmallTransformer(
        vocab_size=64, d_model=32, num_heads=4, num_layers=3,
        d_ff=48, max_seq_len=20, dropout=0.0
    )
    return model.state_dict()


class TestDerivedFromWeights:
    @pytest.mark.parametrize("key,expected", [
        ('vocab_size', 64),
        ('d_model', 32),
        ('num_layers', 3),
        ('d_ff', 48),
        ('max_seq_len', 20),
    ])
    def test_dimension_is_read_from_the_weights(self, state_dict, key, expected):
        arch = infer_architecture(state_dict, num_heads_override=4)

        assert arch[key] == expected

    def test_weights_win_over_a_disagreeing_config(self, state_dict):
        """A config can be stale or wrong; the weights are what actually loads."""
        lying = {'d_model': 999, 'd_ff': 999, 'num_layers': 99, 'vocab_size': 999}

        arch = infer_architecture(state_dict, config=lying, num_heads_override=4)

        assert (arch['d_model'], arch['d_ff'], arch['num_layers'], arch['vocab_size']) == (32, 48, 3, 64)

    def test_result_can_rebuild_the_model(self, state_dict):
        """The returned kwargs must be directly usable to reconstruct the model."""
        arch = infer_architecture(state_dict, num_heads_override=4)

        rebuilt = SmallTransformer(dropout=0.0, **arch)
        rebuilt.load_state_dict(state_dict)  # raises if any dimension is wrong


class TestHeadCountResolution:
    def test_refuses_to_guess_when_nothing_records_the_head_count(self, state_dict):
        """
        Guessing (e.g. d_model // 64) yields a model that loads without error and
        generates noise. Failing loudly is the only safe option.
        """
        with pytest.raises(ValueError, match="num_heads"):
            infer_architecture(state_dict, config=None)

    def test_error_names_the_flag_that_fixes_it(self, state_dict):
        with pytest.raises(ValueError, match="--num_heads"):
            infer_architecture(state_dict, config={})

    def test_config_supplies_the_head_count(self, state_dict):
        arch = infer_architecture(state_dict, config={'num_heads': 8})

        assert arch['num_heads'] == 8

    def test_override_beats_the_config(self, state_dict):
        arch = infer_architecture(state_dict, config={'num_heads': 8}, num_heads_override=2)

        assert arch['num_heads'] == 2

    def test_rejects_a_head_count_that_does_not_divide_d_model(self, state_dict):
        """Better a clear error here than a bare assert deep inside MultiHeadAttention."""
        with pytest.raises(ValueError, match="does not divide"):
            infer_architecture(state_dict, num_heads_override=7)


class TestConfigWithNoneValues:
    def test_present_but_none_head_count_is_not_treated_as_supplied(self, state_dict):
        """
        A config written by older code can hold None. Testing key presence rather than
        value would set num_heads=None and crash on the divisibility check.
        """
        with pytest.raises(ValueError, match="num_heads"):
            infer_architecture(state_dict, config={'num_heads': None})

    def test_none_values_do_not_defeat_weight_derivation(self, state_dict):
        arch = infer_architecture(
            state_dict,
            config={'d_model': None, 'd_ff': None, 'num_layers': None},
            num_heads_override=4,
        )

        assert (arch['d_model'], arch['d_ff'], arch['num_layers']) == (32, 48, 3)


class TestDegradedCheckpoints:
    def test_falls_back_to_config_when_weights_are_absent(self):
        config = {'d_model': 128, 'd_ff': 512, 'num_layers': 4, 'num_heads': 8, 'vocab_size': 100}

        arch = infer_architecture({}, config=config)

        assert arch['d_model'] == 128
        assert arch['d_ff'] == 512
        assert arch['num_layers'] == 4

    def test_layer_count_reflects_actual_blocks_present(self, state_dict):
        """Trailing blocks removed from the weights must reduce the reported depth."""
        trimmed = {k: v for k, v in state_dict.items() if not k.startswith('transformer_blocks.2.')}

        arch = infer_architecture(trimmed, num_heads_override=4)

        assert arch['num_layers'] == 2

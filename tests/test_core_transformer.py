"""
Tests for the model itself.

The stored architecture attributes matter beyond bookkeeping: training.py writes them
into the checkpoint config and inference.py rebuilds the model from that config, so a
missing attribute becomes a silently mis-shaped model at load time.
"""

import pytest
import torch

from coreTransformer import MultiHeadAttention, SmallTransformer


@pytest.fixture
def model():
    return SmallTransformer(
        vocab_size=64, d_model=32, num_heads=4, num_layers=2,
        d_ff=48, max_seq_len=16, dropout=0.0
    )


class TestStoredArchitecture:
    @pytest.mark.parametrize("attribute,expected", [
        ('vocab_size', 64),
        ('d_model', 32),
        ('num_heads', 4),
        ('num_layers', 2),
        ('d_ff', 48),
        ('max_seq_len', 16),
        ('dropout_p', 0.0),
    ])
    def test_constructor_argument_is_retained(self, model, attribute, expected):
        assert getattr(model, attribute) == expected

    def test_dropout_attribute_is_not_shadowed_by_the_module(self, model):
        """
        self.dropout is the nn.Dropout module, so the rate is stored as dropout_p.
        Overwriting self.dropout with a float would break the forward pass.
        """
        assert isinstance(model.dropout, torch.nn.Dropout)
        assert isinstance(model.dropout_p, float)

    def test_head_count_is_visible_for_checkpointing(self, model):
        """
        num_heads cannot be recovered from the weights, so if the model does not
        expose it the checkpoint config cannot record it.
        """
        assert getattr(model, 'num_heads', None) is not None


class TestForward:
    def test_output_shape_is_batch_seq_vocab(self, model):
        input_ids = torch.randint(0, 64, (2, 8))
        logits = model(input_ids, attention_mask=torch.ones(2, 8, dtype=torch.long))

        assert logits.shape == (2, 8, 64)

    def test_forward_without_attention_mask(self, model):
        logits = model(torch.randint(0, 64, (1, 5)))

        assert logits.shape == (1, 5, 64)
        assert torch.isfinite(logits).all()

    def test_accepts_sequences_up_to_max_seq_len(self, model):
        logits = model(torch.randint(0, 64, (1, model.max_seq_len)))

        assert logits.shape[1] == model.max_seq_len

    def test_masked_positions_change_the_output(self, model):
        model.eval()
        input_ids = torch.randint(0, 64, (1, 8))
        full = torch.ones(1, 8, dtype=torch.long)
        partial = full.clone()
        partial[0, 5:] = 0

        with torch.no_grad():
            assert not torch.allclose(model(input_ids, full), model(input_ids, partial))


class TestConstructorValidation:
    def test_rejects_head_count_that_does_not_divide_d_model(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=32, num_heads=5, dropout=0.0, layer_idx=0, num_layers=1)

    def test_model_rejects_indivisible_head_count(self):
        with pytest.raises(AssertionError):
            SmallTransformer(vocab_size=16, d_model=32, num_heads=7, num_layers=1,
                             d_ff=16, max_seq_len=8, dropout=0.0)

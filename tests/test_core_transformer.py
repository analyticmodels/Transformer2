"""
Tests for the model itself.

The stored architecture attributes matter beyond bookkeeping: training.py writes them
into the checkpoint config and inference.py rebuilds the model from that config, so a
missing attribute becomes a silently mis-shaped model at load time.
"""

import pytest
import torch

from transformer2.model.transformer import MultiHeadAttention, SmallTransformer


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


class TestCausalMasking:
    """
    The model is a next-token predictor: labels[i] is input_ids[i+1], so a position
    that can attend to i+1 can read its own target. Attention was bidirectional here
    once, and training drove the loss down by exploiting that leak rather than
    learning to predict, which generation then exposed as noise. These tests pin the
    property down from the outside, on outputs rather than on mask internals.
    """

    def _logits(self, model, input_ids, mask=None):
        with torch.no_grad():
            return model(input_ids, mask)

    def test_a_later_token_cannot_change_an_earlier_position(self, model):
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        altered = ids.clone()
        altered[0, 7] = (ids[0, 7] + 1) % 64

        before = self._logits(model, ids)
        after = self._logits(model, altered)

        # Bit-identical, not merely close: any leak at all is the bug.
        assert torch.equal(before[:, :7], after[:, :7])
        assert not torch.equal(before[:, 7], after[:, 7])

    def test_causality_holds_at_every_position(self, model):
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        base = self._logits(model, ids)

        for cut in range(1, 8):
            altered = ids.clone()
            altered[0, cut] = (ids[0, cut] + 1) % 64
            assert torch.equal(base[:, :cut], self._logits(model, altered)[:, :cut]), cut

    def test_causality_holds_with_a_padding_mask(self, model):
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)
        altered = ids.clone()
        altered[0, 6] = (ids[0, 6] + 1) % 64

        assert torch.equal(self._logits(model, ids, mask)[:, :6],
                           self._logits(model, altered, mask)[:, :6])

    def test_trailing_padding_does_not_change_real_positions(self, model):
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)
        mask[0, 5:] = 0

        padded = self._logits(model, ids, mask)
        # The same prefix with no padding at all must give identical logits
        unpadded = self._logits(model, ids[:, :5], torch.ones(1, 5, dtype=torch.long))

        assert torch.allclose(padded[:, :5], unpadded, atol=1e-6)

    def test_padding_tokens_cannot_influence_real_positions(self, model):
        model.eval()
        ids = torch.randint(0, 64, (1, 8))
        mask = torch.ones(1, 8, dtype=torch.long)
        mask[0, 5:] = 0
        altered = ids.clone()
        altered[0, 5:] = (ids[0, 5:] + 1) % 64

        assert torch.equal(self._logits(model, ids, mask)[:, :5],
                           self._logits(model, altered, mask)[:, :5])

    def test_an_all_padding_row_does_not_produce_nan(self, model):
        model.eval()
        ids = torch.randint(0, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[1] = 0  # a row with nothing to attend to

        logits = self._logits(model, ids, mask)

        assert torch.isfinite(logits).all()

    def test_batch_rows_are_independent(self, model):
        model.eval()
        ids = torch.randint(0, 64, (2, 6))
        mask = torch.ones(2, 6, dtype=torch.long)
        alone = self._logits(model, ids[:1], mask[:1])

        assert torch.allclose(self._logits(model, ids, mask)[:1], alone, atol=1e-6)

    def test_mask_is_causal_and_square(self, model):
        prepared = model._prepare_attention_mask(torch.ones(2, 5, dtype=torch.long))

        assert prepared.shape == (2, 1, 5, 5)
        assert not prepared[0, 0, 0, 1:].any()      # position 0 sees only itself
        assert prepared[0, 0, 4].all()              # the last sees everything

    def test_no_mask_means_causal_only(self, model):
        assert model._prepare_attention_mask(None) is None


class TestConstructorValidation:
    def test_rejects_head_count_that_does_not_divide_d_model(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=32, num_heads=5, dropout=0.0, layer_idx=0, num_layers=1)

    def test_model_rejects_indivisible_head_count(self):
        with pytest.raises(AssertionError):
            SmallTransformer(vocab_size=16, d_model=32, num_heads=7, num_layers=1,
                             d_ff=16, max_seq_len=8, dropout=0.0)

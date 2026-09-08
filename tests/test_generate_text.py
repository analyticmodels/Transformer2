"""
Tests for inference.generate_text.

Covers the sampling-loop failures that made local generation crash on the first token,
plus the contract that the return value is the continuation only.
"""

import pytest
import torch

from conftest import EOS_ID, PAD_ID, SPECIAL_IDS, FixedLogitsModel, ScriptedModel, decode_ids
from inference import generate_text


class TestReturnContract:
    def test_returns_only_the_continuation(self, stub_tokenizer):
        """
        Callers print f"{prompt}{generated}". Seeding the accumulator with the prompt
        makes the prompt appear twice and skews optimize_inference's metrics.
        """
        model = ScriptedModel(vocab_size=64, script=[20, 21, 22])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=3,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert decode_ids(out) == [20, 21, 22]

    def test_prompt_tokens_are_absent_from_the_output(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[30])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=1,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert not set(decode_ids(out)) & set(stub_tokenizer.prompt_ids)

    def test_respects_max_new_tokens(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[20, 21, 22, 23, 24, 25])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=4,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert len(decode_ids(out)) == 4

    def test_zero_new_tokens_yields_empty_output(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[20])

        assert generate_text(model, stub_tokenizer, "hi", max_new_tokens=0) == ""


class TestSamplingLoopShapes:
    """The concatenation of the sampled token used to produce a 3-D tensor."""

    def test_first_decode_step_does_not_crash(self, stub_tokenizer, tiny_model):
        out = generate_text(tiny_model, stub_tokenizer, "hi", max_new_tokens=1)

        assert isinstance(out, str)

    def test_many_steps_against_a_real_model(self, stub_tokenizer, tiny_model):
        out = generate_text(tiny_model, stub_tokenizer, "hi", max_new_tokens=10)

        assert len(decode_ids(out)) <= 10

    def test_context_grows_by_one_token_per_step(self, stub_tokenizer):
        """Each step must feed back exactly one new token."""
        model = ScriptedModel(vocab_size=64, script=[20, 21, 22])
        generate_text(model, stub_tokenizer, "hi", max_new_tokens=3,
                      repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert model.call_count == 3


class TestTopKClamping:
    @pytest.mark.parametrize("top_k", [1, 10, 64, 1000, 60000])
    def test_top_k_at_or_above_vocab_size_does_not_crash(self, stub_tokenizer, tiny_model, top_k):
        """optimize_inference sweeps top_k up to a user-supplied maximum."""
        out = generate_text(tiny_model, stub_tokenizer, "hi", max_new_tokens=3, top_k=top_k)

        assert isinstance(out, str)

    def test_top_k_disabled(self, stub_tokenizer, tiny_model):
        out = generate_text(tiny_model, stub_tokenizer, "hi", max_new_tokens=3, top_k=0)

        assert isinstance(out, str)


class TestSpecialTokenHandling:
    def test_eos_stops_generation_early(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[20, 21, EOS_ID, 22, 23])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=5,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert decode_ids(out) == [20, 21]

    def test_eos_is_not_emitted_into_the_output(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[EOS_ID])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=3,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert decode_ids(out) == []

    def test_special_tokens_are_never_sampled(self, stub_tokenizer, tiny_model):
        """<pad>/<s>/<unk> would otherwise appear verbatim and feed back as context."""
        torch.manual_seed(0)
        emitted = set()
        for _ in range(15):
            out = generate_text(tiny_model, stub_tokenizer, "hi", max_new_tokens=12,
                                top_k=0, top_p=1.0, temperature=2.0)
            emitted.update(decode_ids(out))

        assert not emitted & {i for i in SPECIAL_IDS if i != EOS_ID}

    def test_suppression_holds_when_the_model_strongly_wants_a_pad(self, stub_tokenizer):
        model = ScriptedModel(vocab_size=64, script=[PAD_ID])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=2,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert PAD_ID not in decode_ids(out)

    def test_generation_continues_when_the_tokenizer_has_no_eos(self, stub_tokenizer):
        stub_tokenizer.eos_token_id = None
        stub_tokenizer.special_token_ids = []
        model = ScriptedModel(vocab_size=64, script=[20, 21, 22])

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=3,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert len(decode_ids(out)) == 3


class TestRepetitionPenalty:
    """
    Most logits from these models are negative, and dividing a negative logit makes it
    larger -- boosting exactly the token meant to be suppressed. The penalty has to be
    sign-aware.
    """

    def _model_preferring_repeat(self):
        # index 40 (in the prompt) is preferred; index 41 is the alternative
        logits = [-30.0] * 64
        logits[40] = -1.0
        logits[41] = -2.0
        return FixedLogitsModel(logits)

    def test_penalty_suppresses_a_token_already_in_the_prompt(self, stub_tokenizer):
        stub_tokenizer.prompt_ids = [40]
        model = self._model_preferring_repeat()

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=1,
                            repetition_penalty=10.0, temperature=0.1, top_k=2, top_p=1.0)

        assert decode_ids(out) == [41], "penalised token was still chosen"

    def test_without_a_penalty_the_preferred_token_wins(self, stub_tokenizer):
        """Control: proves the previous assertion is caused by the penalty."""
        stub_tokenizer.prompt_ids = [40]
        model = self._model_preferring_repeat()

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=1,
                            repetition_penalty=1.0, temperature=0.1, top_k=2, top_p=1.0)

        assert decode_ids(out) == [40]

    def test_penalty_applies_to_previously_generated_tokens(self, stub_tokenizer):
        stub_tokenizer.prompt_ids = [5]
        model = self._model_preferring_repeat()

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=2,
                            repetition_penalty=10.0, temperature=0.1, top_k=2, top_p=1.0)

        tokens = decode_ids(out)
        assert len(tokens) == len(set(tokens)), "the same token was emitted twice"

    def test_higher_penalty_reduces_repetition_on_a_real_model(self, stub_tokenizer, tiny_model):
        def repeat_rate(penalty):
            total = 0.0
            for seed in range(20):
                torch.manual_seed(seed)
                tokens = decode_ids(generate_text(
                    tiny_model, stub_tokenizer, "hi", max_new_tokens=24,
                    repetition_penalty=penalty, temperature=0.7, top_k=32))
                total += 1 - len(set(tokens)) / max(1, len(tokens))
            return total / 20

        assert repeat_rate(3.0) < repeat_rate(1.0)


class TestContextWindow:
    def test_generation_past_max_seq_len_does_not_crash(self, stub_tokenizer):
        """
        The learned positional encoding only covers max_seq_len positions, so the
        context has to slide rather than grow without bound.
        """
        model = ScriptedModel(vocab_size=64, script=[20], max_seq_len=8)

        out = generate_text(model, stub_tokenizer, "hi", max_new_tokens=20,
                            repetition_penalty=1.0, top_k=0, top_p=1.0)

        assert len(decode_ids(out)) == 20

    def test_real_model_generates_beyond_its_window(self, stub_tokenizer, tiny_model):
        out = generate_text(tiny_model, stub_tokenizer, "hi",
                            max_new_tokens=tiny_model.max_seq_len * 2)

        assert isinstance(out, str)


class TestOllamaBranch:
    def test_uses_the_model_generate_method_and_slices_new_tokens(self):
        class OllamaTokenizerStub:
            def tokenize(self, text): return [1, 2, 3]
            def decode(self, ids): return f"NEW:{ids}"

        class OllamaModelStub(torch.nn.Module):
            def generate(self, input_ids, max_new_tokens=20, temperature=0.8, top_p=0.9):
                new = torch.tensor([[7, 8]], dtype=input_ids.dtype, device=input_ids.device)
                return torch.cat([input_ids, new], dim=1)

            def forward(self, *args, **kwargs):
                raise AssertionError("local sampling loop used for an Ollama model")

        out = generate_text(OllamaModelStub(), OllamaTokenizerStub(), "hi", use_ollama=True)

        assert out == "NEW:[7, 8]"

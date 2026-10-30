"""
Tests for the Ollama wrapper.

Every test stubs requests.post; nothing here contacts a live Ollama server.
"""

from unittest.mock import patch

import pytest
import torch

from transformer2.inference.ollama import OllamaModel, OllamaTokenizer


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def post():
    with patch('transformer2.inference.ollama.requests.post') as mocked:
        yield mocked


class TestGenerate:
    def test_output_is_prompt_plus_new_tokens(self, post):
        """
        Assigning into a row of input_ids.clone() cannot work: the row has a fixed
        width and the concatenation is longer.
        """
        post.return_value = FakeResponse({'response': 'text', 'tokens': [7, 8, 9]})
        model = OllamaModel('stub')
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

        out = model.generate(input_ids, max_new_tokens=3)

        assert out.shape == (1, 8)
        assert out[0].tolist() == [1, 2, 3, 4, 5, 7, 8, 9]

    def test_callers_slice_recovers_exactly_the_new_tokens(self, post):
        """inference.generate_text slices generated[0, input_ids.size(1):]."""
        post.return_value = FakeResponse({'response': 'text', 'tokens': [7, 8, 9]})
        model = OllamaModel('stub')
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        out = model.generate(input_ids, max_new_tokens=3)

        assert out[0, input_ids.size(1):].tolist() == [7, 8, 9]

    def test_batches_larger_than_one(self, post):
        post.return_value = FakeResponse({'response': 'text', 'tokens': [7, 8, 9]})
        model = OllamaModel('stub')

        out = model.generate(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
                             max_new_tokens=3)

        assert out.shape == (2, 6)

    def test_short_responses_are_padded_to_max_new_tokens(self, post):
        """Rows must stay the same length or the stack fails."""
        post.return_value = FakeResponse({'response': 'text', 'tokens': [7]})
        model = OllamaModel('stub')

        out = model.generate(torch.tensor([[1, 2]], dtype=torch.long), max_new_tokens=4)

        assert out.shape == (1, 6)

    def test_over_long_responses_are_truncated(self, post):
        post.return_value = FakeResponse({'response': 'text', 'tokens': list(range(50))})
        model = OllamaModel('stub')

        out = model.generate(torch.tensor([[1, 2]], dtype=torch.long), max_new_tokens=3)

        assert out.shape == (1, 5)

    def test_dtype_matches_the_input(self, post):
        post.return_value = FakeResponse({'response': 'text', 'tokens': [7, 8]})
        model = OllamaModel('stub')
        input_ids = torch.tensor([[1, 2]], dtype=torch.long)

        assert model.generate(input_ids, max_new_tokens=2).dtype == input_ids.dtype

    def test_generation_api_failure_raises_a_named_error(self, post):
        """
        generate() decodes the prompt before its try block, so the first post is the
        tokenizer's. Let that succeed and fail the generation call itself.
        """
        post.side_effect = [FakeResponse({'response': 'prompt'}), ConnectionError("no server")]
        model = OllamaModel('stub')

        with pytest.raises(RuntimeError, match="Failed to generate text with Ollama"):
            model.generate(torch.tensor([[1, 2]], dtype=torch.long), max_new_tokens=2)

    def test_total_outage_still_raises_rather_than_returning_junk(self, post):
        post.side_effect = ConnectionError("no server")
        model = OllamaModel('stub')

        with pytest.raises(RuntimeError):
            model.generate(torch.tensor([[1, 2]], dtype=torch.long), max_new_tokens=2)


class TestTokenizer:
    def test_tokenize_returns_the_token_list(self, post):
        post.return_value = FakeResponse({'tokens': [1, 2, 3]})

        assert OllamaTokenizer('stub').tokenize("hello") == [1, 2, 3]

    def test_decode_returns_the_response_text(self, post):
        post.return_value = FakeResponse({'response': 'hello'})

        assert OllamaTokenizer('stub').decode([1, 2, 3]) == 'hello'

    def test_vocab_size_is_cached_after_the_first_call(self, post):
        post.return_value = FakeResponse({'vocab_size': 32000})
        tokenizer = OllamaTokenizer('stub')

        assert tokenizer.get_vocab_size() == 32000
        assert tokenizer.get_vocab_size() == 32000
        assert post.call_count == 1

    def test_tokenize_failure_raises_a_named_error(self, post):
        post.side_effect = ConnectionError("no server")

        with pytest.raises(RuntimeError, match="Failed to tokenize"):
            OllamaTokenizer('stub').tokenize("hello")


class TestTrainingIncompatibility:
    def test_model_exposes_no_trainable_parameters(self):
        """
        This is why --use_ollama is rejected for training: an optimizer built over
        these parameters raises "empty parameter list".
        """
        assert list(OllamaModel('stub').parameters()) == []

    def test_optimizer_construction_fails(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            torch.optim.AdamW(OllamaModel('stub').parameters())

"""
Tests for the tokenizer wrappers.

The encode() return type and the special-token accessors are what inference.py
depends on, and both have been a source of breakage.
"""

import pytest
from tokenizers import ByteLevelBPETokenizer

from tokenizer import (
    BaseTokenizerWrapper,
    ByteLevelBPETokenizerWrapper,
    HuggingFaceTokenizerWrapper,
)


@pytest.fixture(scope='module')
def bpe_wrapper(tmp_path_factory):
    """A real ByteLevelBPE tokenizer trained on a tiny corpus."""
    corpus = tmp_path_factory.mktemp('tok') / 'corpus.txt'
    corpus.write_text(
        "\n".join([
            "the quick brown fox jumps over the lazy dog",
            "hello world this is a small training corpus",
            "transformers generate text one token at a time",
        ] * 20),
        encoding='utf-8',
    )
    raw = ByteLevelBPETokenizer()
    raw.train(
        files=[str(corpus)],
        vocab_size=300,
        min_frequency=1,
        special_tokens=ByteLevelBPETokenizerWrapper.SPECIAL_TOKENS,
    )
    return ByteLevelBPETokenizerWrapper(raw)


class TestByteLevelBPEWrapper:
    def test_encode_returns_mapping_not_encoding_object(self, bpe_wrapper):
        """inference.py indexes ['input_ids']; an Encoding object would break it."""
        encoded = bpe_wrapper.encode("hello world")

        assert isinstance(encoded, dict)
        assert set(encoded) == {'input_ids', 'attention_mask'}
        assert not hasattr(encoded, 'ids')

    def test_encode_ids_are_ints_of_equal_length_to_mask(self, bpe_wrapper):
        encoded = bpe_wrapper.encode("hello world")

        assert len(encoded['input_ids']) == len(encoded['attention_mask'])
        assert all(isinstance(i, int) for i in encoded['input_ids'])

    def test_encode_pads_to_max_length(self, bpe_wrapper):
        encoded = bpe_wrapper.encode("hello", max_length=12)

        assert len(encoded['input_ids']) == 12
        assert encoded['attention_mask'][-1] == 0, "padding must be masked out"

    def test_round_trip_decode(self, bpe_wrapper):
        text = "hello world"
        assert bpe_wrapper.decode(bpe_wrapper.encode(text)['input_ids']) == text

    def test_eos_token_id_resolves_from_vocab(self, bpe_wrapper):
        """
        </s> is index 1 in the training order, not the 2 the old code hardcoded
        (2 is <pad>). Sampling <pad> as a stop token would emit it into the output.
        """
        assert bpe_wrapper.get_eos_token_id() == 1

    def test_special_token_ids_cover_all_four(self, bpe_wrapper):
        assert sorted(bpe_wrapper.get_special_token_ids()) == [0, 1, 2, 3]

    def test_special_token_ids_are_never_none(self, bpe_wrapper):
        """A None in this list would blow up logit indexing during sampling."""
        assert all(i is not None for i in bpe_wrapper.get_special_token_ids())

    def test_vocab_size_is_positive(self, bpe_wrapper):
        assert bpe_wrapper.get_vocab_size() > 0

    def test_pad_token_id_is_the_pad_token_not_index_zero(self, bpe_wrapper):
        """
        Index 0 is <s> in this project's special-token order. Padding with it makes
        the pad symbol indistinguishable from a sequence start.
        """
        assert bpe_wrapper.get_pad_token_id() == 2
        assert bpe_wrapper.get_pad_token_id() != bpe_wrapper.get_eos_token_id()


class TestBatchEncoding:
    """encode_batch is the parallel path; it must agree with encode exactly."""

    TEXTS = [
        "the quick brown fox",
        "hello world",
        "transformers generate text one token at a time",
        "",
        "a",
    ]

    def test_matches_per_text_encode(self, bpe_wrapper):
        batched = bpe_wrapper.encode_batch(self.TEXTS)
        one_by_one = [bpe_wrapper.encode(t) for t in self.TEXTS]

        assert [b['input_ids'] for b in batched] == [s['input_ids'] for s in one_by_one]

    def test_attention_masks_match(self, bpe_wrapper):
        batched = bpe_wrapper.encode_batch(self.TEXTS)
        one_by_one = [bpe_wrapper.encode(t) for t in self.TEXTS]

        assert [b['attention_mask'] for b in batched] == [s['attention_mask'] for s in one_by_one]

    def test_preserves_order(self, bpe_wrapper):
        """Parallel encoding must not reorder results."""
        results = bpe_wrapper.encode_batch(self.TEXTS)

        for text, result in zip(self.TEXTS, results):
            assert result['input_ids'] == bpe_wrapper.encode(text)['input_ids']

    def test_returns_one_result_per_input(self, bpe_wrapper):
        assert len(bpe_wrapper.encode_batch(self.TEXTS)) == len(self.TEXTS)

    def test_handles_empty_input(self, bpe_wrapper):
        assert bpe_wrapper.encode_batch([]) == []

    def test_accepts_any_iterable(self, bpe_wrapper):
        """Callers pass list slices; a generator must work too."""
        assert len(bpe_wrapper.encode_batch(iter(self.TEXTS))) == len(self.TEXTS)

    def test_base_class_fallback_loops_over_encode(self):
        """Wrappers without a native batch path still satisfy the contract."""

        class Bare(BaseTokenizerWrapper):
            def encode(self, text): return {'input_ids': [len(text)], 'attention_mask': [1]}
            def decode(self, ids): return ""
            def get_vocab_size(self): return 1
            def get_pad_token_id(self): return 0

        assert Bare().encode_batch(['ab', 'cde']) == [
            {'input_ids': [2], 'attention_mask': [1]},
            {'input_ids': [3], 'attention_mask': [1]},
        ]


class TestBaseWrapperDefaults:
    def test_defaults_are_safe_for_wrappers_that_do_not_override(self):
        """generate_text calls these unconditionally, so they must always exist."""

        class Bare(BaseTokenizerWrapper):
            def encode(self, text): return {'input_ids': [], 'attention_mask': []}
            def decode(self, ids): return ""
            def get_vocab_size(self): return 1
            def get_pad_token_id(self): return 0

        bare = Bare()
        assert bare.get_eos_token_id() is None
        assert bare.get_special_token_ids() == []


class TestHuggingFaceWrapper:
    def test_delegates_special_tokens_to_underlying_tokenizer(self):
        class FakeHF:
            pad_token = '<pad>'
            eos_token = '</s>'
            eos_token_id = 99
            pad_token_id = 5
            all_special_ids = [5, 99]

            def __len__(self): return 128

        wrapper = HuggingFaceTokenizerWrapper(FakeHF())
        assert wrapper.get_eos_token_id() == 99
        assert wrapper.get_special_token_ids() == [5, 99]
        assert wrapper.get_vocab_size() == 128
        assert wrapper.get_pad_token_id() == 5

    def test_encode_batch_unpacks_the_batched_call(self):
        class FakeHF:
            pad_token = '<pad>'
            eos_token_id = 99
            pad_token_id = 5
            all_special_ids = [5, 99]

            def __call__(self, texts, **kwargs):
                assert isinstance(texts, list), "must pass the whole batch at once"
                return {
                    'input_ids': [[1, 2], [3]],
                    'attention_mask': [[1, 1], [1]],
                }

            def __len__(self): return 128

        result = HuggingFaceTokenizerWrapper(FakeHF()).encode_batch(['a', 'b'])

        assert result == [
            {'input_ids': [1, 2], 'attention_mask': [1, 1]},
            {'input_ids': [3], 'attention_mask': [1]},
        ]

    def test_special_token_delegation_unchanged(self):
        class FakeHF:
            pad_token = '<pad>'
            eos_token_id = 99
            pad_token_id = 5
            all_special_ids = [5, 99]

            def __len__(self): return 128

        wrapper = HuggingFaceTokenizerWrapper(FakeHF())

        assert wrapper.get_eos_token_id() == 99
        assert wrapper.get_special_token_ids() == [5, 99]
        assert wrapper.get_vocab_size() == 128
        assert wrapper.get_pad_token_id() == 5

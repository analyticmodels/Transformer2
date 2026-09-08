"""
Shared fixtures and test doubles for the Transformer2 suite.

The doubles here exist so that tests exercise the sampling, checkpoint and training
logic without loading the repository's 350 MB checkpoints or reaching the network.
"""

import re

import pytest
import torch
import torch.nn as nn

from transformer2.model.transformer import SmallTransformer

# Special token ids match tokenizer.py's ByteLevelBPETokenizerWrapper training order:
# ["<s>", "</s>", "<pad>", "<unk>"] -> 0, 1, 2, 3
BOS_ID, EOS_ID, PAD_ID, UNK_ID = 0, 1, 2, 3
SPECIAL_IDS = (BOS_ID, EOS_ID, PAD_ID, UNK_ID)


def decode_ids(text):
    """Recover the token ids from StubTokenizer.decode output."""
    return [int(match) for match in re.findall(r'<(\d+)>', text)]


class StubTokenizer:
    """
    Minimal stand-in implementing the wrapper contract from tokenizer.py.

    decode() renders each id as "<id>" so tests can assert on the exact tokens a
    generation produced rather than on opaque text.
    """

    def __init__(self, vocab_size=64, prompt_ids=(10, 11, 12), eos_token_id=EOS_ID,
                 special_token_ids=SPECIAL_IDS):
        self.vocab_size = vocab_size
        self.prompt_ids = list(prompt_ids)
        self.eos_token_id = eos_token_id
        self.special_token_ids = list(special_token_ids)

    def encode(self, text):
        return {
            'input_ids': list(self.prompt_ids),
            'attention_mask': [1] * len(self.prompt_ids),
        }

    def decode(self, ids):
        return "".join(f"<{int(i)}>" for i in ids)

    def get_vocab_size(self):
        return self.vocab_size

    def get_pad_token_id(self):
        return PAD_ID

    def get_eos_token_id(self):
        return self.eos_token_id

    def get_special_token_ids(self):
        return list(self.special_token_ids)


class FixedLogitsModel(nn.Module):
    """
    Returns the same logit vector at every position, so a test can state exactly
    what the sampler is choosing between.
    """

    def __init__(self, logits, max_seq_len=None):
        super().__init__()
        self.register_buffer('logits', torch.as_tensor(logits, dtype=torch.float32))
        self.call_count = 0
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        self.call_count += 1
        batch_size, seq_len = input_ids.shape
        vector = self.logits.to(input_ids.device)
        return vector.view(1, 1, -1).expand(batch_size, seq_len, -1).clone()


class ScriptedModel(nn.Module):
    """
    Emits a predetermined sequence of tokens, one per forward pass.

    The chosen token gets a dominant logit, so it survives temperature, top-k and
    top-p filtering and the sampled sequence is deterministic.
    """

    def __init__(self, vocab_size, script, max_seq_len=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.script = list(script)
        self.call_count = 0
        # A parameter so .to(device) has something to move
        self.marker = nn.Parameter(torch.zeros(1))
        if max_seq_len is not None:
            self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size), -20.0, device=input_ids.device
        )
        wanted = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        logits[:, -1, wanted] = 40.0
        return logits


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def tiny_model():
    """A real SmallTransformer small enough to run anywhere."""
    return SmallTransformer(
        vocab_size=64, d_model=32, num_heads=4, num_layers=2,
        d_ff=48, max_seq_len=16, dropout=0.0
    )


@pytest.fixture
def write_checkpoint(tmp_path):
    """Write a checkpoint file and return its path."""

    def _write(model, config=None, name='ckpt.pt', state_dict=None):
        payload = {'model_state_dict': state_dict if state_dict is not None else model.state_dict()}
        if config is not None:
            payload['config'] = config
        path = tmp_path / name
        torch.save(payload, path)
        return str(path)

    return _write


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="train_model uses torch.amp autocast/GradScaler on 'cuda'"
)

"""
Tests for training.py.

Two themes: the checkpoint config must describe the weights it sits beside (a wrong
head count there produces a silently mis-shaped model at inference time), and the CLI
hyperparameters must actually reach the model.
"""

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import training
from conftest import requires_cuda
from coreTransformer import SmallTransformer
from inference import infer_architecture
from training import (
    get_curriculum_length,
    model_config,
    save_checkpoint,
    sibling_checkpoint,
)


class ToyDataset(Dataset):
    def __init__(self, n=16, seq_len=8, vocab_size=32):
        self.n, self.seq_len, self.vocab_size = n, seq_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            'input_ids': ids,
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
            'labels': ids.clone(),
        }


@pytest.fixture
def loaders():
    return DataLoader(ToyDataset(), batch_size=2), DataLoader(ToyDataset(8), batch_size=2)


@pytest.fixture
def model():
    return SmallTransformer(vocab_size=32, d_model=16, num_heads=2, num_layers=2,
                            d_ff=32, max_seq_len=8, dropout=0.1)


class TestModelConfig:
    def test_records_the_real_architecture(self, model):
        config = model_config(model)

        assert config['num_heads'] == 2
        assert config['d_model'] == 16
        assert config['d_ff'] == 32
        assert config['num_layers'] == 2
        assert config['vocab_size'] == 32

    def test_head_count_is_not_a_hardcoded_default(self):
        """The old code wrote 8 regardless, which inference.py then trusted."""
        wide = SmallTransformer(vocab_size=32, d_model=64, num_heads=16, num_layers=1,
                                d_ff=32, max_seq_len=8, dropout=0.1)

        assert model_config(wide)['num_heads'] == 16

    def test_omits_attributes_the_model_does_not_expose(self):
        """
        Writing None would satisfy a key-presence check on the reading side and then
        crash on arithmetic. Absent is safer than present-and-None.
        """
        config = model_config(torch.nn.Linear(2, 2))

        assert config == {}

    def test_no_none_values_are_ever_recorded(self):
        class Partial(torch.nn.Module):
            d_model = 16
            num_heads = 2

        assert None not in model_config(Partial()).values()


class TestSaveCheckpoint:
    def _save(self, tmp_path, model, **kwargs):
        path = tmp_path / 'ckpt.pt'
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        save_checkpoint(str(path), 0, model, optimizer, scheduler,
                        torch.amp.GradScaler('cpu'), 1.0, [1.0], [1.0], **kwargs)
        return torch.load(path, map_location='cpu', weights_only=False)

    def test_writes_the_architecture_config(self, tmp_path, model):
        saved = self._save(tmp_path, model)

        assert saved['config']['num_heads'] == 2
        assert saved['config']['d_model'] == 16

    def test_extra_config_is_merged(self, tmp_path, model):
        saved = self._save(tmp_path, model,
                           extra_config={'optimizer_type': 'sgd', 'use_curriculum': True})

        assert saved['config']['optimizer_type'] == 'sgd'
        assert saved['config']['use_curriculum'] is True

    def test_contains_the_full_training_state(self, tmp_path, model):
        saved = self._save(tmp_path, model)

        assert {'epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'loss', 'train_losses', 'val_losses', 'scaler_state_dict'} <= set(saved)

    def test_config_round_trips_through_inference(self, tmp_path, model):
        """The end-to-end contract: what training writes, inference must be able to read."""
        saved = self._save(tmp_path, model)

        arch = infer_architecture(saved['model_state_dict'], config=saved['config'])

        assert arch['num_heads'] == 2
        SmallTransformer(dropout=0.1, **arch).load_state_dict(saved['model_state_dict'])


class TestCurriculumLength:
    def test_single_epoch_does_not_divide_by_zero(self):
        assert get_curriculum_length(0, num_epochs=1) > 0

    def test_starts_at_the_minimum(self):
        assert get_curriculum_length(0, num_epochs=10, min_length=64) == 64

    def test_never_exceeds_the_maximum(self):
        for epoch in range(20):
            assert get_curriculum_length(epoch, 10, 64, 512) <= 512

    def test_increases_with_epoch(self):
        lengths = [get_curriculum_length(e, 10, 64, 512) for e in range(6)]

        assert lengths == sorted(lengths)
        assert lengths[-1] > lengths[0]

    def test_reaches_maximum_by_the_halfway_point(self):
        assert get_curriculum_length(5, num_epochs=10, min_length=64, max_length=512) == 512


@requires_cuda
class TestTrainingLoop:
    def test_runs_and_returns_loss_history(self, model, loaders):
        trained, train_losses, val_losses = training.train_model(
            model, *loaders, num_epochs=2, accumulation_steps=2, checkpoint_freq=100,
            checkpoint_path='/dev/null')

        assert len(train_losses) == 2 and len(val_losses) == 2
        assert all(torch.isfinite(torch.tensor(l)) for l in train_losses)
        assert trained is model

    @pytest.mark.parametrize("optimizer_type,use_curriculum", [
        ('adamw', False), ('adam', False), ('sgd', False), ('sgd', True), ('adamw', True),
    ])
    def test_every_optimizer_and_schedule_combination_runs(
            self, model, loaders, tmp_path, optimizer_type, use_curriculum):
        """--optimizer sgd --use_curriculum used to fall into the wrong scheduler branch."""
        path = tmp_path / f'{optimizer_type}_{use_curriculum}.pt'

        training.train_model(model, *loaders, num_epochs=1, accumulation_steps=2,
                             optimizer_type=optimizer_type, use_curriculum=use_curriculum,
                             checkpoint_path=str(path), checkpoint_freq=100)

        assert path.exists()

    @pytest.mark.parametrize("optimizer_type,use_curriculum", [
        ('sgd', False),      # MultiStepLR: milestones would collapse onto step 0
        ('sgd', True),       # OneCycleLR: raises outright on total_steps=0
        ('adamw', True),     # OneCycleLR again, via the other optimizer
        ('adamw', False),    # cosine warmup
    ])
    def test_degenerate_step_count_does_not_break_the_scheduler(
            self, model, loaders, tmp_path, optimizer_type, use_curriculum):
        """
        With accumulation_steps above the batch count, total_steps computes to 0.
        Each scheduler fails differently there, so every branch has to be covered.
        """
        path = tmp_path / f'short_{optimizer_type}_{use_curriculum}.pt'

        training.train_model(model, *loaders, num_epochs=1, accumulation_steps=64,
                             optimizer_type=optimizer_type, use_curriculum=use_curriculum,
                             checkpoint_path=str(path), checkpoint_freq=100)

        assert path.exists()

    def test_saved_checkpoint_records_the_correct_head_count(self, model, loaders, tmp_path):
        path = tmp_path / 'best.pt'

        training.train_model(model, *loaders, num_epochs=1, accumulation_steps=2,
                             checkpoint_path=str(path), checkpoint_freq=100)

        assert torch.load(path, map_location='cpu', weights_only=False)['config']['num_heads'] == 2

    def test_failures_propagate_instead_of_reporting_success(self, model, loaders, tmp_path, monkeypatch):
        """
        A swallowed exception made callers print "training completed successfully"
        after a crash.
        """
        monkeypatch.chdir(tmp_path)

        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(model, 'forward', explode)

        with pytest.raises(RuntimeError, match="boom"):
            training.train_model(model, *loaders, num_epochs=1, accumulation_steps=2,
                                 checkpoint_path=str(tmp_path / 'x.pt'), checkpoint_freq=100)


class TestCommandLine:
    def _run_main(self, argv, tmp_path):
        captured = {}

        def fake_train_model(model=None, **kwargs):
            captured['model'] = model
            captured['kwargs'] = kwargs
            return model, [], []

        class FakeTokenizer:
            def get_vocab_size(self): return 100

        loaders = (MagicMock(), MagicMock(), FakeTokenizer())
        with patch('dataloader.get_dataloaders', return_value=loaders), \
             patch.object(training, 'train_model', side_effect=fake_train_model), \
             patch.object(sys, 'argv', ['training.py', f'--output_dir={tmp_path}'] + argv):
            training.main()
        return captured

    def test_architecture_flags_reach_the_model(self, tmp_path):
        """These were parsed and then ignored in favour of hardcoded literals."""
        captured = self._run_main(
            ['--d_model=512', '--num_heads=16', '--num_layers=4', '--d_ff=2048',
             '--dropout=0.2', '--max_length=128'], tmp_path)

        model = captured['model']
        assert (model.d_model, model.num_heads, model.num_layers) == (512, 16, 4)
        assert (model.d_ff, model.max_seq_len, model.dropout_p) == (2048, 128, 0.2)

    def test_defaults_are_applied_when_flags_are_absent(self, tmp_path):
        model = self._run_main([], tmp_path)['model']

        assert (model.d_model, model.num_heads, model.num_layers, model.d_ff) == (256, 8, 6, 1024)

    def test_vocab_size_comes_from_the_tokenizer(self, tmp_path):
        assert self._run_main([], tmp_path)['model'].vocab_size == 100

    def test_training_flags_are_forwarded(self, tmp_path):
        kwargs = self._run_main(['--epochs=7', '--patience=2', '--optimizer=sgd',
                                 '--use_curriculum'], tmp_path)['kwargs']

        assert kwargs['num_epochs'] == 7
        assert kwargs['patience'] == 2
        assert kwargs['optimizer_type'] == 'sgd'
        assert kwargs['use_curriculum'] is True

    def test_use_ollama_is_rejected_for_training(self, tmp_path):
        """An Ollama model is served over HTTP: no parameters, no gradients."""
        with pytest.raises(SystemExit):
            self._run_main(['--use_ollama'], tmp_path)

    def test_train_model_rejects_unknown_keywords(self):
        """
        *args/**kwargs used to absorb caller typos, which is how dead parameters went
        unnoticed.
        """
        with pytest.raises(TypeError):
            training.train_model(None, None, None, definitely_not_a_parameter=1)


class TestSiblingCheckpoint:
    """
    Periodic, interrupted and failed checkpoints must land beside the best model.
    They used to be written to the process working directory while only the best
    model honoured --output_dir, scattering one run across two locations.
    """

    def test_resolves_into_the_checkpoint_directory(self):
        assert sibling_checkpoint('/tmp/out/best_model.pt', 'checkpoint_epoch_5.pt') == \
            '/tmp/out/checkpoint_epoch_5.pt'

    def test_bare_filename_stays_bare(self):
        """checkpoint_path with no directory keeps the historical CWD behaviour."""
        assert sibling_checkpoint('best_model.pt', 'failed_checkpoint.pt') == \
            'failed_checkpoint.pt'

    @pytest.mark.parametrize("name", [
        'checkpoint_epoch_1.pt', 'interrupted_checkpoint.pt', 'failed_checkpoint.pt',
    ])
    def test_every_auxiliary_kind_follows_output_dir(self, tmp_path, name):
        resolved = sibling_checkpoint(str(tmp_path / 'best_model.pt'), name)

        assert Path(resolved).parent == tmp_path

    def test_train_model_writes_auxiliaries_beside_the_best_model(self, tmp_path):
        """
        Guard against the literals coming back: assert the source routes all three
        through the helper rather than naming a bare file.
        """
        source = inspect.getsource(training.train_model)

        for literal in ("'interrupted_checkpoint.pt'", "'failed_checkpoint.pt'"):
            assert f"sibling_checkpoint(checkpoint_path, {literal})" in source
        assert "sibling_checkpoint(checkpoint_path, f'checkpoint_epoch_" in source

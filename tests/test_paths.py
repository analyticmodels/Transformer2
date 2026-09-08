"""
Tests for artifact path resolution.

Every path in this project was once relative to the process working directory, so a
script run from anywhere but the repository root silently read the wrong data and
wrote checkpoints somewhere unexpected. These tests pin down that resolution is
anchored to the project, overridable for tests and deployment, and never lands
inside site-packages.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import transformer2.paths as paths

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Resolution must not inherit overrides from the developer's shell."""
    for var in (paths.HOME_ENV, paths.DATA_ENV, paths.CHECKPOINT_ENV,
                paths.TOKENIZER_ENV, paths.OUTPUT_ENV):
        monkeypatch.delenv(var, raising=False)


class TestProjectRoot:
    def test_finds_the_repository_root(self):
        assert paths.project_root() == REPO_ROOT

    def test_independent_of_the_working_directory(self, monkeypatch, tmp_path):
        """The whole point: a foreign CWD must not change where artifacts live."""
        monkeypatch.chdir(tmp_path)

        assert paths.project_root() == REPO_ROOT

    def test_resolves_from_a_foreign_cwd_in_a_fresh_interpreter(self, tmp_path):
        """monkeypatch.chdir cannot catch import-time resolution; a subprocess can."""
        result = subprocess.run(
            [sys.executable, "-c",
             "from transformer2.paths import data_dir; print(data_dir())"],
            cwd=tmp_path, capture_output=True, text=True,
        )

        assert result.stdout.strip() == str(REPO_ROOT / "data"), result.stderr

    def test_home_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.HOME_ENV, str(tmp_path))

        assert paths.project_root() == tmp_path.resolve()

    def test_never_resolves_inside_the_interpreter_prefix(self):
        """A venv rebuild would take gigabytes of corpora with it."""
        assert not paths._is_inside_prefix(paths.project_root())


class TestArtifactDirectories:
    @pytest.mark.parametrize("func,name", [
        (paths.data_dir, "data"),
        (paths.checkpoint_dir, "checkpoints"),
        (paths.output_dir, "outputs"),
    ])
    def test_default_location(self, func, name):
        assert func() == REPO_ROOT / name

    def test_tokenizer_dir_is_per_corpus(self):
        assert paths.tokenizer_dir("bluesky") == REPO_ROOT / "tokenizers" / "bluesky"

    def test_tokenizer_dir_without_a_corpus_is_the_parent(self):
        assert paths.tokenizer_dir() == REPO_ROOT / "tokenizers"

    @pytest.mark.parametrize("env,func", [
        (paths.DATA_ENV, paths.data_dir),
        (paths.CHECKPOINT_ENV, paths.checkpoint_dir),
        (paths.OUTPUT_ENV, paths.output_dir),
    ])
    def test_each_directory_is_individually_overridable(self, monkeypatch, tmp_path, env, func):
        monkeypatch.setenv(env, str(tmp_path))

        assert func() == tmp_path.resolve()

    def test_tokenizer_override_keeps_the_per_corpus_subdirectory(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.TOKENIZER_ENV, str(tmp_path))

        assert paths.tokenizer_dir("reddit") == tmp_path.resolve() / "reddit"

    def test_overrides_are_read_per_call_not_at_import(self, monkeypatch, tmp_path):
        """
        Module constants evaluated at import cannot be redirected by a test. These
        are functions precisely so monkeypatch.setenv works after import.
        """
        before = paths.data_dir()
        monkeypatch.setenv(paths.DATA_ENV, str(tmp_path))

        assert paths.data_dir() != before


class TestEnsure:
    def test_creates_the_parent_of_a_file(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "corpus.parquet"

        paths.ensure(target)

        assert target.parent.is_dir()
        assert not target.exists(), "ensure() must not create the file itself"

    def test_creates_a_directory_given_directly(self, tmp_path):
        target = tmp_path / "checkpoints"

        paths.ensure(target)

        assert target.is_dir()

    def test_returns_the_path_for_chaining(self, tmp_path):
        target = tmp_path / "a" / "b.pt"

        assert paths.ensure(target) == target

    def test_is_idempotent(self, tmp_path):
        target = tmp_path / "twice" / "file.txt"

        paths.ensure(target)
        paths.ensure(target)  # must not raise

        assert target.parent.is_dir()


class TestNoImportSideEffects:
    def test_importing_paths_creates_nothing(self, tmp_path):
        """
        With a mis-resolved root, an import-time mkdir would scatter directories --
        including into site-packages on a wheel install.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import transformer2.paths"],
            cwd=tmp_path, capture_output=True, text=True,
        )

        assert result.returncode == 0, result.stderr
        assert list(tmp_path.iterdir()) == []

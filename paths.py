"""
Where the project keeps its artifacts.

Every path in this project used to be relative to the process working directory, so
running a script from anywhere but the repository root silently read the wrong data
and wrote checkpoints somewhere unexpected -- with no import error to warn you. This
module is the single place that answers "where does this live", anchored to the
project itself rather than to the caller's shell.

Resolution order for the project root:

1. ``TRANSFORMER2_HOME``, if set.
2. The first ancestor of this file containing both ``pyproject.toml`` and
   ``src/transformer2`` (or this module itself, before the package move). Requiring
   both markers stops a stray ``pyproject.toml`` further up the tree from matching,
   and candidates inside the interpreter prefix are rejected so an installed copy
   never writes into site-packages.
3. The same search upward from the working directory, for a wheel-installed package
   invoked inside a checkout.
4. The XDG data directory, so a genuinely installed package still has somewhere to
   write that survives a virtualenv rebuild.

Directories are exposed as functions rather than module constants: constants are
evaluated once at import, which makes them impossible to redirect from a test with
``monkeypatch.setenv``. Nothing here creates a directory -- call :func:`ensure` at
the point of writing.
"""

import os
import sys
from pathlib import Path

HOME_ENV = "TRANSFORMER2_HOME"
DATA_ENV = "TRANSFORMER2_DATA_DIR"
CHECKPOINT_ENV = "TRANSFORMER2_CHECKPOINT_DIR"
TOKENIZER_ENV = "TRANSFORMER2_TOKENIZER_DIR"
OUTPUT_ENV = "TRANSFORMER2_OUTPUT_DIR"


def _is_inside_prefix(candidate):
    """True if candidate sits inside the interpreter prefix (i.e. site-packages)."""
    for prefix in {sys.prefix, sys.base_prefix}:
        try:
            if candidate.is_relative_to(Path(prefix).resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _looks_like_project_root(candidate):
    """
    Recognise the project root.

    The installed/packaged form carries a pyproject.toml beside the package source;
    requiring both markers stops an unrelated pyproject.toml higher up the tree from
    matching. A working checkout is also recognised by its VCS directory, which is
    what identifies the root before the package layout exists.
    """
    if (candidate / "pyproject.toml").is_file():
        if (candidate / "src" / "transformer2").is_dir():
            return True
    return (candidate / ".git").is_dir() and (candidate / "paths.py").is_file()


def _search_upward(start):
    for candidate in [start, *start.parents]:
        if _looks_like_project_root(candidate) and not _is_inside_prefix(candidate):
            return candidate
    return None


def project_root():
    """
    Locate the project root.

    Returns:
        pathlib.Path: The directory artifacts are resolved against
    """
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()

    found = _search_upward(Path(__file__).resolve().parent)
    if found is not None:
        return found

    found = _search_upward(Path.cwd().resolve())
    if found is not None:
        return found

    # Installed with no checkout in sight. Never site-packages: a virtualenv rebuild
    # would take gigabytes of corpora with it.
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "transformer2"


def _resolve(env_var, default_name):
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / default_name


def data_dir():
    """Directory holding collected and postprocessed corpora (parquet)."""
    return _resolve(DATA_ENV, "data")


def checkpoint_dir():
    """Directory holding model checkpoints (.pt)."""
    return _resolve(CHECKPOINT_ENV, "checkpoints")


def tokenizer_dir(dataset=None):
    """
    Directory holding trained tokenizers.

    Args:
        dataset (str, optional): Corpus name. Returns the per-corpus subdirectory
            when given, and the parent otherwise.

    Returns:
        pathlib.Path
    """
    root = _resolve(TOKENIZER_ENV, "tokenizers")
    return root / dataset if dataset else root


def output_dir():
    """Directory for generated output such as tuned inference parameters."""
    return _resolve(OUTPUT_ENV, "outputs")


def ensure(path):
    """
    Create the directory a file is about to be written into.

    Args:
        path (str or pathlib.Path): A file path, or a directory to create

    Returns:
        pathlib.Path: The path unchanged, for chaining into open()/to_parquet()
    """
    path = Path(path)
    parent = path if path.suffix == "" else path.parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    return path

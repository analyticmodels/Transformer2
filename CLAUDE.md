# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project virtualenv: `.venv/bin/python`, not a bare `python`/`python3`.
This machine has more than one Python 3.14 sharing `~/.local/lib/python3.14/site-packages`,
so imports outside the venv resolve inconsistently (`torch` imports under one and fails
under another).

- Install the package: `.venv/bin/pip install -e .`
- With the test suite: `.venv/bin/pip install -e ".[dev]"`

## Build/Run Commands

Installed as a package: `.venv/bin/pip install -e ".[dev]"`. The `t2-*` commands
work from any directory.

- Build a tokenizer: `.venv/bin/t2-tokenizer --dataset_type bluesky`
- Collect data: `.venv/bin/t2-collect --dataset tinystories`
- Postprocess: `.venv/bin/t2-postprocess tinystories --max_tokens 320`
  (or `t2-postprocess reddit --max_tokens 512`; the two take different flags)
- Train: `.venv/bin/t2-train --dataset tinystories --max_length 320`
- Generate: `.venv/bin/t2-generate --prompt "Your prompt here"`
- Tune sampling: `.venv/bin/t2-tune --prompt "..."` -> `outputs/inference_params.json`
- Generate with tuned params: `.venv/bin/t2-generate --prompt "..." --params outputs/inference_params.json`
- End-to-end demo: `.venv/bin/t2-demo --dataset bluesky --skip_data_collection`
- Run tests: `.venv/bin/python -m pytest`

Training requires a GPU (the loop uses `torch.amp` autocast/`GradScaler` on `cuda`).
Inference runs on either.

## Code Style Guidelines

- Imports order: standard library → third-party packages → local modules
- Class style: Follow Google docstring format with comprehensive documentation
- Naming: Descriptive variable/class names, CamelCase for classes, snake_case for functions/variables
- Error handling: Use try/except with specific error types and informative messages
- Model architecture: Proper PyTorch nn.Module subclassing with __init__ and forward methods
- PyTorch patterns: Use device management, gradient clipping, mixed precision training

## Invariants worth preserving

These encode past bugs; the test suite guards each one.

- **Head count cannot be inferred from weights.** Q/K/V projections are `d_model x d_model`
  for any `num_heads`, so a wrong value passes `load_state_dict` and silently generates
  noise. Never guess it — `infer_architecture` raises and asks for `--num_heads`.
- **Checkpoint configs must be read off the model**, not off the training arguments, and
  must omit missing values rather than record `None` (a present-but-`None` entry defeats
  the reader's fallbacks).
- **`generate_text()` returns the continuation only.** Callers prepend the prompt
  themselves; including it double-prints and skews `inference/tuning.py` metrics.
- **The repetition penalty must be sign-aware.** Most logits are negative, and dividing a
  negative logit boosts the token it is meant to suppress.
- **Load failures must raise.** Substituting a freshly initialised model makes callers
  report success while generating from random weights.
- **Tokenizer wrappers return a mapping from `encode()`** (`['input_ids']`), not an
  `Encoding` object, and expose `get_eos_token_id()` / `get_special_token_ids()` rather
  than hardcoded ids.
- **`data/loaders.py` must not be renamed `datasets.py`.** It imports the
  HuggingFace `datasets` library; a sibling of the same name is the collision this
  layout exists to avoid. Likewise never add an `__init__.py` under `tokenizers/`,
  which would shadow the `tokenizers` package.
- **A tokenizer is identified by its fingerprint, not its size.** Every corpus here
  trains to 50000 tokens, so vocab size cannot tell them apart. `tokenizers/<corpus>/
  meta.json` carries a sha256 of vocab.json; training records it in the checkpoint and
  `load_model` refuses a mismatch.
- **`get_dataloaders` must stay deferred inside `trainer.main()`.** The CLI test
  patches it before calling main(); hoisting the import would run the real loader
  against an 800 MB parquet.
- **Reddit needs postprocessing.** The raw `combined_text` carries
  `Title:`/`Content:`/`Comment:` scaffolding and runs to tens of thousands of tokens.
  Training reads `data/reddit_text.parquet` when present and warns when it does not.
- **Token counts are not additive across concatenation.** The separator costs tokens
  and BPE merges across the join, so packed chunks must be re-measured, not summed.
- **Ollama is inference-only** — no parameters, no gradients. `t2-train --use_ollama`
  must keep failing loudly.

## Credentials

Secrets live in `.env` (gitignored); `.env.example` is the committed template.
`config.py` resolves them (`reddit_credentials()`, `huggingface_token()`).
Never hardcode a key, token or secret, and never add one to a docstring or a
notebook cell -- notebook *outputs* count too, since printed URLs can embed them.

## Repository hygiene

Checkpoints (`*.pt`), datasets (`data/*.parquet`), `__pycache__/` and `.venv/` are
gitignored. Do not commit them; several checkpoints are ~350 MB.

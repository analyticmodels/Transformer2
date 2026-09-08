# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project virtualenv: `.venv/bin/python`, not a bare `python`/`python3`.
This machine has more than one Python 3.14 sharing `~/.local/lib/python3.14/site-packages`,
so imports outside the venv resolve inconsistently (`torch` imports under one and fails
under another).

- Install runtime deps: `.venv/bin/pip install -r requirements.txt`
- Install test deps too: `.venv/bin/pip install -r requirements-dev.txt`

## Build/Run Commands

- Build the tokenizer: `.venv/bin/python tokenizer.py --dataset_type bluesky`
- Collect data: `.venv/bin/python data_collection.py --dataset bluesky --max_samples 10000`
- TinyStories: `.venv/bin/python data_collection.py --dataset tinystories` then `.venv/bin/python tinystories_postprocess.py --max_tokens 320`
- Postprocess Reddit (required before training on it): `.venv/bin/python reddit_postprocess.py --max_tokens 512`
- Train a model: `.venv/bin/python training.py --dataset [reddit|bluesky|imdb] --batch_size 8 --epochs 30`
- End-to-end demo: `.venv/bin/python train_demo.py --dataset bluesky --skip_data_collection`
- Generate text: `.venv/bin/python inference.py --prompt "Your prompt here"`
- Tune generation params: `.venv/bin/python optimize_inference.py --prompt "..."`
- Generate with tuned params: `.venv/bin/python generate_text.py --prompt "..." --params inference_params.json`
- Run tests: `.venv/bin/python -m pytest`

Training requires a GPU (the loop uses `torch.amp` autocast/`GradScaler` on `cuda`).
Inference runs on either.

## Project Structure

- Core model: `coreTransformer.py` (transformer architecture)
- Data handling: `dataloader.py` (dataset classes and loaders)
- Data collection: `data_collection.py` (Reddit and Bluesky sources)
- Reddit cleaning/chunking: `reddit_postprocess.py` -> `data/reddit_text.parquet`
- TinyStories cleaning: `tinystories_postprocess.py` -> `data/tinystories_text.parquet`
- Shared chunking: `text_chunking.py`
- Tokenization: `tokenizer.py` (tokenizer wrappers plus a CLI to build the tokenizer)
- Training: `training.py`
- Inference: `inference.py` (loading and generation), `generate_text.py`, `optimize_inference.py`
- Ollama backend: `ollama_integration.py` (inference only)
- Credentials: `config.py` (reads `.env`)
- Tests: `tests/`
- `archive/` holds superseded scripts (`trainingSetup.py`, `curriculumTraining.py`,
  `inferenceTest.py`, the per-dataset training demos). They are kept for reference only —
  do not treat them as entry points or update them alongside changes.

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
  themselves; including it double-prints and skews `optimize_inference` metrics.
- **The repetition penalty must be sign-aware.** Most logits are negative, and dividing a
  negative logit boosts the token it is meant to suppress.
- **Load failures must raise.** Substituting a freshly initialised model makes callers
  report success while generating from random weights.
- **Tokenizer wrappers return a mapping from `encode()`** (`['input_ids']`), not an
  `Encoding` object, and expose `get_eos_token_id()` / `get_special_token_ids()` rather
  than hardcoded ids.
- **Reddit needs postprocessing.** The raw `combined_text` carries
  `Title:`/`Content:`/`Comment:` scaffolding and runs to tens of thousands of tokens.
  Training reads `data/reddit_text.parquet` when present and warns when it does not.
- **Token counts are not additive across concatenation.** The separator costs tokens
  and BPE merges across the join, so packed chunks must be re-measured, not summed.
- **Ollama is inference-only** — no parameters, no gradients. `training.py --use_ollama`
  must keep failing loudly.

## Credentials

Secrets live in `.env` (gitignored); `.env.example` is the committed template.
`config.py` resolves them (`reddit_credentials()`, `huggingface_token()`).
Never hardcode a key, token or secret, and never add one to a docstring or a
notebook cell -- notebook *outputs* count too, since printed URLs can embed them.

## Repository hygiene

Checkpoints (`*.pt`), datasets (`data/*.parquet`), `__pycache__/` and `.venv/` are
gitignored. Do not commit them; several checkpoints are ~350 MB.

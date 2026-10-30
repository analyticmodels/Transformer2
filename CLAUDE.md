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
  (or `t2-postprocess reddit --max_tokens 512`, `t2-postprocess gutenberg
  --max_tokens 512`; the subcommands take different flags)
- Collect Gutenberg: `.venv/bin/t2-collect --dataset gutenberg --max_books 20`
  (omit `--max_books` for the whole ~62k-book English subset; it resumes)
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
- **Gutenberg is streamed, never loaded whole.** The English subset is ~24 GB of
  plain text and ~5B tokens (measured, not estimated; the 1-2 TB figure quoted for
  Gutenberg is the complete all-format mirror, not this subset). Collection flushes parquet row groups as books arrive, postprocessing
  reads back with `iter_batches`, deduplication keeps digests rather than rows,
  and the tokenizer branch stops at `GUTENBERG_TOKENIZER_ROWS`. `pd.read_parquet`
  on the whole corpus, or `drop_duplicates` over it, exhausts memory.
- **Gutenberg chunks one book at a time.** `chunk_segments` is called per book so
  a chunk never runs from the end of one work into the start of another, the same
  rule TinyStories applies per story. Books with no recognisable Gutenberg
  markers are skipped, so the license text never enters the corpus.
- **Gutenberg downloads come from a mirror, not gutenberg.org**, which blocks
  bulk crawling. Raw books are cached under `data/gutenberg_raw/` so a run that
  takes a day is resumable; never remove that cache check.
- **The Gutenberg corpus is sampled at read time, not after loading.** Reading it
  with `pd.read_parquet` holds the Arrow buffers and the decoded Python strings at
  once -- ~44 GB before tokenizing -- and `t2-train --max_texts 0` was OOM-killed at
  82 GB RSS on a 96 GB machine. `sample_parquet_texts` streams row batches so peak
  memory follows `--max_texts` (~5.5 GB per million chunks), not the file size.
- **Attention must stay causal.** `labels[i]` is `input_ids[i+1]`, so a position that
  can attend to `i+1` reads its own target. Attention here was bidirectional once (a
  `[B,1,1,S]` padding mask under a comment claiming to build a causal one), which lets
  training drive the loss down without learning to predict. `_prepare_attention_mask`
  now folds causal and padding masking into one bool `[B,1,S,S]` mask, and
  `TestCausalMasking` in `tests/test_core_transformer.py` asserts bit-identical logits
  at every earlier position when a later token changes.
- **Attention goes through `F.scaled_dot_product_attention`.** The hand-rolled version
  materialised `[B, num_heads, S, S]`, which was the memory ceiling. Do not reintroduce
  an explicit `softmax(QK^T)`; dropout belongs in `dropout_p`, inside the fused kernel.
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

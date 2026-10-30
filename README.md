# Transformer2

A PyTorch implementation of a transformer model for natural language generation. This project provides a complete workflow for downloading/collecting training data, training custom transformer models, and running inference.

## Features

- Custom decoder-only transformer with multi-head attention and pre-norm blocks
- Data collection from multiple sources (Project Gutenberg, TinyStories, Reddit, Bluesky)
- Curriculum learning with sequence length progression
- Mixed precision training with gradient accumulation
- Text generation with temperature, top-k, top-p and repetition penalty
- Grid search over generation parameters
- Optional Ollama backend for inference
- Complete training and inference pipeline

## Project Structure

The package lives under `src/` and is installed with `pip install -e .`, so it
imports from any working directory.

```
src/transformer2/
├── paths.py                 Where artifacts live, anchored to the project root
├── config.py                Credentials from the environment / .env
├── model/transformer.py     The transformer architecture
├── tokenization/bpe.py      Byte-level BPE wrappers, one tokenizer per corpus
├── data/
│   ├── collection.py        Download Bluesky / TinyStories, fetch Reddit
│   ├── postprocess.py       t2-postprocess dispatcher
│   ├── gutenberg.py         Mirror downloads, boilerplate stripping, book chunking
│   ├── reddit.py            Strip discussion scaffolding, chunk by post
│   ├── tinystories.py       Normalise, deduplicate, length-bound
│   ├── chunking.py          Corpus-agnostic chunking shared by all three
│   └── loaders.py           Datasets and DataLoaders
├── training/
│   ├── trainer.py           Training loop and checkpointing
│   └── demo.py              End-to-end collect-then-train demo
└── inference/
    ├── generate.py          Checkpoint loading and sampling
    ├── tuning.py            Grid search over generation parameters
    └── ollama.py            Ollama backend (inference only)
```

Artifacts sit outside the source tree: `data/` (corpora), `checkpoints/`,
`tokenizers/<corpus>/`, `outputs/`. `tests/` holds the suite, `tools/` holds
unpackaged dev scripts, and `archive/` holds superseded code that is kept for
reference and is not runnable.

## Installation

1. Clone the repository.

2. Create a virtual environment. Pin the interpreter explicitly if your machine has
   more than one Python of the same minor version — several installations sharing
   `~/.local/lib/pythonX.Y/site-packages` will import each other's packages and fail
   in confusing ways:

   ```bash
   /usr/bin/python3 -m venv .venv
   ```

3. Install dependencies:

   ```bash
   .venv/bin/pip install -e .
   ```

   This installs the dependencies and puts the seven `t2-*` commands on your PATH.
   Add the test suite with `.venv/bin/pip install -e ".[dev]"`.

4. Configure credentials (only needed for Reddit collection or private Hugging Face
   models):

   ```bash
   cp .env.example .env
   # then edit .env
   ```

`torch` ships with CUDA support from PyPI on Linux. Training uses
`torch.amp` autocast and `GradScaler` on `cuda`, so a GPU is required to train;
inference runs on CPU or GPU.

## Credentials

Secrets are read from the environment, optionally populated from a local `.env` file.
No API key, token or secret belongs anywhere in the source tree.

- `.env` is gitignored; `.env.example` is the committed template listing every
  variable the project reads.
- Exported environment variables take precedence over `.env`, so CI and shell
  exports work without a file.
- `config.py` centralises resolution (`reddit_credentials()`, `huggingface_token()`).

| Variable | Used by |
|---|---|
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | `t2-collect --dataset reddit`, `t2-demo` |
| `REDDIT_USER_AGENT` | Reddit collection (optional; has a default) |
| `HUGGINGFACE_TOKEN` | `tools/check_hf_token.py`, private model downloads |

The Reddit CLI flags (`--client_id`, `--client_secret`) still work but are
discouraged: arguments are visible in shell history and in the process list.

## Usage

The `t2-*` commands are installed into the virtual environment. Prefix them with
`.venv/bin/` or run `source .venv/bin/activate` first. They work from any directory.

### Data Collection

```bash
# TinyStories: ~2.1M synthetic short stories (~250M tokens)
t2-collect --dataset tinystories
t2-tokenizer --dataset_type tinystories     # each corpus needs its own tokenizer
t2-postprocess tinystories --max_tokens 320

# Collect Bluesky data
t2-collect --dataset bluesky --max_samples 10000

# Collect Reddit data
t2-collect --dataset reddit   # credentials from .env

# Project Gutenberg: ~62k public-domain English books (~24 GB, ~5B tokens)
t2-collect --dataset gutenberg --max_books 20    # try the mirror first
t2-collect --dataset gutenberg                   # then the whole subset
```

`--max_samples` defaults to collecting **everything**. Pass it explicitly to cap a
download; omitting it no longer silently truncates an existing corpus.

Collected data is written to `data/` as a parquet file. Note that re-running
collection **overwrites** the existing file, so a smaller `--max_samples` than a
previous run will shrink the corpus.

#### Project Gutenberg

Gutenberg is the only long-form corpus here: edited literary prose rather than
posts or synthetic short stories, which is what gives the model paragraph-scale
structure and a vocabulary large enough to justify 50k BPE tokens.

The full English subset is ~62,000 books, **~24 GB of plain text and roughly 5
billion tokens** (measured across a 56,000-book download, not estimated). Note that
the 1-2 TB figure usually quoted for Project Gutenberg is the *complete* mirror --
every format and language, images, audio and superseded revisions. This collector
fetches a single plain-text file per English book, which is a small fraction of it.

At that size the collector behaves differently from the others:

- **It streams.** Books are flushed to parquet in row groups as they arrive, and
  postprocessing reads them back in batches. The corpus is never held in memory.
- **It downloads from a mirror.** `gutenberg.org` blocks bulk crawling and directs
  heavy users to its mirrors, so books come from `--mirror` (default
  `https://mirrors.xmission.com/gutenberg`) at `--delay` seconds per request.
  Only the catalog is fetched from the main site.
- **It resumes.** Raw books are cached under `data/gutenberg_raw/` and never
  re-fetched. A full run takes about a day at the default 1 req/s; interrupt it
  freely and re-run the same command to continue.

```bash
t2-collect --dataset gutenberg --delay 1.0
t2-tokenizer --dataset_type gutenberg
t2-postprocess gutenberg --max_tokens 512
```

Postprocessing strips the Project Gutenberg header, footer and license, undoes
the ~70-column hard wrapping so paragraphs are single lines, and chunks each book
**on its own** — a chunk never runs from the end of one work into the start of an
unrelated one. Books with no recognisable Gutenberg markers are skipped rather
than ingested, so the license text never reaches the corpus.

Because the corpus is far larger than VRAM, `t2-train --dataset gutenberg` samples
500,000 chunks by default. `--max_texts` raises or removes the cap; rows are drawn
from across the corpus, not off the front, so a sample is a cross-section rather
than the first few hundred books.

Peak system RAM is roughly 5.5 GB per million chunks (the decoded text plus the
`[N, max_length]` int32 token array), measured on the 11.76M-chunk corpus:

| `--max_texts` | tokens | peak RSS |
|---|---|---|
| 500,000 (default) | ~0.2B | ~3 GB |
| 4,000,000 | ~1.6B | ~29 GB |
| 8,000,000 | ~3.2B | ~57 GB |
| 0 (all 11.76M) | ~4.8B | ~85 GB |

`--max_texts 0` needs more RAM than a 96 GB machine has once the DataLoader workers
are up; 4-8 million is the practical range there. Exceeding it gets the process
killed by the kernel OOM killer, which looks like training dying without a Python
traceback -- check `journalctl -k | grep -i oom` if that happens.

Note that removing the Gutenberg header — which training requires — means the
resulting corpus may not be redistributed under the Project Gutenberg trademark.
It is fine for local training.

#### Reddit postprocessing (required)

The collector writes one row per post, concatenating every comment behind
`Title:`/`Content:`/`Comment:` labels. Those labels are scaffolding the model would
learn to reproduce, and a single thread can run to tens of thousands of tokens, so
almost all of it would be truncated away. Rewrite the corpus before training:

```bash
t2-postprocess reddit --max_tokens 512   # match your training --max_length
```

This strips the labels, URLs, markdown, `/u/` and `/r/` references and deleted
comments, then packs the remaining prose into chunks that fill `--max_tokens`
without ending mid-word. Packing stays within a single post, so a chunk reads as one
discussion rather than unrelated fragments. It writes `data/reddit_text.parquet`
with a single `text` column, which training picks up automatically.

Measured on a 3,937-post collection (7.0M tokens), chunking preserves far more of
the corpus than truncating the raw rows:

| `--max_tokens` | chunks | mean tokens | fill | corpus retained |
|---|---|---|---|---|
| 128 | 58,771 | 100 | 78% | 11.7× more than truncation |
| 256 | 28,348 | 211 | 82% | 6.1× |
| 512 | 14,450 | 419 | 82% | 3.3× |
| 1024 | 8,131 | 747 | 73% | 2.0× |

`--max_tokens 512` is a good default: it retains 87% of all tokens at 82% fill.
Titles are excluded as metadata; pass `--keep_titles` to include them.

### Tokenizer

Each corpus gets its own tokenizer under `tokenizers/<corpus>/`, recorded with a
`meta.json` that fingerprints the vocabulary. Training writes that fingerprint into
the checkpoint, and inference refuses a checkpoint whose tokenizer no longer matches:

```bash
t2-tokenizer --dataset_type bluesky --vocab_size 50000
```

A corpus without its own tokenizer is an error, not a silent fallback: building
one is a required step before training or postprocessing a new corpus.

Training and inference both create it on demand if it is missing, but doing it
explicitly is faster and makes the vocabulary size a deliberate choice. The
tokenizer's vocabulary must match the checkpoint's embedding table — inference
refuses to load a mismatched pair rather than producing garbage.

### Training

```bash
# Standard training
t2-train --dataset bluesky --batch_size 8 --epochs 30

# Curriculum learning
t2-train --dataset reddit --batch_size 8 --epochs 30 --use_curriculum

# Gutenberg, capped to what fits in memory (default 500k chunks)
t2-train --dataset gutenberg --max_length 512 --max_texts 4000000

# Custom architecture
t2-train --dataset bluesky --d_model 512 --num_heads 16 --num_layers 8 --d_ff 2048
```

The best model by validation loss is written to `checkpoints/best_model.pt`, with
periodic snapshots as `checkpoint_epoch_N.pt` alongside it. Every checkpoint records the architecture it
was trained with, so inference can rebuild the model without being told.

`--optimizer` accepts `adamw` (default), `adam` or `sgd`. Combining `--use_curriculum`
with any optimizer selects the OneCycle schedule.

### Inference

```bash
# Interactive demo
t2-generate

# Generate text from a prompt
@@@@@



fffff

# Load a specific checkpoint
t2-generate --model_path checkpoints/checkpoint_epoch_10.pt --num_heads 8 --prompt "Once upon a time"
```

`generate_text()` returns the continuation only; the prompt is not echoed back.

#### `--num_heads` and older checkpoints

The number of attention heads cannot be recovered from the weights, because the
Q/K/V projections are `d_model x d_model` for any head count. A wrong value loads
without error and produces noise, so a checkpoint that does not record its head
count will not load until you supply one:

```bash
t2-generate --model_path checkpoints/checkpoint_epoch_5.pt --num_heads 8 --prompt "..."
```

Checkpoints written by the current `t2-train` record it and need no flag.
`t2-tune` accepts `--num_heads` too.

### Tuning Generation Parameters

Grid search writes the best-scoring combination to a JSON file, which the
generation script then consumes:

```bash
t2-tune --prompt "The future of AI is"
t2-generate --prompt "The future of AI is" --params outputs/inference_params.json --num_samples 5
```

### Ollama (inference only)

Generation can be served by a local [Ollama](https://ollama.com) model instead of a
trained checkpoint:

```bash
t2-generate --use_ollama --ollama_model llama2 --prompt "Once upon a time"
```

This is an inference backend only. An Ollama model is served over HTTP and exposes
no trainable parameters or gradients, so `t2-train --use_ollama` exits with an
error rather than appearing to train.

### End-to-end Demo

```bash
# Bluesky demo
t2-demo --dataset bluesky --max_samples 10000

# Reddit demo
t2-demo --dataset reddit   # credentials from .env

# Reuse already-collected data
t2-demo --dataset bluesky --skip_data_collection
```

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

The suite is hermetic: no network access, no GPU required except for the tests that
exercise the training loop directly (skipped automatically when CUDA is
unavailable), and no dependency on the large checkpoints in the repository.

## Model Configuration

Default configuration:
- Sequence length: 512 (`--max_length`)
- Model dimension: 256 (`--d_model`)
- Number of attention heads: 8 (`--num_heads`)
- Number of layers: 6 (`--num_layers`)
- Feed-forward dimension: 1024 (`--d_ff`)
- Dropout: 0.1 (`--dropout`)

## License

MIT

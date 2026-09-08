# Transformer2

A PyTorch implementation of a transformer model for natural language generation. This project provides a complete workflow for downloading/collecting training data, training custom transformer models, and running inference.

## Features

- Custom decoder-only transformer with multi-head attention and pre-norm blocks
- Data collection from multiple sources (Reddit, Bluesky)
- Curriculum learning with sequence length progression
- Mixed precision training with gradient accumulation
- Text generation with temperature, top-k, top-p and repetition penalty
- Grid search over generation parameters
- Optional Ollama backend for inference
- Complete training and inference pipeline

## Project Structure

- `coreTransformer.py` - Core transformer architecture
- `dataloader.py` - Dataset classes and data loading utilities
- `tokenizer.py` - Tokenization utilities (also a CLI for building the tokenizer)
- `training.py` - Model training logic
- `inference.py` - Model loading and text generation
- `generate_text.py` - Generate samples using tuned parameters from a JSON file
- `optimize_inference.py` - Grid search for generation parameters
- `ollama_integration.py` - Ollama model/tokenizer wrappers (inference only)
- `config.py` - Credential loading from the environment / `.env`
- `data_collection.py` - Data collection from different sources
- `reddit_postprocess.py` - Strip metadata from Reddit posts and chunk them for training
- `tinystories_postprocess.py` - Normalise and length-bound the TinyStories corpus
- `text_chunking.py` - Corpus-agnostic chunking shared by both postprocessors
- `train_demo.py` - End-to-end collect-then-train demo script
- `tests/` - Test suite (see [Tests](#tests))
- `archive/` - Superseded scripts kept for reference; not part of the workflow

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
   .venv/bin/pip install -r requirements.txt
   ```

   `requirements.txt` covers everything the project imports, including the Reddit
   collector (`praw`) and the optional BLEU scoring in `optimize_inference.py`
   (`nltk`). For the test suite as well, use `requirements-dev.txt` instead.

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
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | `data_collection.py --dataset reddit`, `train_demo.py` |
| `REDDIT_USER_AGENT` | Reddit collection (optional; has a default) |
| `HUGGINGFACE_TOKEN` | `testHFToken.py`, private model downloads |

The Reddit CLI flags (`--client_id`, `--client_secret`) still work but are
discouraged: arguments are visible in shell history and in the process list.

## Usage

All commands below assume the virtual environment, either via `.venv/bin/python`
or after `source .venv/bin/activate`.

### Data Collection

```bash
# TinyStories: ~2.1M synthetic short stories (~250M tokens)
python data_collection.py --dataset tinystories
python tinystories_postprocess.py --max_tokens 320

# Collect Bluesky data
python data_collection.py --dataset bluesky --max_samples 10000

# Collect Reddit data
python data_collection.py --dataset reddit   # credentials from .env
```

`--max_samples` defaults to collecting **everything**. Pass it explicitly to cap a
download; omitting it no longer silently truncates an existing corpus.

Collected data is written to `data/` as a parquet file. Note that re-running
collection **overwrites** the existing file, so a smaller `--max_samples` than a
previous run will shrink the corpus.

#### Reddit postprocessing (required)

The collector writes one row per post, concatenating every comment behind
`Title:`/`Content:`/`Comment:` labels. Those labels are scaffolding the model would
learn to reproduce, and a single thread can run to tens of thousands of tokens, so
almost all of it would be truncated away. Rewrite the corpus before training:

```bash
python reddit_postprocess.py --max_tokens 512   # match your training --max_length
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

The tokenizer is built once from the collected data and cached in `tokenizer/`:

```bash
python tokenizer.py --dataset_type bluesky --vocab_size 50000
```

Training and inference both create it on demand if it is missing, but doing it
explicitly is faster and makes the vocabulary size a deliberate choice. The
tokenizer's vocabulary must match the checkpoint's embedding table — inference
refuses to load a mismatched pair rather than producing garbage.

### Training

```bash
# Standard training
python training.py --dataset bluesky --batch_size 8 --epochs 30

# Curriculum learning
python training.py --dataset reddit --batch_size 8 --epochs 30 --use_curriculum

# Custom architecture
python training.py --dataset bluesky --d_model 512 --num_heads 16 --num_layers 8 --d_ff 2048
```

The best model by validation loss is written to `best_model.pt`, with periodic
snapshots as `checkpoint_epoch_N.pt`. Every checkpoint records the architecture it
was trained with, so inference can rebuild the model without being told.

`--optimizer` accepts `adamw` (default), `adam` or `sgd`. Combining `--use_curriculum`
with any optimizer selects the OneCycle schedule.

### Inference

```bash
# Interactive demo
python inference.py

# Generate text from a prompt
python inference.py --prompt "Once upon a time" --temperature 0.7 --top_k 50

# Load a specific checkpoint
python inference.py --model_path checkpoint_epoch_30.pt --num_heads 8 --prompt "Once upon a time"
```

`generate_text()` returns the continuation only; the prompt is not echoed back.

#### `--num_heads` and older checkpoints

The number of attention heads cannot be recovered from the weights, because the
Q/K/V projections are `d_model x d_model` for any head count. A wrong value loads
without error and produces noise, so a checkpoint that does not record its head
count will not load until you supply one:

```bash
python inference.py --model_path checkpoint_epoch_5.pt --num_heads 8 --prompt "..."
```

Checkpoints written by the current `training.py` record it and need no flag.
`generate_text.py` and `optimize_inference.py` accept `--num_heads` too.

### Tuning Generation Parameters

Grid search writes the best-scoring combination to a JSON file, which the
generation script then consumes:

```bash
python optimize_inference.py --prompt "The future of AI is" --output inference_params.json
python generate_text.py --prompt "The future of AI is" --params inference_params.json --num_samples 5
```

### Ollama (inference only)

Generation can be served by a local [Ollama](https://ollama.com) model instead of a
trained checkpoint:

```bash
python inference.py --use_ollama --ollama_model llama2 --prompt "Once upon a time"
```

This is an inference backend only. An Ollama model is served over HTTP and exposes
no trainable parameters or gradients, so `training.py --use_ollama` exits with an
error rather than appearing to train.

### End-to-end Demo

```bash
# Bluesky demo
python train_demo.py --dataset bluesky --max_samples 10000

# Reddit demo
python train_demo.py --dataset reddit   # credentials from .env

# Reuse already-collected data
python train_demo.py --dataset bluesky --skip_data_collection
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
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

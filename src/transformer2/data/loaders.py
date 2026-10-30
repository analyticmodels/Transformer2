from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from transformer2.tokenization.bpe import load_or_build_tokenizer
from transformer2.paths import data_dir

def build_labels(input_ids, length, max_length):
    """
    Build next-token targets, ignoring everything that is not real text.

    Position i predicts token i+1. The final real token has nothing to predict, and
    padding must not be learned at all, so both are set to the -100 index the loss
    is configured to ignore. Leaving padding in the targets makes the model spend
    most of its gradient on emitting pad tokens.

    Args:
        input_ids (torch.Tensor): Token ids of shape [max_length]
        length (int): Number of real (non-padding) tokens
        max_length (int): Padded sequence length

    Returns:
        torch.Tensor: Labels of shape [max_length]
    """
    labels = torch.full((max_length,), -100, dtype=torch.long)
    if length > 1:
        labels[:length - 1] = input_ids[1:length]
    return labels


class TransformerDataset(Dataset):
    """
    Pre-tokenises a corpus into a single contiguous tensor.

    Storing one tensor per example is not merely untidy: DataLoader workers receive
    the dataset by pickling it, and torch's default `file_descriptor` sharing strategy
    allocates one shared-memory descriptor per tensor. A corpus of ~1.6M examples
    therefore exhausts the process descriptor limit before the first batch is served.
    One [N, max_length] tensor costs a single descriptor instead of N.

    Token ids are held as int32 rather than int64, halving the footprint; they are
    widened back to long in __getitem__ because nn.Embedding requires it.
    """

    def __init__(self, texts, tokenizer_wrapper, max_length=512, chunk_size=50000):
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
        self.pad_token_id = tokenizer_wrapper.get_pad_token_id()

        num_texts = len(texts)

        # Filled as numpy and handed to torch at the end: assigning a Python list into a
        # numpy row avoids allocating a throwaway tensor per text, which dominates the
        # loop once the encoding itself is batched.
        ids_array = np.full((num_texts, max_length), self.pad_token_id, dtype=np.int32)
        lengths = np.zeros(num_texts, dtype=np.int32)

        # Encode in chunks: the Rust tokenizer parallelises across cores within a batch,
        # while chunking keeps peak memory bounded and lets progress be reported.
        with tqdm(total=num_texts, desc="Tokenizing", unit="text") as progress:
            for start in range(0, num_texts, chunk_size):
                chunk = texts[start:start + chunk_size]
                for offset, encoding in enumerate(self.tokenizer.encode_batch(chunk)):
                    ids = encoding['input_ids'][:max_length]
                    if ids:
                        ids_array[start + offset, :len(ids)] = ids
                    lengths[start + offset] = len(ids)
                progress.update(len(chunk))

        self.input_ids = torch.from_numpy(ids_array)
        self.lengths = torch.from_numpy(lengths)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].long()
        length = int(self.lengths[idx])

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:length] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': build_labels(input_ids, length, self.max_length),
        }

class RedditDataset(Dataset):
    """Dataset for Reddit posts and comments."""
    def __init__(self, texts, tokenizer_wrapper, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx])['input_ids'][:self.max_length]
        length = len(ids)

        pad_id = self.tokenizer.get_pad_token_id()
        input_ids = torch.full((self.max_length,), pad_id, dtype=torch.long)
        if length:
            input_ids[:length] = torch.tensor(ids, dtype=torch.long)

        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:length] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': build_labels(input_ids, length, self.max_length),
        }

REDDIT_TEXT_NAME = 'reddit_text.parquet'
REDDIT_RAW_NAME = 'reddit_posts.parquet'


def load_reddit_texts(parquet_path=None):
    """
    Load the Reddit training texts, preferring the postprocessed corpus.

    reddit_postprocess.py strips the `Title:`/`Content:`/`Comment:` scaffolding and
    rewrites the corpus as uniform chunks. Falling back to the raw collection keeps
    older setups working, but those rows carry the scaffolding and can be tens of
    thousands of tokens long, so almost all of each row is truncated away.

    Args:
        parquet_path (str, optional): Explicit file to load. Auto-detected if None.

    Returns:
        list of str: Training texts

    Raises:
        FileNotFoundError: If neither corpus is present
    """
    if parquet_path is None:
        if (data_dir() / REDDIT_TEXT_NAME).exists():
            parquet_path = str(data_dir() / REDDIT_TEXT_NAME)
        elif (data_dir() / REDDIT_RAW_NAME).exists():
            parquet_path = str(data_dir() / REDDIT_RAW_NAME)
        else:
            raise FileNotFoundError(
                f"No Reddit corpus found. Collect and postprocess it first:\n"
                f"    t2-collect --dataset reddit\n"
                f"    t2-postprocess reddit --max_tokens <your --max_length>"
            )

    print(f"Loading Reddit data from {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if 'text' in df.columns:
        texts = df['text'].tolist()
        print(f"Loaded {len(texts):,} postprocessed Reddit chunks")
    elif 'combined_text' in df.columns:
        texts = df['combined_text'].tolist()
        print(f"Loaded {len(texts):,} raw Reddit posts")
        print("  WARNING: this is the unprocessed corpus. Each row still contains the")
        print("  Title:/Content:/Comment: scaffolding and may be far longer than")
        print("  max_length, so most of it will be truncated. Run:")
        print("      t2-postprocess reddit --max_tokens <your --max_length>")
    else:
        raise ValueError(
            f"{parquet_path} has neither a 'text' nor a 'combined_text' column "
            f"(found: {list(df.columns)})"
        )
    return texts


def get_reddit_dataloaders(parquet_path=None, batch_size=32, max_length=512,
                           val_split=0.1, tokenizer_type="byte-level-bpe"):
    """
    Load Reddit data from parquet file and create dataloaders.

    Args:
        parquet_path (str, optional): Parquet to load. Prefers the postprocessed
            corpus at data/reddit_text.parquet when None.
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        val_split (float): Fraction of data to use for validation
        tokenizer_type (str): Type of tokenizer to use

    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    texts = load_reddit_texts(parquet_path)

    # One producer of tokenizers, with provenance recorded beside the files
    tokenizer = load_or_build_tokenizer('reddit', texts)
    
    # Split data into train and validation sets
    num_val = int(len(texts) * val_split)
    num_train = len(texts) - num_val
    
    train_texts = texts[:num_train]
    val_texts = texts[num_train:]
    
    print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
    
    # Pre-tokenise once rather than per epoch. Postprocessed chunks are bounded by
    # max_tokens, so the contiguous tensor stays small.
    train_dataset = TransformerDataset(train_texts, tokenizer, max_length)
    val_dataset = TransformerDataset(val_texts, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, tokenizer

def get_bluesky_dataloaders(parquet_path=None, batch_size=32, max_length=512, 
                           val_split=0.1, tokenizer_type="byte-level-bpe"):
    """
    Load Bluesky data from parquet file and create dataloaders.
    
    Args:
        parquet_path (str): Path to the parquet file containing Bluesky data
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        val_split (float): Fraction of data to use for validation
        tokenizer_type (str): Type of tokenizer to use
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # Load Bluesky data
    parquet_path = str(parquet_path or data_dir() / 'bluesky_posts.parquet')
    print(f"Loading Bluesky data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Get text data
    texts = df['text'].tolist()
    print(f"Loaded {len(texts)} Bluesky posts")
    
    # One producer of tokenizers, with provenance recorded beside the files
    tokenizer = load_or_build_tokenizer('bluesky', texts)
    
    # Split data into train and validation sets
    num_val = int(len(texts) * val_split)
    num_train = len(texts) - num_val
    
    train_texts = texts[:num_train]
    val_texts = texts[num_train:]
    
    print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
    
    # Create datasets
    # Using TransformerDataset instead of creating a new BlueskyDataset class
    # since the data structure is similar
    train_dataset = TransformerDataset(train_texts, tokenizer, max_length)
    val_dataset = TransformerDataset(val_texts, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, tokenizer

TINYSTORIES_TEXT_NAME = 'tinystories_text.parquet'
TINYSTORIES_RAW_NAME = 'tinystories.parquet'


def get_tinystories_dataloaders(parquet_path=None, batch_size=32, max_length=320,
                                val_split=0.1, tokenizer_type="byte-level-bpe"):
    """
    Build dataloaders over the TinyStories corpus.

    Args:
        parquet_path (str, optional): Parquet to load. Prefers the postprocessed
            corpus at data/tinystories_text.parquet when None.
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        val_split (float): Fraction held out for validation
        tokenizer_type (str): Type of tokenizer to use

    Returns:
        tuple: (train_loader, val_loader, tokenizer)

    Raises:
        FileNotFoundError: If no TinyStories corpus is present
    """
    if parquet_path is None:
        if (data_dir() / TINYSTORIES_TEXT_NAME).exists():
            parquet_path = str(data_dir() / TINYSTORIES_TEXT_NAME)
        elif (data_dir() / TINYSTORIES_RAW_NAME).exists():
            parquet_path = str(data_dir() / TINYSTORIES_RAW_NAME)
            print("  WARNING: using the raw download. Run tinystories_postprocess.py to")
            print("  normalise, deduplicate and length-bound it first.")
        else:
            raise FileNotFoundError(
                "No TinyStories corpus found. Download and postprocess it first:\n"
                "    t2-collect --dataset tinystories\n"
                "    t2-postprocess tinystories --max_tokens <your --max_length>"
            )

    print(f"Loading TinyStories from {parquet_path}")
    texts = pd.read_parquet(parquet_path)['text'].tolist()
    print(f"Loaded {len(texts):,} stories")

    tokenizer = load_or_build_tokenizer('tinystories', texts)

    # Shuffle before splitting: an ordered corpus otherwise puts a systematically
    # different slice in validation than in training.
    rng = np.random.default_rng(42)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]

    num_val = int(len(texts) * val_split)
    train_texts, val_texts = texts[num_val:], texts[:num_val]
    print(f"Training on {len(train_texts):,} samples, validating on {len(val_texts):,}")

    train_dataset = TransformerDataset(train_texts, tokenizer, max_length)
    val_dataset = TransformerDataset(val_texts, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, tokenizer


def sample_parquet_texts(parquet_path, max_texts, batch_size=20000, seed=42):
    """
    Read up to max_texts rows of a corpus without materialising all of it.

    pandas.read_parquet holds the Arrow buffers and the Python strings it decodes
    them into at the same time. On the full Gutenberg corpus that is ~44 GB before a
    single token is encoded, and with the token array on top the process is OOM-killed
    on a 96 GB machine. Streaming row batches keeps only the rows actually retained,
    so peak memory follows max_texts rather than the size of the file.

    Rows are sampled across the whole file rather than taken from the front: the
    corpus is ordered by book, so the first N chunks are the first few hundred books
    rather than a cross-section of the corpus.

    Args:
        parquet_path (str): Corpus to read, with a `text` column
        max_texts (int): Rows to keep; 0 or negative keeps all of them
        batch_size (int): Rows decoded per batch
        seed (int): Seed for the sampling draw

    Returns:
        list of str: At most max_texts texts
    """
    reader = pq.ParquetFile(parquet_path)
    total = reader.metadata.num_rows
    keep_all = max_texts is None or max_texts <= 0 or max_texts >= total
    target = total if keep_all else max_texts

    if not keep_all:
        print(f"  sampling {target:,} of {total:,} chunks "
              f"({100 * target / total:.0f}%); pass --max_texts 0 for all")

    rng = np.random.default_rng(seed)
    # Draw slightly above the target rate so a short draw still fills the quota
    probability = 1.0 if keep_all else min(1.0, 1.05 * target / total)

    texts = []
    for batch in reader.iter_batches(batch_size=batch_size, columns=['text']):
        rows = batch.column('text').to_pylist()
        if probability >= 1.0:
            texts.extend(rows)
        else:
            keep = rng.random(len(rows)) < probability
            texts.extend([t for t, k in zip(rows, keep) if k])
        if len(texts) >= target:
            break

    return texts[:target]


GUTENBERG_TEXT_NAME = 'gutenberg_text.parquet'
GUTENBERG_RAW_NAME = 'gutenberg_books.parquet'

# The full corpus is ~5B tokens. TransformerDataset pre-tokenizes every text into
# one contiguous int32 tensor, so loading all of it would want ~20 GB of RAM before
# training starts. Sample down to something that fits and say so.
GUTENBERG_DEFAULT_MAX_TEXTS = 500000


def get_gutenberg_dataloaders(parquet_path=None, batch_size=32, max_length=512,
                              val_split=0.1, tokenizer_type="byte-level-bpe",
                              max_texts=None):
    """
    Build dataloaders over the Project Gutenberg corpus.

    Args:
        parquet_path (str, optional): Parquet to load. Prefers the postprocessed
            corpus at data/gutenberg_text.parquet when None.
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        val_split (float): Fraction held out for validation
        tokenizer_type (str): Type of tokenizer to use
        max_texts (int, optional): Keep at most this many chunks. Defaults to
            GUTENBERG_DEFAULT_MAX_TEXTS; pass 0 or a negative number for all of them.

    Returns:
        tuple: (train_loader, val_loader, tokenizer)

    Raises:
        FileNotFoundError: If no Gutenberg corpus is present
    """
    if parquet_path is None:
        if (data_dir() / GUTENBERG_TEXT_NAME).exists():
            parquet_path = str(data_dir() / GUTENBERG_TEXT_NAME)
        elif (data_dir() / GUTENBERG_RAW_NAME).exists():
            parquet_path = str(data_dir() / GUTENBERG_RAW_NAME)
            print("  WARNING: using the raw download. Run t2-postprocess gutenberg to")
            print("  strip the Gutenberg boilerplate and chunk the books first.")
        else:
            raise FileNotFoundError(
                "No Gutenberg corpus found. Collect and postprocess it first:\n"
                "    t2-collect --dataset gutenberg\n"
                "    t2-postprocess gutenberg --max_tokens <your --max_length>"
            )

    if max_texts is None:
        max_texts = GUTENBERG_DEFAULT_MAX_TEXTS

    print(f"Loading Project Gutenberg from {parquet_path}")
    texts = sample_parquet_texts(parquet_path, max_texts)
    print(f"Loaded {len(texts):,} chunks")

    # Shuffle before splitting: the corpus is ordered by book, so an unshuffled
    # split would hold out whole authors rather than a representative sample.
    rng = np.random.default_rng(42)
    texts = [texts[i] for i in rng.permutation(len(texts))]

    tokenizer = load_or_build_tokenizer('gutenberg', texts)

    num_val = int(len(texts) * val_split)
    train_texts, val_texts = texts[num_val:], texts[:num_val]
    print(f"Training on {len(train_texts):,} samples, validating on {len(val_texts):,}")

    train_dataset = TransformerDataset(train_texts, tokenizer, max_length)
    val_dataset = TransformerDataset(val_texts, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, tokenizer


def get_dataloaders(batch_size=32, max_length=512, dataset_type="bluesky",
                    max_texts=None):
    """
    Get dataloaders for training.

    Args:
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        dataset_type (str): One of 'bluesky', 'reddit', 'tinystories', 'gutenberg'
        max_texts (int, optional): Cap on the number of texts loaded. Only the
            Gutenberg corpus is large enough to need one; ignored elsewhere.

    Returns:
        tuple: (train_loader, val_loader, tokenizer)

    Raises:
        ValueError: If dataset_type is not one of the supported corpora
    """
    if dataset_type == "tinystories":
        return get_tinystories_dataloaders(batch_size=batch_size, max_length=max_length)
    if dataset_type == "reddit":
        return get_reddit_dataloaders(batch_size=batch_size, max_length=max_length)
    if dataset_type == "bluesky":
        return get_bluesky_dataloaders(batch_size=batch_size, max_length=max_length)
    if dataset_type == "gutenberg":
        return get_gutenberg_dataloaders(batch_size=batch_size, max_length=max_length,
                                         max_texts=max_texts)

    raise ValueError(
        f"Unsupported dataset_type {dataset_type!r}. "
        f"Choose one of: bluesky, reddit, tinystories, gutenberg."
    )

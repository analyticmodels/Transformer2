import pathlib
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import hashlib
import json
import os
from abc import ABC, abstractmethod

from transformer2.paths import ensure, tokenizer_dir
# from  dataloader import get_reddit_dataloaders, get_bluesky_dataloaders


SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]
META_NAME = "meta.json"


def vocab_fingerprint(directory):
    """
    Hash a tokenizer's vocabulary.

    Every tokenizer this project trains has exactly 50000 entries, so vocabulary
    size cannot distinguish a Bluesky tokenizer from a Reddit one. The fingerprint
    is what lets a checkpoint prove it is being loaded with the tokenizer it was
    trained on.

    Args:
        directory (str or pathlib.Path): Directory holding vocab.json

    Returns:
        str: Hex sha256 of vocab.json
    """
    return hashlib.sha256((pathlib.Path(directory) / "vocab.json").read_bytes()).hexdigest()


def write_tokenizer_meta(directory, dataset_type, vocab_size, min_frequency, source=None):
    """Record which corpus a tokenizer came from, beside the tokenizer itself."""
    directory = pathlib.Path(directory)
    meta = {
        "dataset": dataset_type,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "source": str(source) if source else None,
        "vocab_sha256": vocab_fingerprint(directory),
    }
    (directory / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def read_tokenizer_meta(directory):
    """Read a tokenizer's provenance, or None if it predates provenance tracking."""
    path = pathlib.Path(directory) / META_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class BaseTokenizerWrapper(ABC):
    @abstractmethod
    def encode(self, text):
        pass
    
    @abstractmethod
    def decode(self, ids):
        pass
    
    @abstractmethod
    def get_vocab_size(self):
        pass
    
    @abstractmethod
    def get_pad_token_id(self):
        pass

    def encode_batch(self, texts):
        """
        Encode many texts at once.

        This default just loops. Wrappers whose backend has native batch support should
        override it: the Rust tokenizers release the GIL and encode a batch across all
        cores, which is several times faster than calling encode() per text.

        Args:
            texts (iterable of str): Texts to encode

        Returns:
            list of dict: One {'input_ids', 'attention_mask'} mapping per text, in order
        """
        return [self.encode(text) for text in texts]

    def get_eos_token_id(self):
        """ID of the end-of-sequence token, or None if the tokenizer has none."""
        return None

    def get_special_token_ids(self):
        """IDs of every special (non-text) token, used to suppress them during sampling."""
        return []

class ByteLevelBPETokenizerWrapper(BaseTokenizerWrapper):
    SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<pad>"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._vocab_size = len(self.tokenizer.get_vocab())
        
    def encode(self, text, max_length=None):
        encoding = self.tokenizer.encode(text)
        input_ids = encoding.ids
        if max_length:
            input_ids = input_ids[:max_length] + [self.get_pad_token_id()] * (max_length - len(input_ids))
        attention_mask = [1 if id != self.get_pad_token_id() else 0 for id in input_ids]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def encode_batch(self, texts):
        pad_token_id = self.get_pad_token_id()
        return [
            {
                'input_ids': encoding.ids,
                'attention_mask': [1 if i != pad_token_id else 0 for i in encoding.ids],
            }
            for encoding in self.tokenizer.encode_batch(list(texts))
        ]

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self._vocab_size

    def get_pad_token_id(self):
        # Resolve <pad> from the vocabulary rather than assuming index 0, which is
        # <s> in this project's special-token order.
        pad_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        return pad_id if pad_id is not None else 0

    def get_eos_token_id(self):
        return self.tokenizer.token_to_id(self.EOS_TOKEN)

    def get_special_token_ids(self):
        ids = [self.tokenizer.token_to_id(token) for token in self.SPECIAL_TOKENS]
        return [token_id for token_id in ids if token_id is not None]

class HuggingFaceTokenizerWrapper(BaseTokenizerWrapper):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def encode(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            return_attention_mask=True
        )

    def encode_batch(self, texts):
        batch = self.tokenizer(
            list(texts),
            truncation=True,
            return_attention_mask=True
        )
        return [
            {'input_ids': ids, 'attention_mask': mask}
            for ids, mask in zip(batch['input_ids'], batch['attention_mask'])
        ]

    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

    def get_special_token_ids(self):
        return list(self.tokenizer.all_special_ids)

def train_byte_level_tokenizer(dataset_type, vocab_size=50000, min_frequency=2):
    """
    Train a Byte-Level BPE tokenizer on the specified dataset.

    Args:
        dataset_type (str): The type of dataset to load (e.g., "wikitext", "bluesky", "reddit").
        vocab_size (int): The size of the vocabulary.
        min_frequency (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        ByteLevelBPETokenizer: The trained tokenizer.
    """
    import pandas as pd
    
    # The corpus dump lives beside the tokenizer it produces
    destination = ensure(tokenizer_dir(dataset_type))
    training_data_path = str(destination / "training_data.txt")
    
    # Load the dataset based on the dataset_type
    if dataset_type == "bluesky":
        # Check if the bluesky data file exists
        bluesky_path = "data/bluesky_posts.parquet"
        if os.path.exists(bluesky_path):
            print(f"Loading Bluesky data from {bluesky_path}")
            df = pd.read_parquet(bluesky_path)
            
            # Extract text data from dataframe
            with open(training_data_path, "w", encoding="utf-8") as f:
                # The text column in the Bluesky dataset is called 'text'
                for text in df['text']:
                    if text and text.strip():  # Skip empty lines
                        f.write(text + "\n")
            
            print(f"Saved {len(df)} Bluesky posts to {training_data_path}")
        else:
            print(f"Bluesky data file not found at {bluesky_path}")
            print("Falling back to wikitext dataset")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            text_field = "text"
            
            # Write the dataset text to the training data file
            with open(training_data_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    text = item[text_field]
                    if text and text.strip():  # Skip empty lines
                        f.write(text + "\n")
    
    elif dataset_type == "reddit":
        # Check if the reddit data file exists
        reddit_path = "data/reddit_posts.parquet"
        if os.path.exists(reddit_path):
            print(f"Loading Reddit data from {reddit_path}")
            df = pd.read_parquet(reddit_path)
            
            # Extract text data from dataframe
            with open(training_data_path, "w", encoding="utf-8") as f:
                # The text column in the Reddit dataset is called 'combined_text'
                for text in df['combined_text']:
                    if text and text.strip():  # Skip empty lines
                        f.write(text + "\n")
            
            print(f"Saved {len(df)} Reddit posts to {training_data_path}")
        else:
            print(f"Reddit data file not found at {reddit_path}")
            print("Falling back to wikitext dataset")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            text_field = "text"
            
            # Write the dataset text to the training data file
            with open(training_data_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    text = item[text_field]
                    if text and text.strip():  # Skip empty lines
                        f.write(text + "\n")
    
    elif dataset_type == "tinystories":
        # Prefer the postprocessed corpus; fall back to the raw download
        for candidate in ("data/tinystories_text.parquet", "data/tinystories.parquet"):
            if os.path.exists(candidate):
                print(f"Loading TinyStories data from {candidate}")
                df = pd.read_parquet(candidate)
                with open(training_data_path, "w", encoding="utf-8") as f:
                    for text in df['text']:
                        if text and text.strip():
                            f.write(text.replace("\n", " ") + "\n")
                print(f"Saved {len(df)} stories to {training_data_path}")
                break
        else:
            raise FileNotFoundError(
                "No TinyStories corpus found. Download it first:\n"
                "    python data_collection.py --dataset tinystories"
            )

    elif dataset_type == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_field = "text"
        
        # Write the dataset text to the training data file
        with open(training_data_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item[text_field]
                if text and text.strip():  # Skip empty lines
                    f.write(text + "\n")
    
    elif dataset_type == "imdb":
        dataset = load_dataset("imdb", split="train")
        text_field = "text"
        
        # Write the dataset text to the training data file
        with open(training_data_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item[text_field]
                if text and text.strip():  # Skip empty lines
                    f.write(text + "\n")
    
    else:
        # Use a default dataset if the requested one isn't available
        print(f"Dataset type {dataset_type} not directly supported. Using wikitext as fallback.")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_field = "text"
        
        # Write the dataset text to the training data file
        with open(training_data_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item[text_field]
                if text and text.strip():  # Skip empty lines
                    f.write(text + "\n")

    # Initialize and train the Byte-Level BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[training_data_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )

    # Save the tokenizer files for future use, beside a record of which corpus
    # they came from -- vocabulary size alone cannot identify a tokenizer.
    destination = ensure(tokenizer_dir(dataset_type))
    tokenizer.save_model(str(destination))
    write_tokenizer_meta(destination, dataset_type, vocab_size, min_frequency,
                         source=training_data_path)
    print(f"Saved the {dataset_type} tokenizer to {destination}")
    return tokenizer

def create_tokenizer(dataset_type, tokenizer_type="byte-level-bpe", *,
                     directory=None, **kwargs):
    """
    Load the tokenizer belonging to a corpus.

    `dataset_type` is a required positional argument. It used to default to
    "wikitext" and be ignored for path resolution, so five of six call sites never
    passed it and every corpus silently shared one vocabulary.

    Args:
        dataset_type (str): Corpus the tokenizer was built from
        tokenizer_type (str): "byte-level-bpe" (default) or "gpt2"
        directory (str or pathlib.Path, optional): Override the resolved location
        **kwargs: Ignored; accepted so existing callers keep working

    Returns:
        BaseTokenizerWrapper: The wrapped tokenizer

    Raises:
        FileNotFoundError: If no tokenizer exists for this corpus. Falling back to
            GPT-2 here would silently swap in a 50257-token vocabulary and train or
            generate against the wrong one.
        ValueError: If tokenizer_type is unknown
    """
    if tokenizer_type == "gpt2":
        return HuggingFaceTokenizerWrapper(GPT2Tokenizer.from_pretrained('gpt2'))

    if tokenizer_type != "byte-level-bpe":
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    directory = pathlib.Path(directory) if directory else tokenizer_dir(dataset_type)
    vocab, merges = directory / "vocab.json", directory / "merges.txt"

    if not (vocab.is_file() and merges.is_file()):
        raise FileNotFoundError(
            f"No tokenizer for '{dataset_type}' at {directory}. Build one with:\n"
            f"    python tokenizer.py --dataset_type {dataset_type}"
        )

    tokenizer = ByteLevelBPETokenizer.from_file(str(vocab), str(merges))
    meta = read_tokenizer_meta(directory)
    if meta is None:
        print(f"Using tokenizer at {directory} (no provenance recorded).")
    else:
        print(f"Using the '{meta['dataset']}' tokenizer at {directory}.")
    return ByteLevelBPETokenizerWrapper(tokenizer)



def load_or_build_tokenizer(dataset_type, texts=None, vocab_size=50000, min_frequency=2):
    """
    Return the tokenizer for a corpus, training one if it does not exist yet.

    This is the single producer of tokenizers. dataloader.py previously carried its
    own inline training code that wrote the same directory with different settings
    and no provenance record, so two producers disagreed about what the files meant.

    Args:
        dataset_type (str): Corpus name
        texts (list of str, optional): Corpus text, used when one has to be trained.
            Falls back to rebuilding from the dataset on disk when omitted.
        vocab_size (int): Vocabulary size when training
        min_frequency (int): Minimum token frequency when training

    Returns:
        BaseTokenizerWrapper
    """
    directory = tokenizer_dir(dataset_type)
    if (directory / "vocab.json").is_file() and (directory / "merges.txt").is_file():
        return create_tokenizer(dataset_type)

    print(f"No tokenizer for '{dataset_type}' yet; training one...")
    if texts is None:
        train_byte_level_tokenizer(dataset_type, vocab_size=vocab_size,
                                   min_frequency=min_frequency)
        return create_tokenizer(dataset_type)

    raw = ByteLevelBPETokenizer()
    raw.train_from_iterator(
        (texts[i:i + 1000] for i in range(0, len(texts), 1000)),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    destination = ensure(directory)
    raw.save_model(str(destination))
    write_tokenizer_meta(destination, dataset_type, vocab_size, min_frequency,
                         source=f"{dataset_type} dataloader corpus")
    print(f"Saved the {dataset_type} tokenizer to {destination}")
    return create_tokenizer(dataset_type)


def main():
    """Build (or reuse) the byte-level BPE tokenizer for a dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare tokenizer for transformer model")
    parser.add_argument('--dataset_type', type=str, default='bluesky',
                        choices=['bluesky', 'reddit', 'wikitext', 'tinystories'],
                        help='Dataset to build the tokenizer from')
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='Vocabulary size for the tokenizer')
    parser.add_argument('--min_frequency', type=int, default=2,
                        help='Minimum frequency for a token to be included in vocabulary')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even if a cached tokenizer is present')

    args = parser.parse_args()

    print(f"Preparing tokenizer with {args.dataset_type} dataset...")

    # Rebuild when asked, or when no tokenizer is cached. The previous guard tested
    # for tokenizer/training_data.txt, which exists as soon as ANY corpus has been
    # processed -- so --dataset_type was silently ignored and the cached vocabulary
    # from a different corpus was loaded instead.
    directory = tokenizer_dir(args.dataset_type)
    cached = (directory / "vocab.json").is_file() and (directory / "merges.txt").is_file()
    if args.force or not cached:
        print(f"Building tokenizer from the {args.dataset_type} corpus...")
        train_byte_level_tokenizer(
            args.dataset_type,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    else:
        print(f"Reusing the cached {args.dataset_type} tokenizer at {directory}.")
        print("  Pass --force to rebuild it.")

    tokenizer = create_tokenizer(args.dataset_type)

    encoded = tokenizer.encode("Hello, how are you?")
    print(f"Tokenizer test - Encoded: {encoded}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("\nTokenizer preparation complete.")
    return tokenizer


if __name__ == "__main__":
    main()

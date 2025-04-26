from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import os
from abc import ABC, abstractmethod
# from  dataloader import get_reddit_dataloaders, get_bluesky_dataloaders

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

class ByteLevelBPETokenizerWrapper(BaseTokenizerWrapper):
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
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def get_pad_token_id(self):
        return 0  # Default pad token ID

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
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        return len(self.tokenizer)
    
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

def train_byte_level_tokenizer(dataset_type, vocab_size=50000, min_frequency=2):
    """
    Train a Byte-Level BPE tokenizer on the specified dataset.

    Args:
        dataset_type (str): The type of dataset to load (e.g., "wikitext", "bluesky").
        vocab_size (int): The size of the vocabulary.
        min_frequency (int): The minimum frequency for a token to be included in the vocabulary.

    Returns:
        ByteLevelBPETokenizer: The trained tokenizer.
    """
    # Load the dataset based on the dataset_type
    if dataset_type == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_field = "text"
    elif dataset_type == "imdb":
        dataset = load_dataset("imdb", split="train")
        text_field = "text"
    else:
        # Use a default dataset if the requested one isn't available
        print(f"Dataset type {dataset_type} not directly supported. Using wikitext as fallback.")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_field = "text"

    # Write the dataset text to a temporary file for tokenizer training
    os.makedirs("tokenizer", exist_ok=True)
    training_data_path = "tokenizer/training_data.txt"
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

    # Save the tokenizer files for future use
    tokenizer.save_model("tokenizer")
    return tokenizer

def create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type="wikitext", **kwargs):
    """
    Factory function to create and wrap tokenizers.

    Args:
        tokenizer_type (str): The type of tokenizer to create ("byte-level-bpe" or "gpt2").
        dataset_type (str): The dataset type to use for training the tokenizer (if applicable).

    Returns:
        BaseTokenizerWrapper: A wrapped tokenizer instance.
    """
    # Check if we have pre-trained GPT2 tokenizer files
    if tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return HuggingFaceTokenizerWrapper(tokenizer)
    
    # For byte-level BPE tokenizer
    elif tokenizer_type == "byte-level-bpe":
        # Check for existing tokenizer files
        if os.path.exists("tokenizer/vocab.json") and os.path.exists("tokenizer/merges.txt"):
            try:
                # Try to load the existing tokenizer
                tokenizer = ByteLevelBPETokenizer.from_file(
                    "tokenizer/vocab.json",
                    "tokenizer/merges.txt"
                )
                print("Using existing tokenizer files.")
                return ByteLevelBPETokenizerWrapper(tokenizer)
            except Exception as e:
                print(f"Error loading existing tokenizer: {e}")
                print("Will create a new tokenizer.")
        
        # If no tokenizer files or loading failed, check for pre-existing training data
        if os.path.exists("tokenizer/training_data.txt"):
            print("Using existing training data file for tokenizer training.")
            # Initialize and train tokenizer
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train(
                files=["tokenizer/training_data.txt"],
                vocab_size=kwargs.get('vocab_size', 50000),
                min_frequency=kwargs.get('min_frequency', 2),
                special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
            )
            tokenizer.save_model("tokenizer")
            return ByteLevelBPETokenizerWrapper(tokenizer)
        
        # If no training data, we'd have to generate it
        print("No existing tokenizer or training data found.")
        print("For faster tokenizer initialization, please run 'python tokenizer.py' first.")
        print("Using GPT2 tokenizer as fallback.")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return HuggingFaceTokenizerWrapper(tokenizer)
    
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# Only execute this code if the file is run directly (not imported)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare tokenizer for transformer model")
    parser.add_argument('--dataset_type', type=str, default='wikitext',
                       help='Dataset to use for tokenizer training (wikitext, imdb)')
    parser.add_argument('--vocab_size', type=int, default=50000,
                       help='Vocabulary size for the tokenizer')
    
    args = parser.parse_args()
    
    print(f"Preparing tokenizer with {args.dataset_type} dataset...")
    
    # Create tokenizer training data if needed
    if not os.path.exists("tokenizer/training_data.txt"):
        print("Creating tokenizer training data...")
        # Load and prepare dataset for tokenizer training
        train_byte_level_tokenizer(args.dataset_type, vocab_size=args.vocab_size)
    
    # Now create the tokenizer
    tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type=args.dataset_type)
    
    # Test the tokenizer
    prompt = "Hello, how are you?"
    encoded = tokenizer.encode(prompt)
    print(f"Tokenizer test - Encoded: {encoded}")
    
    print("\nTokenizer preparation complete.")
    print("You can now use inference.py for text generation.")
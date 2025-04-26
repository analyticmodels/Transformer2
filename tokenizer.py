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
    # # Load the dataset based on the dataset_type
    # if dataset_type == "reddit":
    #     dataset = get_reddit_dataloaders(batch_size=32, max_length=512)
    # elif dataset_type == "bluesky":
    #     dataset = get_bluesky_dataloaders(batch_size=32, max_length=512)
    #     print (f"dataset obj: {dataset}")

    # Write the dataset text to a temporary file for tokenizer training
    os.makedirs("tokenizer", exist_ok=True)
    training_data_path = "tokenizer/training_data.txt"
    with open(training_data_path, "w", encoding="utf-8") as f:
        for text in dataset['train']['text']:
            if text.strip():  # Skip empty lines
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
    if tokenizer_type == "byte-level-bpe":
        if os.path.exists("tokenizer/vocab.json"):
            tokenizer = ByteLevelBPETokenizer.from_file(
                "tokenizer/vocab.json",
                "tokenizer/merges.txt"
            )
        else:
            tokenizer = train_byte_level_tokenizer(dataset_type, **kwargs)
        return ByteLevelBPETokenizerWrapper(tokenizer)
    
    elif tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return HuggingFaceTokenizerWrapper(tokenizer)
    
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# Example usage
tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type="bluesky")
prompt = "Hello, how are you?"
encoded = tokenizer.encode(prompt)
print(f"Encoded input: {encoded}")
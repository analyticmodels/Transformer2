from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import os
from abc import ABC, abstractmethod

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
        
    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        return {
            'input_ids': encoding.ids,
            'attention_mask': [1] * len(encoding.ids)
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

def create_tokenizer(tokenizer_type="byte-level-bpe", **kwargs):
    """Factory function to create and wrap tokenizers"""
    if tokenizer_type == "byte-level-bpe":
        if os.path.exists("tokenizer/vocab.json"):
            tokenizer = ByteLevelBPETokenizer.from_file(
                "tokenizer/vocab.json",
                "tokenizer/merges.txt"
            )
        else:
            tokenizer = train_byte_level_tokenizer(**kwargs)
        return ByteLevelBPETokenizerWrapper(tokenizer)
    
    elif tokenizer_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return HuggingFaceTokenizerWrapper(tokenizer)
    
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def train_byte_level_tokenizer(vocab_size=50000, min_frequency=2):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    with open("tokenizer_training_data.txt", "w", encoding="utf-8") as f:
        for text in dataset['train']['text']:
            if text.strip():
                f.write(text + "\n")
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=["tokenizer_training_data.txt"],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )
    
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")
    os.remove("tokenizer_training_data.txt")
    
    return tokenizer
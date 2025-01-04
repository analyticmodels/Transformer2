from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
import numpy as np
import os
from tokenizers import ByteLevelBPETokenizer
from tokenizer import *

class TransformerDataset(Dataset):
    def __init__(self, texts, tokenizer_wrapper, max_length=512):
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        pad_token_id = self.tokenizer.get_pad_token_id()
        
        for text in texts:
            # Encode text
            encoding = self.tokenizer.encode(text)
            
            # Convert to tensors and handle padding
            input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
            
            # Pad or truncate
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
                padding_length = max_length - len(input_ids)
                pad_id_tensor = torch.full((padding_length,), pad_token_id, dtype=torch.long)
                zero_tensor = torch.zeros(padding_length, dtype=torch.long)
                
                input_ids = torch.cat([input_ids, pad_id_tensor])
                attention_mask = torch.cat([attention_mask, zero_tensor])
            
            # Create labels
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100
            
            # Ensure all tensors are long type
            self.input_ids.append(input_ids.long())
            self.attention_masks.append(attention_mask.long())
            self.labels.append(labels.long())
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def get_dataloaders(batch_size=32, max_length=512, tokenizer_type="byte-level-bpe"):
    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_type)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Process datasets
    train_texts = [text for text in dataset['train']['text'] if text.strip()]
    val_texts = [text for text in dataset['validation']['text'] if text.strip()]
    
    # Create datasets
    train_dataset = TransformerDataset(train_texts, tokenizer, max_length)
    val_dataset = TransformerDataset(val_texts, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    # Test with different tokenizer types
    for tokenizer_type in ["byte-level-bpe", "gpt2"]:
        print(f"\nTesting with {tokenizer_type} tokenizer...")
        train_loader, val_loader, tokenizer = get_dataloaders(
            batch_size=8,
            tokenizer_type=tokenizer_type
        )
        
        print("Checking first batch...")
        for batch in train_loader:
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Vocab size: {tokenizer.get_vocab_size()}")
            break
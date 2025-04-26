from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class TransformerDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for text in texts:
            # Tokenize and prepare input
            encodings = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Get input_ids and ensure they're within vocab size
            input_ids = encodings['input_ids'].squeeze(0)  # Remove batch dimension
            input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)
            
            # Create labels (shifted input_ids for language modeling)
            labels = input_ids.clone()
            # Shift the labels to predict next token
            labels[:-1] = input_ids[1:]
            labels[-1] = -100  # Ignore last token prediction
            
            # Store tensors
            self.input_ids.append(input_ids)
            self.attention_masks.append(encodings['attention_mask'].squeeze(0))
            self.labels.append(labels)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def get_dataloaders(batch_size=32, max_length=512):
    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Initialize tokenizer and set padding token
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
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

# Test the dataloader
if __name__ == "__main__":
    print("Initializing dataloaders...")
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=8)
    
    print("\nChecking first batch...")
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs max value: {input_ids.max().item()}")
        print(f"Input IDs min value: {input_ids.min().item()}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        break
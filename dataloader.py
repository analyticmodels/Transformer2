from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
import numpy as np
import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from tokenizer import create_tokenizer

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

class RedditDataset(Dataset):
    """Dataset for Reddit posts and comments."""
    def __init__(self, texts, tokenizer_wrapper, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)
        
        # Get token IDs and create attention mask
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            pad_id = self.tokenizer.get_pad_token_id()
            
            input_ids = torch.cat([
                input_ids, 
                torch.full((padding_length,), pad_id, dtype=torch.long)
            ])
            
            attention_mask = torch.cat([
                attention_mask, 
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        # Create shifted labels for language modeling
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore loss for the last token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def get_reddit_dataloaders(parquet_path='data/reddit_posts.parquet', batch_size=32, max_length=512, 
                           val_split=0.1, tokenizer_type="byte-level-bpe"):
    """
    Load Reddit data from parquet file and create dataloaders.
    
    Args:
        parquet_path (str): Path to the parquet file containing Reddit data
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        val_split (float): Fraction of data to use for validation
        tokenizer_type (str): Type of tokenizer to use
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    # Load Reddit data
    print(f"Loading Reddit data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Get text data
    texts = df['combined_text'].tolist()
    print(f"Loaded {len(texts)} Reddit posts and comments")
    
    # Create or load tokenizer
    if os.path.exists("tokenizer/vocab.json"):
        print("Loading existing tokenizer")
        tokenizer = create_tokenizer(tokenizer_type)
    else:
        print("Training new tokenizer on Reddit data")
        # Initialize tokenizer
        raw_tokenizer = ByteLevelBPETokenizer()
        
        # Create vocabulary from Reddit data
        def batch_iterator():
            for i in range(0, len(texts), 1000):
                yield texts[i:i + 1000]
        
        # Train tokenizer on Reddit data
        raw_tokenizer.train_from_iterator(
            batch_iterator(),
            vocab_size=50000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
        )
        
        # Save tokenizer
        os.makedirs("tokenizer", exist_ok=True)
        raw_tokenizer.save_model("tokenizer")
        
        # Create wrapped tokenizer
        tokenizer = create_tokenizer(tokenizer_type)
    
    # Split data into train and validation sets
    num_val = int(len(texts) * val_split)
    num_train = len(texts) - num_val
    
    train_texts = texts[:num_train]
    val_texts = texts[num_train:]
    
    print(f"Training on {len(train_texts)} samples, validating on {len(val_texts)} samples")
    
    # Create datasets
    train_dataset = RedditDataset(train_texts, tokenizer, max_length)
    val_dataset = RedditDataset(val_texts, tokenizer, max_length)
    
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

def get_bluesky_dataloaders(parquet_path='data/bluesky_posts.parquet', batch_size=32, max_length=512, 
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
    print(f"Loading Bluesky data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Get text data
    texts = df['text'].tolist()
    print(f"Loaded {len(texts)} Bluesky posts")
    
    # Create or load tokenizer
    if os.path.exists("tokenizer/vocab.json"):
        print("Loading existing tokenizer")
        tokenizer = create_tokenizer(tokenizer_type)
    else:
        print("Training new tokenizer on Bluesky data")
        # Initialize tokenizer
        raw_tokenizer = ByteLevelBPETokenizer()
        
        # Create vocabulary from Bluesky data
        def batch_iterator():
            for i in range(0, len(texts), 1000):
                yield texts[i:i + 1000]
        
        # Train tokenizer on Bluesky data
        raw_tokenizer.train_from_iterator(
            batch_iterator(),
            vocab_size=50000,
            min_frequency=2,
            special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
        )
        
        # Save tokenizer
        os.makedirs("tokenizer", exist_ok=True)
        raw_tokenizer.save_model("tokenizer")
        
        # Create wrapped tokenizer
        tokenizer = create_tokenizer(tokenizer_type)
    
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

def get_dataloaders(batch_size=32, max_length=512, dataset_type="imdb"):
    """
    Get dataloaders for training.
    
    Args:
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        dataset_type (str): Type of dataset to use ('imdb', 'reddit', or 'bluesky')
        
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    """
    if dataset_type == "reddit":
        return get_reddit_dataloaders(batch_size=batch_size, max_length=max_length)
    elif dataset_type == "bluesky":
        return get_bluesky_dataloaders(batch_size=batch_size, max_length=max_length)
    
    #
    dataset = load_dataset("imdb")
    
    # Initialize tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Create vocabulary from IMDB data
    def batch_iterator():
        for i in range(0, len(dataset['train']), 1000):
            yield dataset['train'][i:i + 1000]['text']
    
    # Train tokenizer on IMDB data
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=50000,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"]
    )
    
    # Save tokenizer
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")
    
    class IMDBDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.texts = data['text']
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer.encode(text)
            
            # Ensure proper padding and truncation
            input_ids = torch.tensor(encoding.ids[:self.max_length], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            # Pad if necessary
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            
            # Create shifted labels for language modeling
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
    # Create datasets
    train_dataset = IMDBDataset(dataset['train'], tokenizer, max_length)
    val_dataset = IMDBDataset(dataset['test'], tokenizer, max_length)
    
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

"""
Train a transformer model on Reddit data for 30 epochs.
"""

import os
import torch
import argparse
from tqdm import tqdm

from coreTransformer import SmallTransformer
from dataloader import get_dataloaders
from curriculumTraining import train_model

def main():
    parser = argparse.ArgumentParser(description='Train transformer model on Reddit data')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    
    args = parser.parse_args()
    
    print("\nTRAINING TRANSFORMER MODEL ON REDDIT DATA")
    
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load data with the reddit dataset type
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.batch_size,
        dataset_type="reddit"
    )
    
    # Create model
    model = SmallTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Train model for specified number of epochs
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=3,
        accumulation_steps=8
    )
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: best_model.pt")

if __name__ == "__main__":
    main()

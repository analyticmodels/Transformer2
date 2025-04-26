"""
Bluesky Training Demo

This script demonstrates the complete workflow for:
1. Collecting Bluesky data using getBlueskyTraining.py
2. Training the transformer model on the collected data using trainingSetup.py

Usage:
    python blueskyTrainingDemo.py
"""

import argparse
import os
import subprocess
import sys
import time

def check_data_exists():
    """Check if the Bluesky data file exists."""
    return os.path.exists('data/bluesky_posts.parquet')

def collect_bluesky_data(args):
    """Collect Bluesky data using getBlueskyTraining.py."""
    print("\n" + "="*80)
    print("STEP 1: Collecting Bluesky Data")
    print("="*80)
    
    # Build command for data collection
    cmd = [
        'python', 'getBlueskyTraining.py'
    ]
    
    # Add optional arguments
    cmd.extend(['--max_samples', str(args.max_samples)])
    cmd.extend(['--output_path', 'data/bluesky_posts.parquet'])
    
    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print("Error collecting Bluesky data.")
        sys.exit(1)
    
    print("\nBluesky data collection completed successfully.")
    
    # Wait a moment to ensure file is fully written
    time.sleep(1)
    
    # Verify data file exists
    if not check_data_exists():
        print("Error: Bluesky data file not found at data/bluesky_posts.parquet")
        sys.exit(1)
    
    print(f"Data file created at: data/bluesky_posts.parquet")

def train_model(args):
    """Train the transformer model on the collected Bluesky data."""
    print("\n" + "="*80)
    print("STEP 2: Training Transformer Model on Bluesky Data")
    print("="*80)
    
    # Build command for training
    cmd = [
        'python', 'trainingSetup.py',
        '--dataset', 'bluesky',
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--patience', str(args.patience),
        '--accumulation_steps', str(args.accumulation_steps),
        '--max_length', str(args.max_length),
        '--d_model', str(args.d_model),
        '--num_heads', str(args.num_heads),
        '--num_layers', str(args.num_layers),
        '--d_ff', str(args.d_ff),
        '--dropout', str(args.dropout)
    ]
    
    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print("Error training model.")
        sys.exit(1)
    
    print("\nModel training completed successfully.")
    print(f"Best model saved to: best_model.pt")

def main():
    parser = argparse.ArgumentParser(description='Bluesky Training Demo')
    
    # Data collection parameters
    parser.add_argument('--max_samples', type=int, default=10000, 
                        help='Maximum number of posts to fetch from the Bluesky dataset')
    
    # Model training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Skip data collection if data already exists
    parser.add_argument('--skip_data_collection', action='store_true',
                        help='Skip data collection if data already exists')
    
    args = parser.parse_args()
    
    print("\nBluesky Training Demo")
    print("This script demonstrates the complete workflow for collecting Bluesky data and training a transformer model.")
    
    # Check if data exists and if we should skip collection
    if args.skip_data_collection and check_data_exists():
        print("\nSkipping data collection as requested (--skip_data_collection) and data file exists.")
    else:
        collect_bluesky_data(args)
    
    # Train the model
    train_model(args)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nYou can now use the trained model for inference.")
    print("The best model is saved at: best_model.pt")

if __name__ == "__main__":
    main()

"""
Reddit Training Demo

This script demonstrates the complete workflow for:
1. Collecting Reddit data using getRedditTraining.py
2. Training the transformer model on the collected data using trainingSetup.py

Usage:
    python redditTrainingDemo.py --client_id YOUR_REDDIT_CLIENT_ID --client_secret YOUR_REDDIT_CLIENT_SECRET
"""

import argparse
import os
import subprocess
import sys
import time

def check_data_exists():
    """Check if the Reddit data file exists."""
    return os.path.exists('data/reddit_posts.parquet')

def collect_reddit_data(args):
    """Collect Reddit data using getRedditTraining.py."""
    print("\n" + "="*80)
    print("STEP 1: Collecting Reddit Data")
    print("="*80)
    
    # Build command for data collection
    cmd = [
        'python', 'getRedditTraining.py',
        '--client_id', '***REMOVED***',
        '--client_secret', '***REMOVED***',
        '--user_agent', 'python:llm-training-data:v1.0 (by /u/analyticmodels)'
    ]
    
    # Add optional arguments if provided
    if args.subreddits:
        cmd.extend(['--subreddits'] + args.subreddits)
    
    if args.limit:
        cmd.extend(['--limit', str(args.limit)])
    
    if args.post_types:
        cmd.extend(['--post_types'] + args.post_types)
    
    # Execute the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        print("Error collecting Reddit data.")
        sys.exit(1)
    
    print("\nReddit data collection completed successfully.")
    
    # Wait a moment to ensure file is fully written
    time.sleep(1)
    
    # Verify data file exists
    if not check_data_exists():
        print("Error: Reddit data file not found at data/reddit_posts.parquet")
        sys.exit(1)
    
    print(f"Data file created at: data/reddit_posts.parquet")

def train_model(args):
    """Train the transformer model on the collected Reddit data."""
    print("\n" + "="*80)
    print("STEP 2: Training Transformer Model on Reddit Data")
    print("="*80)
    
    # Build command for training
    cmd = [
        'python', 'trainingSetup.py',
        '--dataset', 'reddit',
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
    parser = argparse.ArgumentParser(description='Reddit Training Demo')
    
    # Reddit API credentials
    parser.add_argument('--client_id', help='Reddit API client ID')
    parser.add_argument('--client_secret', help='Reddit API client secret')
    parser.add_argument('--user_agent', default='python:llm-training-data:v1.0 (by /u/your_username)', 
                        help='User agent string for Reddit API')
    
    # Data collection parameters
    parser.add_argument('--subreddits', nargs='+', 
                        default=['AskReddit', 'explainlikeimfive', 'todayilearned', 'science'],
                        help='List of subreddits to fetch posts from')
    parser.add_argument('--limit', type=int, default=500, 
                        help='Maximum number of posts to fetch per subreddit')
    parser.add_argument('--post_types', nargs='+', default=['hot', 'top'],
                        help='Types of posts to fetch (hot, new, top, rising)')
    
    # Model training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=3,w
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
    
    print("\nReddit Training Demo")
    print("This script demonstrates the complete workflow for collecting Reddit data and training a transformer model.")
    
    # Check if data exists and if we should skip collection
    if args.skip_data_collection and check_data_exists():
        print("\nSkipping data collection as requested (--skip_data_collection) and data file exists.")
    else:
        collect_reddit_data(args)
    
    # Train the model
    train_model(args)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nYou can now use the trained model for inference.")
    print("The best model is saved at: best_model.pt")

if __name__ == "__main__":
    main()

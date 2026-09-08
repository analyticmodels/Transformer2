"""
Transformer Training Demo

This script demonstrates the complete workflow for:
1. Collecting data from chosen source (Reddit or Bluesky)
2. Training the transformer model on the collected data

Usage:
    # For Bluesky training:
    python train_demo.py --dataset bluesky
    
    # For Reddit training (credentials come from .env; see .env.example):
    python train_demo.py --dataset reddit
"""

import argparse
import os
import sys
import time

from config import reddit_credentials
from paths import data_dir

def check_data_exists(dataset_type):
    """Check if the data file exists for the specified dataset type."""
    if dataset_type == 'bluesky':
        return (data_dir() / 'bluesky_posts.parquet').exists()
    elif dataset_type == 'reddit':
        return (data_dir() / 'reddit_posts.parquet').exists()
    else:
        return False

def collect_data(args):
    """Collect data using data_collection.py."""
    print("\n" + "="*80)
    print(f"STEP 1: Collecting {args.dataset.capitalize()} Data")
    print("="*80)
    
    # Import the data collection function
    from data_collection import collect_data
    
    # Collect data based on dataset type
    if args.dataset == 'bluesky':
        print(f"Collecting Bluesky data (max_samples={args.max_samples})...")
        df = collect_data(
            dataset_type='bluesky',
            max_samples=args.max_samples
        )
    elif args.dataset == 'reddit':
        print(f"Collecting Reddit data from {', '.join(['r/' + sub for sub in args.subreddits])}...")
        df = collect_data(
            dataset_type='reddit',
            client_id=args.client_id,
            client_secret=args.client_secret,
            user_agent=args.user_agent,
            subreddits=args.subreddits,
            limit=args.limit,
            post_types=args.post_types
        )
    else:
        print(f"Unsupported dataset type: {args.dataset}")
        sys.exit(1)
    
    if df is None or len(df) == 0:
        print(f"Error collecting {args.dataset} data.")
        sys.exit(1)
    
    print(f"\n{args.dataset.capitalize()} data collection completed successfully.")
    print(f"Collected {len(df)} items.")
    
    # Wait a moment to ensure file is fully written
    time.sleep(1)
    
    # Verify data file exists
    if not check_data_exists(args.dataset):
        print(f"Error: {args.dataset} data file not found")
        sys.exit(1)

def train_model(args):
    """Train the transformer model on the collected data."""
    print("\n" + "="*80)
    print(f"STEP 2: Training Transformer Model on {args.dataset.capitalize()} Data")
    print("="*80)
    print(f"Optimizer: {args.optimizer.upper()}")
    
    # Import the training function
    from training import train_model, main as training_main
    
    # Set command-line arguments for training
    sys.argv = [
        'training.py',
        f'--dataset={args.dataset}',
        f'--batch_size={args.batch_size}',
        f'--epochs={args.epochs}',
        f'--patience={args.patience}',
        f'--accumulation_steps={args.accumulation_steps}',
        f'--max_length={args.max_length}',
        f'--d_model={args.d_model}',
        f'--num_heads={args.num_heads}',
        f'--num_layers={args.num_layers}',
        f'--d_ff={args.d_ff}',
        f'--dropout={args.dropout}',
        f'--optimizer={args.optimizer}'
    ]
    
    if args.use_curriculum:
        sys.argv.append('--use_curriculum')
    
    # Run the training
    try:
        print(f"Starting model training with {args.epochs} epochs...")
        training_main()
        print("\nModel training completed successfully.")
        print(f"Best model saved to: best_model.pt")
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Transformer Training Demo')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True, choices=['reddit', 'bluesky'],
                       help='Dataset to use for training')
    
    # Reddit API credentials
    parser.add_argument('--client_id', help='Reddit API client ID. Prefer REDDIT_CLIENT_ID '
                                            'in .env; command-line secrets leak into shell history')
    parser.add_argument('--client_secret', help='Reddit API client secret. Prefer '
                                                'REDDIT_CLIENT_SECRET in .env')
    parser.add_argument('--user_agent', default='python:transformer-training:v1.0', 
                        help='User agent string for Reddit API')
    
    # Reddit data collection parameters
    parser.add_argument('--subreddits', nargs='+', 
                        default=['AskReddit', 'explainlikeimfive', 'todayilearned', 'science'],
                        help='List of subreddits to fetch posts from (for Reddit)')
    parser.add_argument('--limit', type=int, default=500, 
                        help='Maximum number of posts to fetch per subreddit (for Reddit)')
    parser.add_argument('--post_types', nargs='+', default=['hot', 'top'],
                        help='Types of posts to fetch: hot, new, top, rising (for Reddit)')
    
    # Bluesky data collection parameters
    parser.add_argument('--max_samples', type=int, default=10000, 
                        help='Maximum number of posts to fetch from Bluesky dataset')
    
    # Model training parameters -
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
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer to use (adamw, adam, or sgd)')
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Use curriculum learning (gradually increase sequence length)')
    
    # Skip data collection if data already exists
    parser.add_argument('--skip_data_collection', action='store_true',
                        help='Skip data collection if data already exists')
    
    args = parser.parse_args()
    
    # Validate arguments. Credentials may come from .env rather than the command line,
    # so resolve them before deciding whether collection is possible.
    if args.dataset == 'reddit' and not args.skip_data_collection:
        client_id, client_secret, _ = reddit_credentials(
            client_id=args.client_id, client_secret=args.client_secret
        )
        if not client_id or not client_secret:
            if not check_data_exists('reddit'):
                parser.error(
                    "Reddit data collection requires credentials. Set REDDIT_CLIENT_ID and "
                    "REDDIT_CLIENT_SECRET in .env (see .env.example), or pass "
                    "--client_id/--client_secret."
                )
    
    print("\n" + "="*80)
    print(f"Transformer Training Demo: {args.dataset.capitalize()} Dataset")
    print("="*80)
    print(f"This script demonstrates the complete workflow for collecting {args.dataset} data")
    print("and training a transformer model.")
    
    # Check if data exists and if we should skip collection
    if args.skip_data_collection and check_data_exists(args.dataset):
        print(f"\nSkipping data collection as requested (--skip_data_collection) and {args.dataset} data file exists.")
    else:
        collect_data(args)
    
    # Train the model
    train_model(args)
    
    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nYou can now use the trained model for inference with:")
    print("python inference.py --prompt \"Your prompt here\"")

if __name__ == "__main__":
    main()
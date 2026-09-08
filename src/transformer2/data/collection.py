import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
from datasets import load_dataset

from transformer2.config import reddit_credentials
from transformer2.paths import data_dir

# Corpus filenames; the directory they live in comes from paths.data_dir().
BLUESKY_NAME = "bluesky_posts.parquet"
TINYSTORIES_NAME = "tinystories.parquet"
REDDIT_NAME = "reddit_posts.parquet"

def download_bluesky_dataset(output_path=None, max_samples=10000):
    """
    Download the Bluesky dataset from Hugging Face and save it as a parquet file.
    
    Args:
        output_path (str): Path to save the processed dataset
        max_samples (int): Maximum number of samples to include
    
    Returns:
        pd.DataFrame: The processed dataset
    """
    output_path = str(output_path or data_dir() / BLUESKY_NAME)
    print(f"Downloading Bluesky dataset from Hugging Face (alpindale/two-million-bluesky-posts)")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("alpindale/two-million-bluesky-posts", split="train")
    
    # Convert to pandas DataFrame for easier processing
    df = pd.DataFrame(dataset)
    
    # Limit to max_samples
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    # Process the text content
    processed_texts = []
    
    print(f"Processing {len(df)} Bluesky posts")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Extract the text content
        text = row.get('text', '')
        
        # Skip empty posts
        if not text or len(text.strip()) < 10:
            continue
            
        # Clean the text
        text = text.replace('\n\n', '\n').strip()
        
        # Add to processed texts
        processed_texts.append({
            'text': text,
            'id': row.get('id', ''),
            'author': row.get('author', ''),
            'created_at': row.get('created_at', '')
        })
    
    # Create a new DataFrame with processed texts
    processed_df = pd.DataFrame(processed_texts)
    
    # Save to parquet
    processed_df.to_parquet(output_path, index=False)
    print(f"Saved {len(processed_df)} Bluesky posts to {output_path}")
    
    return processed_df


def collect_tinystories(output_path=None, max_samples=None,
                        split="train"):
    """
    Download the TinyStories corpus.

    TinyStories is synthetic short fiction written with a deliberately small
    vocabulary, built to train models in the 1M-30M parameter range to produce
    coherent English. That is the size regime this project targets, which is why it
    suits it better than a general web scrape.

    Args:
        output_path (str): Where to write the parquet file
        max_samples (int, optional): Keep only this many stories. All of them if None.
        split (str): Dataset split to download ('train' or 'validation')

    Returns:
        pandas.DataFrame: The stories, with a single `text` column
    """
    output_path = str(output_path or data_dir() / TINYSTORIES_NAME)
    print(f"Downloading TinyStories ({split} split) from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", split=split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Keeping the first {max_samples:,} stories")

    df = pd.DataFrame({'text': dataset['text']})
    df = df[df['text'].notna()]

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_parquet(output_path, index=False)

    lengths = df['text'].str.len()
    print(f"Saved {len(df):,} stories to {output_path}")
    print(f"  characters: total {lengths.sum():,}, mean {lengths.mean():.0f}")
    return df


def collect_reddit_data(output_path=None, subreddits=None, limit=500, post_types=None, 
                        client_id=None, client_secret=None, user_agent=None):
    """
    Fetch Reddit posts for training data.
    
    Args:
        output_path (str): Path to save the processed dataset
        subreddits (list): Subreddits to collect posts from
        limit (int): Maximum posts per subreddit and post type
        post_types (list): Types of posts to fetch ('hot', 'new', 'top', 'rising')
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): User agent for Reddit API
        
    Returns:
        pd.DataFrame: The processed dataset
    """
    output_path = str(output_path or data_dir() / REDDIT_NAME)

    try:
        import praw
    except ImportError:
        print("praw package is required for Reddit data collection.")
        print("Please install it with: pip install praw")
        return None
    
    # Fall back to the environment / .env when the caller did not supply credentials
    client_id, client_secret, user_agent = reddit_credentials(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent
    )

    if not client_id or not client_secret:
        print("Reddit API credentials required.")
        print("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env (see .env.example),")
        print("or export them in your shell.")
        return None
    
    if subreddits is None:
        subreddits = ['datascience', 'MachineLearning', 'learnpython', 'Python']
    
    if post_types is None:
        post_types = ['hot', 'top']
    
    # Set up Reddit API
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        print(f"Connected to Reddit API successfully.")
    except Exception as e:
        print(f"Error connecting to Reddit API: {e}")
        return None
    
    all_posts = []
    
    # Fetch posts from each subreddit
    for subreddit in subreddits:
        for post_type in post_types:
            try:
                posts = fetch_subreddit_posts(reddit, subreddit, limit, post_type)
                all_posts.extend(posts)
                print(f"Fetched {len(posts)} {post_type} posts from r/{subreddit}")
            except Exception as e:
                print(f"Error fetching {post_type} posts from r/{subreddit}: {e}")
    
    # Process and save the data
    if all_posts:
        df = process_reddit_posts(all_posts)
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} Reddit posts to {output_path}")
        
        # Print some statistics
        print(f"\nData collection complete:")
        print(f"Total posts: {len(df)}")
        print(f"Average text length: {df['combined_text'].str.len().mean():.1f} characters")
        print(f"Total text: {df['combined_text'].str.len().sum():,} characters")
        
        return df
    else:
        print("No posts were collected. Check your API credentials and subreddit names.")
        return None

def fetch_subreddit_posts(reddit, subreddit_name, limit=100000, post_type='top'):
    """
    Fetch posts from a specified subreddit.
    
    Args:
        reddit (praw.Reddit): Reddit API instance
        subreddit_name (str): Name of the subreddit to fetch posts from
        limit (int): Maximum number of posts to fetch
        post_type (str): Type of posts to fetch ('hot', 'new', 'top', 'rising')
        
    Returns:
        list: List of dictionaries containing post data
    """
    subreddit = reddit.subreddit(subreddit_name)
    
    # Select the appropriate listing based on post_type
    if post_type == 'hot':
        posts = subreddit.hot(limit=limit)
    elif post_type == 'new':
        posts = subreddit.new(limit=limit)
    elif post_type == 'top':
        posts = subreddit.top(limit=limit)
    elif post_type == 'rising':
        posts = subreddit.rising(limit=limit)
    else:
        raise ValueError(f"Invalid post_type: {post_type}")
    
    post_data = []
    for post in tqdm(posts, total=limit, desc=f"Fetching {post_type} posts from r/{subreddit_name}"):
        # Skip stickied posts
        if post.stickied:
            continue
            
        # Get post data
        post_dict = {
            'id': post.id,
            'title': post.title,
            'selftext': post.selftext,
            'created_utc': post.created_utc,
            'score': post.score,
            'num_comments': post.num_comments,
            'subreddit': subreddit_name,
            'permalink': post.permalink,
            'url': post.url
        }
        
        # Get top-level comments
        post.comments.replace_more(limit=0)  # Skip "load more comments" links
        comments = []
        for comment in post.comments:
            if not comment.stickied and len(comment.body) > 50:  # Skip short comments
                comments.append(comment.body)
        
        post_dict['comments'] = comments
        post_data.append(post_dict)
        
        # Respect Reddit's API rate limits
        time.sleep(0.1)
    
    return post_data

def process_reddit_posts(posts):
    """
    Process and clean the Reddit post data.
    
    Args:
        posts (list): List of post dictionaries
        
    Returns:
        pandas.DataFrame: Processed DataFrame with text data
    """
    # Convert to DataFrame
    df = pd.DataFrame(posts)
    
    # Create a combined text field for training
    df['combined_text'] = df.apply(
        lambda row: f"Title: {row['title']}\n\nContent: {row['selftext']}\n\n" + 
                   "\n\n".join([f"Comment: {comment}" for comment in row['comments']]),
        axis=1
    )
    
    # Filter out posts with very short text
    df = df[df['combined_text'].str.len() > 200]
    
    # Clean the text
    df['combined_text'] = df['combined_text'].str.replace(r'\n{3,}', '\n\n', regex=True)
    
    return df

def collect_data(dataset_type, output_path=None, **kwargs):
    """
    Unified function to collect data from different sources.
    
    Args:
        dataset_type (str): Type of dataset to collect ('reddit', 'bluesky', 'tinystories')
        output_path (str, optional): Path to save the dataset
        **kwargs: Additional arguments for specific data collection functions

    Returns:
        pd.DataFrame: The collected dataset
    """
    if dataset_type == 'tinystories':
        if output_path is None:
            output_path = str(data_dir() / TINYSTORIES_NAME)
        return collect_tinystories(output_path=output_path, **kwargs)

    elif dataset_type == 'bluesky':
        if output_path is None:
            output_path = str(data_dir() / BLUESKY_NAME)
        return download_bluesky_dataset(output_path=output_path, **kwargs)
    
    elif dataset_type == 'reddit':
        if output_path is None:
            output_path = str(data_dir() / REDDIT_NAME)
        return collect_reddit_data(output_path=output_path, **kwargs)
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def main():
    parser = argparse.ArgumentParser(description='Collect training data for transformer models')
    
    # Main arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['reddit', 'bluesky', 'tinystories'],
                        help='Type of dataset to collect')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the collected data')
    
    # Bluesky arguments
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to download (for TinyStories)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to include (Bluesky, TinyStories). '
                             'All of them if omitted.')
    
    # Reddit arguments
    parser.add_argument('--client_id', type=str, default=None,
                        help='Reddit API client ID. Prefer REDDIT_CLIENT_ID in .env; '
                             'credentials on the command line leak into shell history')
    parser.add_argument('--client_secret', type=str, default=None,
                        help='Reddit API client secret. Prefer REDDIT_CLIENT_SECRET in .env')
    parser.add_argument('--user_agent', type=str, default=None,
                        help='User agent string for Reddit API (or REDDIT_USER_AGENT)')
    parser.add_argument('--subreddits', nargs='+', default=None,
                        help='List of subreddits to fetch posts from (for Reddit)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Maximum number of posts to fetch per subreddit (for Reddit)')
    parser.add_argument('--post_types', nargs='+', default=None,
                        help='Types of posts to fetch: hot, new, top, rising (for Reddit)')
    
    args = parser.parse_args()
    
    # Collect data based on dataset type
    if args.dataset == 'tinystories':
        collect_data(
            dataset_type='tinystories',
            output_path=args.output,
            max_samples=args.max_samples,
            split=args.split
        )
    elif args.dataset == 'bluesky':
        collect_data(
            dataset_type='bluesky',
            output_path=args.output,
            max_samples=args.max_samples
        )
    elif args.dataset == 'reddit':
        collect_data(
            dataset_type='reddit',
            output_path=args.output,
            client_id=args.client_id,
            client_secret=args.client_secret,
            user_agent=args.user_agent,
            subreddits=args.subreddits,
            limit=args.limit,
            post_types=args.post_types
        )

if __name__ == "__main__":
    main()
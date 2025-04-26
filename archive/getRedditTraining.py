'''write code to get reddit text data for training my LLM '''

import praw
import pandas as pd
import os
import time
from tqdm import tqdm
import argparse

def setup_reddit_api(client_id, client_secret, user_agent):
    """
    Set up and return a Reddit API instance using PRAW.
    
    Args:
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): User agent string for API requests
        
    Returns:
        praw.Reddit: Authenticated Reddit API instance
    """
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

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

def process_posts(posts):
    """
    Process and clean the post data.
    
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

def save_data(df, output_path='data/reddit_posts.parquet'):
    """
    Save the processed data to a parquet file.
    
    Args:
        df (pandas.DataFrame): DataFrame with processed text data
        output_path (str): Path to save the parquet file
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} posts to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fetch Reddit posts for LLM training')
    parser.add_argument('--client_id', default='***REMOVED***', help='Reddit API client ID')
    parser.add_argument('--client_secret', default='***REMOVED***', help='Reddit API client secret')
    parser.add_argument('--user_agent', default='python:llm-training-data:v1.0 (by /u/analyticmodels)', 
                        help='User agent string for Reddit API')
    parser.add_argument('--subreddits', nargs='+', default=['datascience', 'MachineLearning', 'learnpython', 'Python'],
                        help='List of subreddits to fetch posts from')
    parser.add_argument('--limit', type=int, default=500, 
                        help='Maximum number of posts to fetch per subreddit')
    parser.add_argument('--post_types', nargs='+', default=['hot', 'top'],
                        help='Types of posts to fetch (hot, new, top, rising)')
    parser.add_argument('--output', default='data/reddit_posts.parquet',
                        help='Output path for the parquet file')
    
    args = parser.parse_args()
    
    # Set up Reddit API
    reddit = setup_reddit_api(args.client_id, args.client_secret, args.user_agent)
    
    all_posts = []
    
    # Fetch posts from each subreddit
    for subreddit in args.subreddits:
        for post_type in args.post_types:
            try:
                posts = fetch_subreddit_posts(reddit, subreddit, args.limit, post_type)
                all_posts.extend(posts)
                print(f"Fetched {len(posts)} {post_type} posts from r/{subreddit}")
            except Exception as e:
                print(f"Error fetching {post_type} posts from r/{subreddit}: {e}")
    
    # Process and save the data
    if all_posts:
        df = process_posts(all_posts)
        save_data(df, args.output)
        
        # Print some statistics
        print(f"\nData collection complete:")
        print(f"Total posts: {len(df)}")
        print(f"Average text length: {df['combined_text'].str.len().mean():.1f} characters")
        print(f"Total text: {df['combined_text'].str.len().sum():,} characters")
        
        # Sample output
        print("\nSample post:")
        sample = df.sample(1).iloc[0]
        print(f"Title: {sample['title']}")
        print(f"Subreddit: r/{sample['subreddit']}")
        print(f"Score: {sample['score']}")
        print(f"Comments: {len(sample['comments'])}")
        print(f"Text length: {len(sample['combined_text'])} characters")
    else:
        print("No posts were collected. Check your API credentials and subreddit names.")

if __name__ == "__main__":
    main()

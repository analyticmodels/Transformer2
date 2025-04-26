import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def download_bluesky_dataset(output_path="data/bluesky_posts.parquet", max_samples=10000):
    """
    Download the Bluesky dataset from Hugging Face and save it as a parquet file.
    
    Args:
        output_path (str): Path to save the processed dataset
        max_samples (int): Maximum number of samples to include
    
    Returns:
        pd.DataFrame: The processed dataset
    """
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and process Bluesky dataset')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to include')
    parser.add_argument('--output_path', type=str, default='data/bluesky_posts.parquet',
                        help='Path to save the processed dataset')
    
    args = parser.parse_args()
    
    download_bluesky_dataset(output_path=args.output_path, max_samples=args.max_samples)

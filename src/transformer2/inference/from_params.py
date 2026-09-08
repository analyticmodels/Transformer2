#!/usr/bin/env python3
"""
Generate text using optimized parameters.

This script loads parameters from a JSON file created by optimize_inference.py
and uses them to generate text with the transformer model.

Usage:
    python generate_text.py --prompt "Your prompt here" --params inference_params.json
"""

import argparse
import json
import torch
from transformer2.inference.generate import load_model, generate_text
import os
from transformer2.paths import output_dir

def main():
    parser = argparse.ArgumentParser(description='Generate text using optimized parameters')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Prompt to generate text from')
    parser.add_argument('--params', type=str, default=None,
                        help='JSON file with tuned parameters (default: <outputs>/inference_params.json)')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset_type', type=str, default='bluesky',
                        help='Type of dataset the model was trained on')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of text samples to generate')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Override maximum tokens to generate')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Attention head count the model was trained with '
                             '(required for checkpoints that record no config)')
    
    args = parser.parse_args()
    
    # Check if parameter file exists
    args.params = str(args.params or output_dir() / 'inference_params.json')
    if not os.path.exists(args.params):
        print(f"Error: Parameter file {args.params} not found.")
        return
    
    # Load parameters from file
    with open(args.params, 'r') as f:
        params_data = json.load(f)
    
    best_params = params_data['best_params']
    print(f"Using optimized parameters: {best_params}")
    
    # Override max_tokens if specified
    if args.max_tokens is not None:
        best_params['max_tokens'] = args.max_tokens
        print(f"Overriding max_tokens to {args.max_tokens}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    
    # Direct loading approach
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use the improved load_model function from inference.py
    # This automatically detects model dimensions from the checkpoint
    model, tokenizer = load_model(
        model_path=args.model_path,
        dataset_type=args.dataset_type,
        num_heads=args.num_heads
    )
    
    # Generate multiple text samples
    print(f"\nGenerating {args.num_samples} samples with prompt: '{args.prompt}'")
    print("-" * 50)
    
    for i in range(args.num_samples):
        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=best_params['max_tokens'],
            temperature=best_params['temperature'],
            top_k=best_params['top_k'],
            top_p=best_params['top_p'],
            repetition_penalty=best_params['repetition_penalty']
        )
        
        print(f"\nSample {i+1}:")
        print(f"{args.prompt}{generated}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()
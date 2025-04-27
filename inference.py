import torch
import os
from coreTransformer import SmallTransformer
from tokenizer import create_tokenizer

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(model_path='best_model.pt', model=None, tokenizer=None, dataset_type="bluesky"):
    """
    Load a transformer model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        model (nn.Module, optional): Model to load weights into. If None, a new SmallTransformer will be created.
        tokenizer: Tokenizer to use. If None, a new tokenizer will be created.
        dataset_type (str): Type of dataset the model was trained on.
        
    Returns:
        tuple: (model, tokenizer)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type=dataset_type)
    
    # Create model if not provided
    if model is None:
        model = SmallTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=256,      # Default architecture
            num_heads=8,      
            num_layers=6,     
            d_ff=1024,        
            max_seq_len=512,
            dropout=0.1
        )
    
    # Load model weights
    try:
        # Skip the weights_only=True attempt since we know it causes warnings
        # with pickle protocol 5 and just use weights_only=False directly
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, 
                 top_k=50, top_p=0.9, repetition_penalty=1.2, model_path=None, debug=False):
    """
    Generate text using a transformer model.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        prompt (str): The text prompt to start generation from
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling (higher = more random)
        top_k (int): Number of highest probability tokens to keep for sampling
        top_p (float): Cumulative probability threshold for nucleus sampling
        repetition_penalty (float): Penalty applied to previously generated tokens
        model_path (str, optional): Path to model checkpoint if loading is needed
        debug (bool): Whether to print debug information
        
    Returns:
        str: The generated text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model if path is provided
    if model_path:
        model, tokenizer = load_model(model_path, model, tokenizer)
    
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    if debug:
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Input tokens: {input_ids}")
    
    generated_tokens = []
    
    # Define special tokens based on tokenizer type
    special_tokens = []
    eos_token_id = None
    
    # Check if tokenizer has all_special_ids attribute (HuggingFace tokenizers)
    if hasattr(tokenizer, 'all_special_ids'):
        special_tokens = tokenizer.all_special_ids
        eos_token_id = tokenizer.eos_token_id
    else:
        # For ByteLevelBPETokenizerWrapper, we'll use some defaults
        special_tokens = list(range(4))  # Assuming first 4 tokens are special
        eos_token_id = 2  # </s> is often token 2
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            try:
                outputs = model(input_ids, attention_mask)
                
                if debug and i == 0:
                    print(f"Model output shape: {outputs.shape}")
                
                # Handle different output formats
                if len(outputs.shape) == 3:  # [batch_size, seq_len, vocab_size]
                    next_token_logits = outputs[:, -1, :]
                elif len(outputs.shape) == 2:  # [batch_size, vocab_size]
                    next_token_logits = outputs
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
                    
                if debug and i == 0:
                    print(f"Next token logits shape: {next_token_logits.shape}")
            except Exception as e:
                if debug:
                    print(f"Error in forward pass: {e}")
                    print(f"Input shape: {input_ids.shape}")
                    print(f"Attention mask shape: {attention_mask.shape}")
                raise
            
            # Filter special tokens except EOS
            for token_id in special_tokens:
                if token_id != eos_token_id:
                    next_token_logits[:, token_id] = float('-inf')
            
            # Apply repetition penalty
            for token in set(generated_tokens):
                next_token_logits[0, token] /= repetition_penalty
            
            # Temperature and top-k/p sampling
            next_token_logits = next_token_logits / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Filter with top-k and top-p
            top_k_probs, top_k_indices = torch.topk(next_token_probs, min(top_k, next_token_probs.size(-1)))
            cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
            remove_indices = cumulative_probs > top_p
            remove_indices[..., 1:] = remove_indices[..., :-1].clone()
            remove_indices[..., 0] = 0
            top_k_probs[remove_indices] = 0.0
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            next_token = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]
            
            if next_token == eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            # Ensure correct dimensions before concatenating
            next_token_2d = next_token.unsqueeze(0)  # [1, 1]
            if len(input_ids.shape) == 3:
                next_token_3d = next_token_2d.unsqueeze(0)  # [1, 1, 1]
                input_ids = torch.cat([input_ids, next_token_3d], dim=1)
            else:
                input_ids = torch.cat([input_ids, next_token_2d], dim=1)
            
            # Add attention mask for new token
            if len(attention_mask.shape) == 3:
                new_mask = torch.ones((1, 1, 1), dtype=torch.long, device=device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            else:
                new_mask = torch.ones((1, 1), dtype=torch.long, device=device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)
    
    try:
        return tokenizer.decode(generated_tokens)
    except:
        return "[Error decoding tokens]"

def run_interactive_demo(model_path='best_model.pt', dataset_type="bluesky"):
    """
    Run an interactive demo for text generation.
    
    Args:
        model_path (str): Path to the model checkpoint
        dataset_type (str): Type of dataset the model was trained on
    """
    print("\n" + "="*50)
    print("Text Generation Demo")
    print("="*50)
    
    # Load model and tokenizer
    model, tokenizer = load_model(model_path, dataset_type=dataset_type)
    
    # Default generation parameters
    params = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 100
    }
    
    print(f"\nModel loaded from: {model_path}")
    print(f"Generation parameters: {params}")
    print("\nEnter a prompt to generate text. Type 'exit' to quit.")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'exit':
            break
            
        try:
            generated = generate_text(model, tokenizer, prompt, **params)
            print(f"\nGenerated: {prompt}{generated}")
        except Exception as e:
            print(f"Error generating text: {e}")

# Main function for CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Text generation with transformer models')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset_type', type=str, default='bluesky',
                        help='Type of dataset the model was trained on (bluesky, reddit, imdb)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt to generate from (if not provided, will run interactive demo)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of highest probability tokens to keep for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Cumulative probability threshold for nucleus sampling')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help='Penalty applied to previously generated tokens')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    if args.prompt:
        # Single generation mode
        model, tokenizer = load_model(args.model_path, dataset_type=args.dataset_type)
        
        generated = generate_text(
            model, 
            tokenizer, 
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            debug=args.debug
        )
        
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {args.prompt}{generated}")
    else:
        # Interactive demo mode
        run_interactive_demo(args.model_path, args.dataset_type)
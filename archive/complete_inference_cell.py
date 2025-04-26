# Test with different parameters
import torch
from torch import nn
from datasets import load_dataset
from transformers import GPT2Tokenizer
import math
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

from coreTransformer import *
from tokenizer import *
from trainingSetup import *
#from curriculumTraining import *
from dataloader import *
# Do not import from inferenceTest to avoid conflicts
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
k    # Pass the dataset_type to create_tokenizer
    tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type="bluesky")

    # Create model with the same architecture as the trained model
    model = SmallTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,      # Match the trained model dimensions
        num_heads=8,      
        num_layers=6,     
        d_ff=1024,        
        max_seq_len=512,
        dropout=0.1
    )
    
    # Define the test_inference function with dimension handling fix
    def test_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, 
                      top_k=50, top_p=0.9, repetition_penalty=1.2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        try:
            checkpoint = torch.load('best_model.pt', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Allow missing keys
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return ""
        
        model = model.to(device)
        model.eval()
        
        # Clean tokenization
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        
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
            # Typically special tokens are in the beginning of the vocabulary
            special_tokens = list(range(4))  # Assuming first 4 tokens are special
            eos_token_id = 2  # </s> is often token 2
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, attention_mask)
                
                # Handle model output shape
                if len(outputs.shape) == 3:  # [batch_size, seq_len, vocab_size]
                    next_token_logits = outputs[:, -1, :]  # Take logits for the last token
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")
                
                print(f"Model output shape: {outputs.shape}")
                print(f"Next token logits shape: {next_token_logits.shape}")
                
                # Ensure next_token_logits is 2D
                assert len(next_token_logits.shape) == 2, f"Expected 2D logits, got {next_token_logits.shape}"
                
                # Filter special tokens except EOS
                if special_tokens:
                    for token_id in special_tokens:
                        if token_id != eos_token_id and token_id < next_token_logits.size(1):
                            next_token_logits[:, token_id] = float('-inf')
                
                # Apply repetition penalty
                if generated_tokens:  # Ensure generated_tokens is not empty
                    for token in set(generated_tokens):
                        if token < next_token_logits.size(1):  # Ensure token is within bounds
                            next_token_logits[:, token] /= repetition_penalty
                
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
                print(f"Prob: {top_k_probs[0]}")
                next_token = top_k_indices[0, torch.multinomial(top_k_probs[0], 1)]
                print(f"Next token: {next_token}")
                if next_token == eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
        
        try:
            return tokenizer.decode(generated_tokens)
        except:
            return "[Error decoding tokens]"
    
    test_prompts = [
        "In the distant future,",
        "The scientist discovered",
        "Once upon a time in",
    ]
    
    # Different parameter sets to try
    sampling_params = [
        {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_new_tokens": 50
        },
        {
            "temperature": 0.9,
            "top_k": 40,
            "top_p": 0.85,
            "repetition_penalty": 1.3,
            "max_new_tokens": 50
        },
        {
            "temperature": 0.8,
            "top_k": 30,
            "top_p": 0.8,
            "repetition_penalty": 1.4,
            "max_new_tokens": 50
        }
    ]
    
    print("\nTesting inference with different parameters:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        for params in sampling_params:
            try:
                print(f"\nParameters: {params}")
                generated = test_inference(model, tokenizer, prompt, **params)
                print(f"Generated: {prompt}{generated}")
            except Exception as e:
                print(f"Error generating text with parameters {params}: {e}")

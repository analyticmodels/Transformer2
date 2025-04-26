import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer 

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
            
            # Handle different output formats
            if len(outputs.shape) == 3:  # [batch_size, seq_len, vocab_size]
                next_token_logits = outputs[:, -1, :]
            elif len(outputs.shape) == 2:  # [batch_size, vocab_size]
                next_token_logits = outputs
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
            
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
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
    
    try:
        return tokenizer.decode(generated_tokens)
    except:
        return "[Error decoding tokens]"

# Test with different parameters
if __name__ == "__main__":
    test_prompts = [
        "In the distant future,",
        "The scientist discovered",
        "Once upon a time in",
    ]
    
    # Different parameter sets to try
    sampling_params = [
        {
            "temperature": 0.6,         # More conservative temperature
            "top_k": 100,              # Larger top-k
            "top_p": 0.95,             # Higher top-p
            "repetition_penalty": 1.5,  # Stronger repetition penalty
            "max_new_tokens": 50
        },
        {
            "temperature": 0.7,
            "top_k": 80,
            "top_p": 0.92,
            "repetition_penalty": 1.4,
            "max_new_tokens": 50
        },
        {
            "temperature": 0.8,
            "top_k": 60,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
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

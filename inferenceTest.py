import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer 

def test_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, 
                  top_k=50, top_p=0.9, repetition_penalty=1.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with better error handling
    try:
        checkpoint = torch.load('best_model.pt', weights_only=True)
        # Check if we have the expected keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # Try using checkpoint directly
            
        # Remove any unexpected keys
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
            
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return ""
    
    model = model.to(device)
    model.eval()
    
    try:
        # Encode prompt with error checking
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        
        # Track generated tokens for repetition penalty
        generated = list(input_ids[0].cpu().numpy())
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                try:
                    # Forward pass with size checks
                    if input_ids.size(1) > model.max_seq_len:
                        input_ids = input_ids[:, -model.max_seq_len:]
                        attention_mask = attention_mask[:, -model.max_seq_len:]
                    
                    outputs = model(input_ids, attention_mask)
                    next_token_logits = outputs[:, -1, :].float()  # Ensure float type
                    
                    # Apply repetition penalty
                    for token in set(generated[-50:]):  # Only consider last 50 tokens
                        next_token_logits[0, token] /= repetition_penalty
                    
                    # Apply temperature with stability check
                    next_token_logits = next_token_logits / max(temperature, 1e-8)
                    
                    # Apply top-k filtering
                    top_k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs[0], 1)
                    
                    # Early stopping checks
                    if next_token.item() == tokenizer.get_pad_token_id():
                        break
                    
                    # Update input tensors
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), dtype=torch.long, device=device)
                    ], dim=1)
                    
                    generated.append(next_token.item())
                    
                    # Check for repetition
                    if len(generated) > 4:
                        last_tokens = generated[-4:]
                        if len(set(last_tokens)) == 1:
                            break
                    
                except RuntimeError as e:
                    print(f"Error during generation step: {e}")
                    break
            
            # Decode with error handling
            try:
                generated_tokens = generated[len(encoding['input_ids']):]
                return tokenizer.decode(generated_tokens)
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                return ""
                
    except Exception as e:
        print(f"Error during inference: {e}")
        return ""

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
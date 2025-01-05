import torch
from torch import nn

def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, temperature=0.7, filter_value=-float('Inf')):
    """Filter logits using top-k and nucleus (top-p) sampling"""
    # Temperature sampling
    logits = logits / temperature
    
    # Top-k filtering
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    # Top-p filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits

def test_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, 
                  top_k=50, top_p=0.9, repetition_penalty=1.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    try:
        checkpoint = torch.load('best_model.pt', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return ""
    
    model = model.to(device)
    model.eval()
    
    # Encode prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    
    # Track generated tokens for repetition penalty
    generated = list(input_ids[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids, attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Apply repetition penalty
            for token in set(generated):
                next_token_logits[0, token] /= repetition_penalty
            
            # Apply filtering
            next_token_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs[0], 1)
            
            # Stop if we predict a padding token
            if next_token.item() == tokenizer.get_pad_token_id():
                break
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            generated.append(next_token.item())
            
            # Early stopping for repetitive sequences
            if len(generated) > 4:
                # Check for repeating patterns
                last_tokens = generated[-4:]
                if len(set(last_tokens)) == 1:  # All recent tokens are the same
                    break
    
    return tokenizer.decode(generated[len(encoding['input_ids']):])

# Test with different parameters
if __name__ == "__main__":
    test_prompts = [
        "In the distant future,",
        "The scientist discovered",
        "Once upon a time in",
    ]
    
    sampling_params = [
        {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_k": 40, "top_p": 0.85, "repetition_penalty": 1.3},
        {"temperature": 0.9, "top_k": 30, "top_p": 0.8, "repetition_penalty": 1.4},
    ]
    
    print("\nTesting inference with different parameters:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        for params in sampling_params:
            try:
                generated = test_inference(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=50,
                    **params
                )
                print(f"\nParameters: {params}")
                print(f"Generated: {prompt}{generated}")
            except Exception as e:
                print(f"Error generating text: {e}")
def test_inference(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Load model with correct state dict access
    try:
        checkpoint = torch.load('best_model.pt', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return ""
    
    # Move model to device
    model = model.to(device)
    
    # Encode the prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    
    generated_tokens = []
    
    with torch.no_grad():
        try:
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = model(input_ids, attention_mask)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                
                # Ensure logits are finite
                top_k_logits = torch.where(
                    torch.isfinite(top_k_logits),
                    top_k_logits,
                    torch.zeros_like(top_k_logits)
                )
                
                # Apply softmax with better numerical stability
                top_k_logits = top_k_logits - torch.max(top_k_logits, dim=-1, keepdim=True)[0]
                exp_logits = torch.exp(top_k_logits)
                next_token_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
                
                # Ensure probabilities are valid
                next_token_probs = torch.where(
                    torch.isfinite(next_token_probs),
                    next_token_probs,
                    torch.ones_like(next_token_probs) / next_token_probs.size(-1)
                )
                
                # Sample from the filtered distribution
                try:
                    next_token_idx = torch.multinomial(next_token_probs[0], 1)
                    next_token = top_k_indices[0, next_token_idx]
                except RuntimeError:
                    print("Error sampling next token, using most likely token instead")
                    next_token = top_k_indices[0, 0].unsqueeze(0)
                
                # Append the new token
                generated_tokens.append(next_token.item())
                
                # Update input_ids and attention_mask for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
                
        except Exception as e:
            print(f"Error during generation: {e}")
            return tokenizer.decode(generated_tokens)
    
    return tokenizer.decode(generated_tokens)

# # Example usage
# if __name__ == "__main__":
#     # Test prompts
#     test_prompts = [
#         "In the distant future,",
#         "The scientist discovered",
#         "Once upon a time in",
#     ]
    
#     print("\nTesting inference:")
#     for prompt in test_prompts:
#         print(f"\nPrompt: {prompt}")
#         try:
#             generated = test_inference(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt=prompt,
#                 max_new_tokens=50,
#                 temperature=0.9,
#                 top_k=50
#             )
#             print(f"Generated: {prompt}{generated}")
#         except Exception as e:
#             print(f"Error generating text for prompt '{prompt}': {e}")
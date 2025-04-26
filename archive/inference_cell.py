# Test with different parameters
if __name__ == "__main__":
    _, _, tokenizer = get_dataloaders(batch_size=8, dataset_type="bluesky")

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

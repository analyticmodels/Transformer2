import torch
from coreTransformer import SmallTransformer
from tokenizer import create_tokenizer

# ollama_integration is imported lazily inside the use_ollama branches so that the
# local inference path does not depend on it (or on `requests`) at all.

# Note: this module deliberately does not set TOKENIZERS_PARALLELISM. Doing so at import
# time is process-wide, and disabling it makes the batched tokenization in dataloader.py
# roughly six times slower for anything that imports this module first (a notebook that
# generates and then trains, for example). Inference encodes one prompt at a time and
# forks nothing, so it gains nothing from the setting either way.

def infer_architecture(state_dict, config=None, num_heads_override=None):
    """
    Work out the architecture of a checkpoint.

    Weight shapes are authoritative wherever they can be read, because a recorded
    config can disagree with the weights it was saved alongside. Only the head count
    cannot be recovered from the weights (the Q/K/V projections are d_model x d_model
    for any number of heads), so it comes from the config or an explicit override.

    Args:
        state_dict (dict): The checkpoint's model_state_dict
        config (dict, optional): The checkpoint's recorded config, if present
        num_heads_override (int, optional): Head count supplied by the caller, which
            takes priority over the recorded config

    Returns:
        dict: Keyword arguments for SmallTransformer (minus dropout)

    Raises:
        ValueError: If the head count is neither recorded nor supplied, since guessing
            it produces a model that loads cleanly and generates noise.
    """
    # A config entry that is present but None must not defeat the fallbacks below
    config = {k: v for k, v in (config or {}).items() if v is not None}

    # d_model: the width of any layer norm in the first block
    norm_key = 'transformer_blocks.0.norm1.weight'
    if norm_key in state_dict:
        d_model = state_dict[norm_key].shape[0]
    else:
        d_model = config.get('d_model', 1024)

    # vocab_size: the embedding table is [vocab_size, d_model]
    if 'embedding.weight' in state_dict:
        vocab_size = state_dict['embedding.weight'].shape[0]
    else:
        vocab_size = config.get('vocab_size')

    # d_ff: the output width of the first feed-forward projection
    ff_bias_key = 'transformer_blocks.0.ff.0.bias'
    ff_weight_key = 'transformer_blocks.0.ff.0.weight'
    if ff_bias_key in state_dict:
        d_ff = state_dict[ff_bias_key].shape[0]
    elif ff_weight_key in state_dict:
        d_ff = state_dict[ff_weight_key].shape[0]
    else:
        d_ff = config.get('d_ff', d_model * 4)

    # num_layers: count the transformer blocks present in the weights
    num_layers = 0
    while f'transformer_blocks.{num_layers}.norm1.weight' in state_dict:
        num_layers += 1
    if num_layers == 0:
        num_layers = config.get('num_layers', 6)

    # max_seq_len: the positional encoding is [1, max_seq_len, d_model]
    max_seq_len = 512
    if 'positional_encoding' in state_dict:
        pos_shape = state_dict['positional_encoding'].shape
        if len(pos_shape) == 3:
            max_seq_len = pos_shape[1]

    # num_heads cannot be derived from the weights: Q/K/V are d_model x d_model for any
    # head count, so a wrong value loads cleanly and silently reshapes attention. There is
    # no safe guess -- checkpoints in this repo were trained with 8 and with 16 at the same
    # d_model -- so refuse rather than invent one.
    if num_heads_override is not None:
        num_heads = num_heads_override
        heads_source = "override"
    elif 'num_heads' in config:
        num_heads = config['num_heads']
        heads_source = "checkpoint config"
    else:
        raise ValueError(
            "This checkpoint records no num_heads, and the head count cannot be recovered "
            "from the weights (Q/K/V are d_model x d_model for every head count). A wrong "
            "value loads without error and generates noise, so pass the head count the "
            "model was trained with explicitly, e.g. --num_heads 8."
        )

    if d_model % num_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} from the {heads_source} does not divide d_model={d_model}. "
            f"Pass the head count the model was trained with via --num_heads."
        )

    print(f"Checkpoint architecture: vocab_size={vocab_size}, d_model={d_model}, "
          f"num_heads={num_heads}, num_layers={num_layers}, d_ff={d_ff}, max_seq_len={max_seq_len}")
    if heads_source == "checkpoint config":
        print("  num_heads came from the checkpoint config and cannot be verified against "
              "the weights; pass --num_heads if generation looks like noise.")

    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'max_seq_len': max_seq_len,
    }

def load_model(model_path='best_model.pt', model=None, tokenizer=None, dataset_type="bluesky",
               use_ollama=False, ollama_model="llama2", num_heads=None):
    """
    Load a transformer model from checkpoint.

    Args:
        model_path (str): Path to the model checkpoint
        model (nn.Module, optional): Model to load weights into. If None, a new SmallTransformer will be created.
        tokenizer: Tokenizer to use. If None, a new tokenizer will be created.
        dataset_type (str): Type of dataset the model was trained on.
        use_ollama (bool): Serve generation from a local Ollama model instead of a checkpoint.
        ollama_model (str): Name of the Ollama model to use when use_ollama is set.
        num_heads (int, optional): Override the attention head count recorded in the checkpoint.

    Returns:
        tuple: (model, tokenizer)

    Raises:
        RuntimeError: If the checkpoint cannot be read or its weights do not fit the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For Ollama models, create the model and tokenizer directly
    if use_ollama:
        from ollama_integration import OllamaModel, OllamaTokenizer
        if tokenizer is None:
            tokenizer = OllamaTokenizer(ollama_model)
        if model is None:
            model = OllamaModel(model_name=ollama_model)
        return model, tokenizer

    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type=dataset_type)
    
    # For local models, load the checkpoint to get model config
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Could not read checkpoint {model_path}: {e}") from e

    if 'model_state_dict' not in checkpoint:
        raise RuntimeError(f"Checkpoint {model_path} has no 'model_state_dict' entry")

    state_dict = checkpoint['model_state_dict']

    # Build a matching model if the caller did not supply one
    if model is None:
        architecture = infer_architecture(
            state_dict,
            config=checkpoint.get('config'),
            num_heads_override=num_heads
        )

        # The tokenizer has to agree with the embedding table. create_tokenizer silently
        # falls back to GPT-2 (50257 tokens) when tokenizer/vocab.json is missing, which
        # would otherwise surface as an opaque shape error from load_state_dict.
        tokenizer_vocab_size = tokenizer.get_vocab_size()
        if architecture['vocab_size'] != tokenizer_vocab_size:
            raise RuntimeError(
                f"Tokenizer vocabulary ({tokenizer_vocab_size} tokens) does not match "
                f"checkpoint {model_path} ({architecture['vocab_size']} tokens). The "
                f"tokenizer in tokenizer/ is not the one this model was trained with; "
                f"regenerate it with: python tokenizer.py --dataset_type <dataset>"
            )

        model = SmallTransformer(dropout=0.1, **architecture)

    # Load strictly: a silently half-loaded model generates noise that looks like a
    # bad checkpoint rather than a bug, so surface the mismatch instead.
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Checkpoint {model_path} does not fit the model architecture: {e}\n"
            f"If the checkpoint was written before the config fix, pass an explicit "
            f"--num_heads, and check that the tokenizer matches the one used for training."
        ) from e

    print(f"Model loaded successfully from {model_path}")

    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7,
                 top_k=50, top_p=0.9, repetition_penalty=1.2, model_path=None, debug=False,
                 use_ollama=False, ollama_model="llama2", dataset_type="bluesky"):
    """
    Generate a continuation for a prompt.

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
        use_ollama (bool): Generate with an Ollama model instead of the local transformer
        ollama_model (str): Name of the Ollama model to use when use_ollama is set
        dataset_type (str): Dataset the tokenizer was trained on, used only when reloading

    Returns:
        str: The continuation only. The prompt is not included, so callers that want the
            full text should print the prompt themselves.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model if path is provided
    if model_path:
        model, tokenizer = load_model(
            model_path,
            model,
            tokenizer,
            dataset_type=dataset_type,
            use_ollama=use_ollama,
            ollama_model=ollama_model
        )

    model = model.to(device)
    model.eval()

    # Tokenize input
    if use_ollama:
        input_ids = torch.tensor([tokenizer.tokenize(prompt)], dtype=torch.long).to(device)
    else:
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    if debug:
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Input tokens: {input_ids}")
    
    if use_ollama:
        # For Ollama models, use the built-in generate method
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # Extract the newly generated tokens
        new_tokens = generated[0, input_ids.size(1):].tolist()
        return tokenizer.decode(new_tokens)
    else:
        # The prompt is tracked separately from the continuation: it feeds the repetition
        # penalty but must stay out of the return value, which is the continuation only.
        prompt_tokens = input_ids[0].tolist()
        generated_tokens = []

        # EOS ends generation; the remaining special tokens are never valid samples
        eos_token_id = tokenizer.get_eos_token_id()
        suppressed_token_ids = [
            token_id for token_id in tokenizer.get_special_token_ids()
            if token_id != eos_token_id
        ]

        # The learned positional encoding only covers max_seq_len positions
        max_seq_len = getattr(model, 'max_seq_len', None)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Keep the context inside the positional encoding window
                if max_seq_len is not None and input_ids.size(1) > max_seq_len:
                    input_ids = input_ids[:, -max_seq_len:]
                    attention_mask = attention_mask[:, -max_seq_len:]

                # Get model predictions
                outputs = model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs[:, -1, :]

                # Never sample pad/bos/unk
                for token_id in suppressed_token_ids:
                    next_token_logits[:, token_id] = float('-inf')

                # Apply the repetition penalty to the raw logits, before temperature.
                # The penalty has to be sign-aware: most logits here are negative, and
                # dividing a negative logit makes it larger, which boosts exactly the
                # token being penalized.
                if repetition_penalty != 1.0:
                    seen = list(set(prompt_tokens + generated_tokens))
                    seen_logits = next_token_logits[0, seen]
                    next_token_logits[0, seen] = torch.where(
                        seen_logits < 0,
                        seen_logits * repetition_penalty,
                        seen_logits / repetition_penalty
                    )

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering, clamped to the vocabulary size
                if top_k > 0:
                    k = min(int(top_k), next_token_logits.size(-1))
                    kth_logit = torch.topk(next_token_logits, k)[0][..., -1, None]
                    next_token_logits[next_token_logits < kth_logit] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = float('-inf')

                # Sample from the filtered distribution. multinomial returns [1, 1] here,
                # which is already the shape needed to append to input_ids.
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                # If debug mode is on, print token info
                if debug:
                    print(f"\nToken {len(generated_tokens)}: {token_id}")
                    print(f"Probability: {probs[0, token_id].item():.4f}")
                    try:
                        print(f"Text: {tokenizer.decode([token_id])}")
                    except Exception:
                        print("[Could not decode token]")

                if eos_token_id is not None and token_id == eos_token_id:
                    break

                generated_tokens.append(token_id)

                # Extend the context with the sampled token
                input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
                new_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    try:
        return tokenizer.decode(generated_tokens)
    except Exception as e:
        print(f"Error decoding generated tokens: {e}")
        return ""

def run_interactive_demo(model_path='best_model.pt', dataset_type="bluesky", use_ollama=False,
                         ollama_model="llama2", num_heads=None):
    """
    Run an interactive demo for text generation.

    Args:
        model_path (str): Path to the model checkpoint
        dataset_type (str): Type of dataset the model was trained on
        use_ollama (bool): Generate with an Ollama model instead of the local transformer
        ollama_model (str): Name of the Ollama model to use when use_ollama is set
        num_heads (int, optional): Override the attention head count recorded in the checkpoint
    """
    print("\n" + "="*50)
    print("Text Generation Demo")
    print("="*50)

    # Load model and tokenizer
    model, tokenizer = load_model(
        model_path,
        dataset_type=dataset_type,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        num_heads=num_heads
    )

    # Default generation parameters. use_ollama has to travel with them, otherwise
    # an Ollama model would be driven through the local sampling loop.
    params = {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "max_new_tokens": 100,
        "use_ollama": use_ollama,
        "ollama_model": ollama_model
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
                        help='Path to the model checkpoint (not used with --use_ollama)')
    parser.add_argument('--dataset_type', type=str, default='bluesky',
                        help='Type of dataset the model was trained on (bluesky, reddit, imdb)')
    parser.add_argument('--use_ollama', action='store_true',
                        help='Use Ollama model instead of local transformer')
    parser.add_argument('--ollama_model', type=str, default='llama2',
                        help='Ollama model to use (default: llama2)')
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
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Override the attention head count recorded in the checkpoint '
                             '(needed for checkpoints written before the config fix)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')

    args = parser.parse_args()

    if args.prompt:
        # Single generation mode
        model, tokenizer = load_model(
            args.model_path,
            dataset_type=args.dataset_type,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model,
            num_heads=args.num_heads
        )

        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            debug=args.debug,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model
        )

        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {args.prompt}{generated}")
    else:
        # Interactive demo mode
        run_interactive_demo(
            args.model_path,
            args.dataset_type,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model,
            num_heads=args.num_heads
        )
#!/usr/bin/env python3
"""
Optimize Inference Parameters

This script tests various combinations of inference parameters to find the optimal
settings for text generation quality.

Usage:
    python optimize_inference.py --prompt "Your prompt here" --metric ppl
"""

import argparse
import itertools
import torch
import numpy as np
from tqdm import tqdm
import time
import json
from inference import load_model, generate_text

# Optional: only import if available
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    print("NLTK not installed. BLEU score calculation will be disabled.")
    print("If you want to use BLEU scores, install NLTK: pip install nltk")
    HAS_NLTK = False

def log_perplexity(generated_text, model, tokenizer, device):
    """
    Calculate the perplexity of the generated text.
    Lower perplexity is better.
    """
    try:
        # Tokenize the generated text
        encoded = tokenizer.encode(generated_text)
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        
        # Create shifted labels (next token prediction)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last token prediction
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Calculate loss (CrossEntropy)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        
        # Handle output shapes
        if len(outputs.shape) == 3:  # [batch_size, seq_len, vocab_size]
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        else:
            raise ValueError(f"Unexpected output shape: {outputs.shape}")
        
        # Calculate perplexity
        loss = loss.mean().item()
        perplexity = np.exp(loss)
        return perplexity
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')  # Return infinity for failed calculations

def diversity_score(generated_text):
    """
    Calculate the lexical diversity of the generated text.
    Higher diversity is generally better (but not always).
    """
    if not generated_text or len(generated_text) < 2:
        return 0.0
        
    # Tokenize into words
    if HAS_NLTK:
        words = nltk.word_tokenize(generated_text.lower())
    else:
        words = generated_text.lower().split()
    
    # Calculate unique word ratio
    unique_words = set(words)
    if not words:
        return 0.0
    diversity = len(unique_words) / len(words)
    
    return diversity

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between reference and candidate texts.
    Higher is better.
    """
    if not HAS_NLTK:
        print("NLTK not installed. Cannot calculate BLEU score.")
        return 0.0
        
    # Tokenize texts
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Calculate BLEU score with smoothing
    smoothie = SmoothingFunction().method1
    try:
        score = sentence_bleu(reference_tokens, candidate_tokens, 
                             weights=(0.25, 0.25, 0.25, 0.25), 
                             smoothing_function=smoothie)
        return score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0

def evaluate_generation(prompt, generated_text, model, tokenizer, device, reference=None, verbose=False):
    """
    Evaluate the quality of generated text using multiple metrics.
    """
    # Length score - penalize very short outputs
    length = len(generated_text.split())
    length_score = min(1.0, length / 50.0) if length < 50 else 1.0
    
    # Diversity score - variety of words used
    div_score = diversity_score(generated_text)
    
    # Calculate perplexity if model available
    ppl = log_perplexity(prompt + generated_text, model, tokenizer, device)
    ppl_score = 1.0 / max(1.0, ppl / 100.0)  # Normalize and invert (lower ppl is better)
    
    # BLEU score if reference provided
    bleu = 0.0
    if reference:
        bleu = calculate_bleu(reference, generated_text)
    
    # Combined score (weighted)
    score = (
        0.3 * length_score +
        0.3 * div_score +
        0.4 * ppl_score +
        (0.3 * bleu if reference else 0.0)
    )
    
    if verbose:
        print(f"Length: {length} (score: {length_score:.3f})")
        print(f"Diversity: {div_score:.3f}")
        print(f"Perplexity: {ppl:.3f} (score: {ppl_score:.3f})")
        if reference:
            print(f"BLEU: {bleu:.3f}")
        print(f"Combined score: {score:.3f}")
    
    return {
        "score": score,
        "length": length,
        "diversity": div_score,
        "perplexity": ppl,
        "bleu": bleu if reference else None,
    }

def grid_search(prompt, model, tokenizer, params_grid, reference=None, num_samples=1):
    """
    Perform grid search over parameter combinations to find optimal settings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        params_grid['temperature'],
        params_grid['top_k'],
        params_grid['top_p'],
        params_grid['repetition_penalty']
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for temp, top_k, top_p, rep_penalty in tqdm(param_combinations):
        # Generate text with current parameters
        try:
            start_time = time.time()
            
            # Generate multiple samples for more robust evaluation
            sample_metrics = []
            for _ in range(num_samples):
                generated = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=params_grid['max_tokens'],
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_penalty
                )
                
                # Evaluate the generated text
                metrics = evaluate_generation(prompt, generated, model, tokenizer, device, reference)
                sample_metrics.append(metrics)
            
            # Average metrics across samples
            avg_metrics = {
                k: np.mean([m[k] for m in sample_metrics]) 
                for k in sample_metrics[0] if k != 'bleu' or reference
            }
            
            # Record generation time
            generation_time = (time.time() - start_time) / num_samples
            
            # Store results
            results.append({
                'params': {
                    'temperature': temp,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repetition_penalty': rep_penalty,
                    'max_tokens': params_grid['max_tokens']
                },
                'metrics': avg_metrics,
                'generation_time': generation_time,
                'generated_text': generated  # Save last generated sample
            })
            
        except Exception as e:
            print(f"Error with params (temp={temp}, top_k={top_k}, top_p={top_p}, rep_penalty={rep_penalty}): {e}")
    
    # Sort results by score
    results.sort(key=lambda x: x['metrics']['score'], reverse=True)
    return results

def main():
    parser = argparse.ArgumentParser(description='Optimize inference parameters for text generation')
    parser.add_argument('--prompt', type=str, default="The future of AI is",
                        help='Prompt to use for testing (default: "The future of AI is")')
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference text to compare generation against (for BLEU score)')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset_type', type=str, default='bluesky',
                        help='Type of dataset the model was trained on')
    parser.add_argument('--metric', type=str, default='combined', choices=['combined', 'ppl', 'diversity', 'length', 'bleu'],
                        help='Metric to optimize for')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to generate for each parameter combination')
    parser.add_argument('--output', type=str, default='inference_params.json',
                        help='Output file to save best parameters')
    parser.add_argument('--temperature_min', type=float, default=0.5, help='Minimum temperature')
    parser.add_argument('--temperature_max', type=float, default=1.0, help='Maximum temperature')
    parser.add_argument('--temperature_steps', type=int, default=3, help='Number of temperature steps')
    parser.add_argument('--top_k_min', type=int, default=10, help='Minimum top_k')
    parser.add_argument('--top_k_max', type=int, default=50, help='Maximum top_k')
    parser.add_argument('--top_k_steps', type=int, default=3, help='Number of top_k steps')
    parser.add_argument('--top_p_min', type=float, default=0.8, help='Minimum top_p')
    parser.add_argument('--top_p_max', type=float, default=0.95, help='Maximum top_p')
    parser.add_argument('--top_p_steps', type=int, default=3, help='Number of top_p steps')
    parser.add_argument('--rep_penalty_min', type=float, default=1.0, help='Minimum repetition penalty')
    parser.add_argument('--rep_penalty_max', type=float, default=1.5, help='Maximum repetition penalty')
    parser.add_argument('--rep_penalty_steps', type=int, default=3, help='Number of repetition penalty steps')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # Create parameter grid
    temps = np.linspace(args.temperature_min, args.temperature_max, args.temperature_steps)
    top_ks = np.linspace(args.top_k_min, args.top_k_max, args.top_k_steps, dtype=int)
    top_ps = np.linspace(args.top_p_min, args.top_p_max, args.top_p_steps)
    rep_penalties = np.linspace(args.rep_penalty_min, args.rep_penalty_max, args.rep_penalty_steps)
    
    params_grid = {
        'temperature': temps,
        'top_k': top_ks,
        'top_p': top_ps,
        'repetition_penalty': rep_penalties,
        'max_tokens': args.max_tokens
    }
    
    print("Parameter grid:")
    for param, values in params_grid.items():
        if param != 'max_tokens':
            print(f"  {param}: {list(values)}")
        else:
            print(f"  {param}: {values}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    try:
        # Monkey-patch the load_model function to disable weights_only
        from inference import load_model as original_load_model
        
        def patched_load_model(*args, **kwargs):
            """Patched version that disables weights_only for compatibility"""
            import types
            
            # Get the original function code
            original_code = original_load_model.__code__
            
            # Create a modified version of the function with weights_only=False
            def modified_function(*func_args, **func_kwargs):
                try:
                    import inspect
                    source = inspect.getsource(original_load_model)
                    # Replace weights_only=True with weights_only=False
                    patched_source = source.replace("weights_only=True", "weights_only=False")
                    
                    # Define the modified function in the local scope
                    local_dict = {}
                    exec(patched_source, globals(), local_dict)
                    
                    # Call the modified function
                    patched_func = local_dict['load_model']
                    return patched_func(*func_args, **func_kwargs)
                except Exception as e:
                    print(f"Error in monkey patching: {e}")
                    # Fallback to original with direct parameter override
                    return original_load_model(*func_args, **func_kwargs)
            
            return modified_function(*args, **kwargs)
        
        # Override the load_model function for this call
        model, tokenizer = patched_load_model(args.model_path, dataset_type=args.dataset_type)
    except Exception as e:
        print(f"Error with patched loading: {e}")
        print("Falling back to direct loading...")
        # Direct loading as fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        
        # Create model directly
        from coreTransformer import SmallTransformer
        from tokenizer import create_tokenizer
        
        tokenizer = create_tokenizer(tokenizer_type="byte-level-bpe", dataset_type=args.dataset_type)
        model = SmallTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Run grid search
    print(f"Running grid search with prompt: '{args.prompt}'")
    results = grid_search(
        args.prompt, 
        model, 
        tokenizer, 
        params_grid, 
        args.reference,
        args.num_samples
    )
    
    # Display top results
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5]):
        params = result['params']
        metrics = result['metrics']
        print(f"\n{i+1}. Score: {metrics['score']:.4f}")
        print(f"   Parameters: temp={params['temperature']:.2f}, top_k={params['top_k']}, "
              f"top_p={params['top_p']:.2f}, rep_penalty={params['repetition_penalty']:.2f}")
        print(f"   Perplexity: {metrics['perplexity']:.3f}")
        print(f"   Diversity: {metrics['diversity']:.3f}")
        print(f"   Generation time: {result['generation_time']:.3f}s")
        print(f"   Generated text: {args.prompt}{result['generated_text']}")
    
    # Save best parameters to file
    best_params = results[0]['params']
    
    # Convert numpy values to Python native types for JSON serialization
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_to_native_types(obj.tolist())
        else:
            return obj
    
    # Convert all values to native Python types
    output_data = convert_to_native_types({
        'prompt': args.prompt,
        'best_params': best_params,
        'metrics': results[0]['metrics'],
        'generated_text': results[0]['generated_text'],
        'all_results': results[:10]  # Save top 10 results
    })
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBest parameters saved to {args.output}")
    print(f"Best parameters: temp={best_params['temperature']:.2f}, top_k={best_params['top_k']}, "
          f"top_p={best_params['top_p']:.2f}, rep_penalty={best_params['repetition_penalty']:.2f}")
    
    # Generate a final sample with the best parameters
    print("\nGenerating final sample with best parameters...")
    generated = generate_text(
        model, 
        tokenizer, 
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=best_params['temperature'],
        top_k=best_params['top_k'],
        top_p=best_params['top_p'],
        repetition_penalty=best_params['repetition_penalty']
    )
    
    print(f"\nFinal generated text:")
    print(f"{args.prompt}{generated}")

if __name__ == "__main__":
    main()
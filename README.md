# Transformer2

A PyTorch implementation of a transformer model for natural language generation. This project provides a complete workflow for downloading/collecting training data, training custom transformer models, and running inference.

## Features

- Custom transformer architecture with Multi-Head Attention
- Data collection from multiple sources (Reddit, Bluesky)
- Curriculum learning with sequence length progression
- Mixed precision training with gradient accumulation
- Advanced text generation with temperature, top-k, and top-p sampling
- Complete training and inference pipeline

## Project Structure

- `coreTransformer.py` - Core transformer architecture
- `dataloader.py` - Dataset classes and data loading utilities
- `tokenizer.py` - Tokenization utilities
- `training.py` - Model training logic
- `inference.py` - Text generation utilities
- `data_collection.py` - Data collection from different sources
- `train_demo.py` - End-to-end training demo script

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```
   pip install torch transformers tokenizers datasets pandas tqdm
   ```
4. For Reddit data collection, also install:
   ```
   pip install praw
   ```

## Usage

### Data Collection

```bash
# Collect Bluesky data
python data_collection.py --dataset bluesky --max_samples 10000

# Collect Reddit data
python data_collection.py --dataset reddit --client_id YOUR_CLIENT_ID --client_secret YOUR_CLIENT_SECRET
```

### Training

```bash
# Standard training
python training.py --dataset bluesky --batch_size 8 --epochs 30

# Curriculum learning
python training.py --dataset reddit --batch_size 8 --epochs 30 --use_curriculum
```

### Inference

```bash
# Interactive demo
python inference.py

# Generate text from a prompt
python inference.py --prompt "Once upon a time" --temperature 0.7 --top_k 50
```

### End-to-end Demo

```bash
# Bluesky demo
python train_demo.py --dataset bluesky --max_samples 10000

# Reddit demo
python train_demo.py --dataset reddit --client_id YOUR_CLIENT_ID --client_secret YOUR_CLIENT_SECRET
```

## Model Configuration

Default configuration:
- Sequence length: 512
- Model dimension: 256
- Number of attention heads: 8
- Number of layers: 6
- Feed-forward dimension: 1024
- Dropout: 0.1

## License

MIT
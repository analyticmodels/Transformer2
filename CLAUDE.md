# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Train a model: `python trainingSetup.py --dataset [reddit|bluesky|imdb] --batch_size 8 --epochs 30`
- Train Reddit model: `python redditTrainingDemo.py`
- Train Bluesky model: `python blueskyTrainingDemo.py`
- Run inference test: `python inferenceTest.py`
- Demo script: `python blueskyTrainingDemo.py --skip_data_collection`

## Code Style Guidelines
- Imports order: standard library → third-party packages → local modules
- Class style: Follow Google docstring format with comprehensive documentation
- Naming: Descriptive variable/class names, CamelCase for classes, snake_case for functions/variables
- Error handling: Use try/except with specific error types and informative messages
- Model architecture: Proper PyTorch nn.Module subclassing with __init__ and forward methods
- PyTorch patterns: Use device management, gradient clipping, mixed precision training

## Project Structure
- Core model: `coreTransformer.py` (transformer architecture)
- Data handling: `dataloader.py` (dataset classes and loaders)
- Tokenization: `tokenizer.py` (tokenizer wrappers)
- Training: `trainingSetup.py`, `curriculumTraining.py`
- Inference: `inferenceTest.py`
# Create a conda environment
conda create -n transformer_dev python=3.11 -y
conda activate transformer_dev

# Install essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers wandb numpy pandas tqdm
# Check if environment exists
conda env list

# If you need to remove and recreate the environment
conda deactivate
conda remove --name transformer_dev --all
conda create -n transformer_dev python=3.10 -y
import torch

# Load the checkpoint
checkpoint = torch.load('/home/sean/repos/Transformer2/best_model.pt', map_location='cpu')

# Print the keys
print('Keys in checkpoint:', checkpoint.keys())

# Print the config
print('Config:', checkpoint.get('config', {}))

# Check for specific feed-forward dimension
ff_key = 'transformer_blocks.0.ff.0.bias'
if 'model_state_dict' in checkpoint and ff_key in checkpoint['model_state_dict']:
    print(f'FF shape: {checkpoint["model_state_dict"][ff_key].shape}')
    
# Also check the weight shape
ff_weight_key = 'transformer_blocks.0.ff.0.weight'
if 'model_state_dict' in checkpoint and ff_weight_key in checkpoint['model_state_dict']:
    print(f'FF weight shape: {checkpoint["model_state_dict"][ff_weight_key].shape}')
    
# Try to find d_ff value anywhere in checkpoint
if 'd_ff' in checkpoint.get('config', {}):
    print(f'Config d_ff: {checkpoint["config"]["d_ff"]}')
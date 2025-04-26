import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism with separate projections for Q, K, V.
    
    Args:
        d_model (int): Model dimension/hidden size
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        layer_idx (int): Index of this layer in the transformer
        num_layers (int): Total number of layers in the transformer
    """
    def __init__(self, d_model, num_heads, dropout, layer_idx, num_layers):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Single initialization for all projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer-specific dropout with depth-based scaling
        self.dropout = nn.Dropout(dropout * (layer_idx + 1) / num_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize all projection layers at once with scaled initialization
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, mean=0, std=0.02 / math.sqrt(2 * self.num_heads))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        B, S, D = x.shape
        
        # Separate Q, K, V projections
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.out_proj(context)

class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization architecture.
    
    This block implements the following structure:
    1. Layer normalization -> Multi-head attention -> Residual connection
    2. Layer normalization -> Feed-forward network (FFN) -> Residual connection
    
    The FFN uses GELU activation for better performance on language tasks compared to ReLU.
    Layer-specific dropout increases with depth for better regularization of later layers.
    
    Args:
        d_model (int): Model dimension/hidden size
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward network hidden dimension
        dropout (float): Dropout probability
        layer_idx (int): Index of this layer in the transformer
        num_layers (int): Total number of layers in the transformer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout, layer_idx, num_layers):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        
        # Layer normalization with improved epsilon
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Multi-head attention with scaled initialization
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            layer_idx=layer_idx,
            num_layers=num_layers
        )
        
        # Feed-forward network using GELU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU typically performs better than ReLU for language tasks
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer-specific dropout that increases with depth
        self.dropout = nn.Dropout(dropout * (layer_idx + 1) / num_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize feed-forward layers with scaled initialization
        for module in self.ff.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02 / math.sqrt(2 * self.num_layers))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Pre-norm architecture with controlled residual connections
        normed = self.norm1(x)
        x = x + self.dropout(self.attention(normed, attention_mask))
        
        normed = self.norm2(x)
        x = x + self.dropout(self.ff(normed))
        
        return x

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout):
        super(SmallTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Embed input tokens and add positional encoding
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = self.dropout(x)

        # Transpose attention_mask to match the expected shape [seq_len, batch_size]
        if attention_mask is not None:
            attention_mask = attention_mask.transpose(0, 1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None)

        # Project to vocabulary size
        logits = self.output_layer(x)  # Shape: [batch_size, seq_len, vocab_size]
        return logits
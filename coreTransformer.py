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
    """A transformer model for language modeling tasks with curriculum learning support.
    
    This implementation uses:
    - Scaled embedding with learned positional encodings
    - Pre-layer normalization architecture for stability
    - Progressive layer-wise dropout
    - Separate input and output projections
    
    Args:
        vocab_size (int): Size of the vocabulary
        d_model (int): Model dimension/hidden size
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        d_ff (int): Feed-forward network hidden dimension
        max_seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Token and position embeddings with proper scaling
        self.embedding_scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Input processing
        self.layer_norm_embed = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                layer_idx=i,
                num_layers=num_layers
            ) for i in range(num_layers)
        ])
        
        # Output processing
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_proj = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        # Initialize embedding layers
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize projection layers
        for module in [self.input_proj, self.output_proj, self.fc]:
            nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * self.num_layers))
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Input token ids of shape [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_len, vocab_size]
        """
        B, S = input_ids.shape
        assert S <= self.max_seq_len, f"Input sequence length {S} exceeds maximum {self.max_seq_len}"
        
        # Embeddings with proper scaling
        x = self.embedding(input_ids) * self.embedding_scale
        x = x + self.pos_embedding[:, :S, :]
        x = self.layer_norm_embed(x)
        x = self.dropout(x)
        x = self.input_proj(x)
        
        # Process attention mask once
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        return self.fc(x)
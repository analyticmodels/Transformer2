import math
import torch
import torch.nn as nn

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Increased model capacity and better initialization
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layer_norm_embed = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Add input projection layer
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Transformer blocks with improved architecture
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
        
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        # Add output projection before final layer
        self.output_proj = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        # Better weight initialization
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize projection layers
        for module in [self.input_proj, self.output_proj, self.fc]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        assert S <= self.max_seq_len, f"Input sequence length {S} exceeds maximum {self.max_seq_len}"
        
        # Embeddings with proper scaling
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :S, :]
        x = self.layer_norm_embed(x)
        x = self.dropout(x)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply transformer blocks with residual connections
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        return self.fc(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, layer_idx, num_layers):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        
        # Layer normalization with better epsilon
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
        
        # Improved feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer-specific dropout
        self.dropout = nn.Dropout(dropout * (layer_idx + 1) / num_layers)
        
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture with controlled residual connections
        attended = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attended)
        
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, layer_idx, num_layers):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Separate Q, K, V projections for better control
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer-specific dropout
        self.dropout = nn.Dropout(dropout * (layer_idx + 1) / num_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        # Scaled initialization for attention
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
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
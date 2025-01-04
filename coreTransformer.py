import torch
import torch.nn as nn
import math
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, 
                 d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers  # Store num_layers as instance variable
        
        # Scale embedding weights
        self.embedding_scale = math.sqrt(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe.unsqueeze(0))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize transformer blocks with layer normalization first
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        # Initialize final layer with small weights
        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedding with small normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize final layer with very small weights
        nn.init.normal_(self.fc.weight, mean=0, std=0.02/math.sqrt(2 * self.num_layers))
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask=None):
        # Embed and scale
        x = self.embedding(input_ids) * self.embedding_scale
        
        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        return self.fc(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Layer norm first (pre-norm) architecture
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self._init_weights()
        
    def _init_weights(self):
        # Initialize linear layers with small weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        normed = self.norm1(x)
        x = x + self.attention(normed, attention_mask)
        
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attention_mask is not None:
            # Reshape attention mask to match scores dimensions
            # attention_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to V
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, d_model)
        
        return self.proj(context)
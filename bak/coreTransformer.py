import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)
        
        return self.proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, 
                 d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Initialize embedding with padding token
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Create and register positional encoding buffer
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Input validation
        if input_ids.max() >= self.vocab_size:
            input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pe[:, :seq_len]
        x = self.dropout(x)
        
        # Process attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        return self.fc(x)
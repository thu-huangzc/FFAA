import torch
import torch.nn as nn
import torch.nn.functional as F
        
class SimpleResBlock(nn.Module):
    def __init__(self, input_dim):
        super(SimpleResBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return self.norm(x + self.mlp(x))

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        attn_output, _ = self.cross_attn(query=x, key=y, value=y)
        output = self.norm(attn_output)
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.cross_attn(query=x, key=x, value=x)
        # Residual Connection and Layer Normalization
        output = self.norm(attn_output + x)
        return output

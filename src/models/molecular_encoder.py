"""
Tunable Molecular Encoder for LNP Property Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TunedMolecularEncoder(nn.Module):
    """Tunable molecular encoder with proper dimension handling."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=64, 
                 num_heads=4, dropout=0.2):
        super().__init__()
        
        # Handle dimension compatibility
        self.num_heads = num_heads
        if input_dim % num_heads != 0:
            padding = num_heads - (input_dim % num_heads)
            self.input_projection = nn.Linear(input_dim, input_dim + padding)
            embed_dim = input_dim + padding
        else:
            self.input_projection = nn.Identity()
            embed_dim = input_dim
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-forward layers with residual connections
        self.layers = nn.ModuleList()
        dims = [embed_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.input_projection(x)
        
        # Self-attention with residual
        x_expanded = x.unsqueeze(1)
        attn_out, _ = self.attention(x_expanded, x_expanded, x_expanded)
        x = self.norm1(x + attn_out.squeeze(1))
        
        # Feed-forward with skip connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + 0.1 * residual  # Weighted residual connection
        
        return self.output_proj(x)

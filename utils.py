import torch
import torch.nn as nn
from torch.utils.data import dataloader, dataset, TensorDataset
import torch.nn.functional as F
import math

#My custom Positional Encodings
class Positional_Encoding(nn.Module):  # Fixed typo: "Postional" → "Positional"
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Masked_Attention(nn.Module):
    """Improved masked self-attention with efficient causal masking"""
    def __init__(self, num_heads, embedings, dropout=0.1):
        super().__init__()
        assert embedings % num_heads == 0, "embedings must be divisible by num_heads"

        self.num_heads = num_heads
        self.embedings = embedings
        self.heads = embedings // num_heads
        self.dropout = nn.Dropout(dropout)

        # QKV projections (keeping your original names)
        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.outlayer = nn.Linear(embedings, embedings)

        # Register causal mask as buffer (moved to device automatically)
        self.register_buffer('causal_mask', None)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed = x.size()

        # Project to Q, K, V (keeping your original variable names)
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)

        # Scaled dot-product attention
        first_term = torch.matmul(Q, K.transpose(-2, -1))
        second_term = math.sqrt(self.heads)
        scores = first_term / second_term

        # Apply padding mask if provided
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(padding_mask, float('-inf'))

        # Apply causal mask (prevents attending to future tokens)
        # FIXED: Create mask once and reuse, don't recreate every forward pass
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            # Create upper triangular matrix (1s above diagonal)
            self.causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()

        scores = scores.masked_fill(self.causal_mask, float('-inf'))

        # Softmax and dropout
        val = F.softmax(scores, dim=-1)
        val = self.dropout(val)

        # Apply attention to values
        output = torch.matmul(val, V)

        # Reshape back to [B, S, E]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)

        # Final linear projection
        out = self.outlayer(output)
        return out



# My Custom Postional Feed Forward Network
class Position_Feedforward(nn.Module):
    def __init__(self, input_ch, output_ch, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, output_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_ch, input_ch)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
         x= self.fc1(x)
         x = self.relu(x)
         x = self.dropout(x)
         x = self.fc2(x)
         x = self.dropout(x)
         return x
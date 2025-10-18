import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from utils import *
'''
This is my Custom Transformer Decoder Architeture code which has 'N' number layers
and can be used for the tasks like text generation
'''

class Multi_Decoder(nn.Module):
    def __init__(self, d_model, heads, hid_lay, dropout = 0.1):
        super().__init__()
        self.fc_layer = Position_Feedforward(d_model, hid_lay)
        self.Atten = Masked_Attention(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        atten_op = self.Atten(x)
        x = self.norm1(x+self.dropout(atten_op))

        layer_op = self.fc_layer(x)
        x = self.norm2(x + self.dropout(layer_op))
        return x

class Chatbot(nn.Module):
    def __init__(self, vocab_size, heads,d_model, hid_layer,seq_len, num_layers, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pos = Positional_Encoding(seq_len, d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.decoder_layers = nn.ModuleList([
            Multi_Decoder(d_model, heads, hid_layer, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        embed_op = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos(embed_op)

        for i in self.decoder_layers:
            x = i(x)
        x = self.final_norm(x)
        x = self.out_layer(x)
        return x
        
    

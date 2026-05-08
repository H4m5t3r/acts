import torch
import torch.nn as nn
import math

class SinCosPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=500):
        super().__init__()

        # Create a long enough matrix of positional encodings (1, max_len, d_model)
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # Copilot's cooking
        # Store as a plain tensor attribute on CPU (do NOT register as buffer).
        # register_buffer would cause the entire tensor to be moved on model.to().
        self.pe = pe

    def forward(self, x):
        seq_len = x.size(1)
        # Move only the slice we need to the input's device (cheap if seq_len small)
        pe_slice = self.pe[:, :seq_len, :].to(x.device)
        return x + pe_slice
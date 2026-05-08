import torch.nn as nn
from sin_cos_positional_encoding import SinCosPositionalEncoding

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, dim_feedforward, num_layers, output_dim, max_seq_len, dropout=0.1, mlp_width=1024):
        super().__init__()
        # Input embedding/projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = SinCosPositionalEncoding(d_model=model_dim, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # (batch_size, seq_len, model_dim)
            # batch_first=False # (seq_len, batch_size, model_dim)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.act = nn.LeakyReLU(negative_slope=0.1) # default 0.01
        # self.mlp_dropout = nn.Dropout(0.3)
        # self.mlp_dropout = nn.Dropout(0.1)
        self.mlp_dropout = nn.Dropout(0.0)
        # self.output_layer0 = nn.Linear(model_dim, 1024)
        # self.output_layer1 = nn.Linear(1024, 512)
        # self.output_layer2 = nn.Linear(512, output_dim)
        self.output_layer0 = nn.Linear(model_dim, mlp_width)
        self.output_layer1 = nn.Linear(mlp_width, int(mlp_width/2))
        self.output_layer2 = nn.Linear(int(mlp_width/2), output_dim)
        # self.mlp_dropout = nn.Dropout(0.1)
        # self.output_layer0 = nn.Linear(model_dim, 512)
        # self.output_layer1 = nn.Linear(512, 256)
        # self.output_layer2 = nn.Linear(256, output_dim)

        # self.single_output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # -> (batch_size, seq_len, model_dim)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        if mask is not None:
            inverted_mask = ~mask  # invert: True = keep
            inverted_mask = inverted_mask.unsqueeze(-1).float()
            x = (x * inverted_mask).sum(1) / inverted_mask.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)  # simple mean if no mask

        # Extra MLP layer to have something more than a linear layer
        output = self.output_layer2(
            self.mlp_dropout(self.act(self.output_layer1(
                self.mlp_dropout(self.act(self.output_layer0(x)))
            )))
        )
        return output


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, n_hidden_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class NoMLPTransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, dim_feedforward, num_layers, output_dim, max_seq_len, dropout=0.3):
        super().__init__()
        # Input embedding/projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = SinCosPositionalEncoding(d_model=model_dim, max_seq_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # (batch_size, seq_len, model_dim)
            # batch_first=False # (seq_len, batch_size, model_dim)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.single_output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # -> (batch_size, seq_len, model_dim)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=mask)

        if mask is not None:
            inverted_mask = ~mask  # invert: True = keep
            inverted_mask = inverted_mask.unsqueeze(-1).float()
            x = (x * inverted_mask).sum(1) / inverted_mask.sum(1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)  # simple mean if no mask

        output = self.single_output_projection(x)
        return output

def printModelSummary(model):
    print("Model summary:")
    print(model)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model type:", model.__class__.__name__)
    if hasattr(model, 'input_dim'):
        print("Input dimension:", model.input_dim)
    if hasattr(model, 'output_dim'):
        print("Output dimension:", model.output_dim)
    if hasattr(model, 'model_dim'):
        print("Model dimension:", model.model_dim)
    if hasattr(model, 'num_heads'):
        print("Number of heads:", model.num_heads)
    if hasattr(model, 'num_layers'):
        print("Number of layers:", model.num_layers)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.embed = nn.Embedding(args.embed_num, args.embed_dim)

        self.rnn_layer = nn.GRU(
            input_size=args.embed_dim,
            hidden_size=args.kernel_num,
            num_layers=2,
            bidirectional=False,
            dropout=args.dropout,
            batch_first=True
        )
        self.decoder = nn.Linear(args.kernel_num * 2, 1)

    def forward(self, x):
        x = self.embed(x)  
        _, hidden = self.rnn_layer(x)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        logit = self.decoder(hidden).squeeze(-1)

        return logit

import torch
import numpy as np
import torch.nn as nn
from model.utils.functions import PLU


class Encoder(nn.Module):
    def __init__(self, encode_dim, dropout):
        super(Encoder, self).__init__()
        self.in_dim = encode_dim[0]
        self.hidden_dim = encode_dim[1]
        self.out_dim = encode_dim[2]

        self.fc0 = nn.Linear(self.in_dim, self.hidden_dim, bias=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.out_dim, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        return x

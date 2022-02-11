import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, lstm_dim, num_layer=1):
        super(LSTM, self).__init__()
        self.in_dim = lstm_dim[0]
        self.hidden_dim = lstm_dim[1]
        self.num_layer = num_layer
        self.rnn = nn.LSTM(self.in_dim, self.hidden_dim, self.num_layer)

    def init_hidden(self, batch_size):
        self.h = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()
        self.c = torch.zeros((self.num_layer, batch_size, self.hidden_dim)).cuda()

    def forward(self, x):
        x, (self.h, self.c) = self.rnn(x, (self.h, self.c))
        return x

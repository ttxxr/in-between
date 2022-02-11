import torch
from torch import nn
from model.utils.functions import PLU


class ShortMotionDiscriminator(nn.Module):
    def __init__(self, length=3, in_dim=128, hidden_dim=512, out_dim=1):
        super(ShortMotionDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length

        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size=self.length, bias=True)
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim // 2, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x


class LongMotionDiscriminator(nn.Module):
    def __init__(self, length=10, in_dim=128, hidden_dim=512, out_dim=1):
        super(LongMotionDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length

        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size=self.length, bias=True)
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim // 2, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x


class OrientationDiscriminator(nn.Module):
    def __init__(self, length=10, in_dim=128, hidden_dim=512, out_dim=1):
        super(OrientationDiscriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.length = length

        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size=self.length, bias=True)
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim // 2, out_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        x = self.fc2(x)
        return x
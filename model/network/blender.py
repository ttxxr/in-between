import torch
from torch import nn
from model.utils.functions import PLU


class Blender(nn.Module):
    def __init__(self, blender_dim, output_dim, dropout=.0):
        super(Blender, self).__init__()

        self.blender_dim = blender_dim
        self.output_dim = output_dim

        # all joints
        self.fc0 = nn.Linear(self.blender_dim, self.blender_dim//2, bias=True)
        self.fc1 = nn.Linear(self.blender_dim//2, self.blender_dim//4, bias=True)
        self.fc_output = nn.Linear(self.blender_dim//4, self.output_dim, bias=True)

        # per joint
        # self.fc_output = nn.Linear(self.blender_dim, self.output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)
        output = self.sigmoid(self.fc_output(x))
        # output = nn.Softmax(-1)(self.fc_output(x))

        return output

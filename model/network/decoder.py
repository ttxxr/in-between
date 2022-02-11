import torch
from torch import nn
from model.utils.functions import PLU


class Decoder(nn.Module):
    def __init__(self, decoder_dim, output_dim, contact_dim, dropout=.0):
        super(Decoder, self).__init__()

        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.contact_dim = contact_dim

        self.fc0 = nn.Linear(self.decoder_dim[0], self.decoder_dim[1], bias=True)
        self.fc1 = nn.Linear(self.decoder_dim[1], self.decoder_dim[2], bias=True)
        self.fc_output = nn.Linear(self.decoder_dim[2], self.output_dim, bias=True)
        self.fc_contact = nn.Linear(self.decoder_dim[2], self.contact_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc0(x)
        x = PLU(x)
        x = self.fc1(x)
        x = PLU(x)

        output = self.fc_output(x)
        output_contact = self.sigmoid(self.fc_contact(x))

        return output, output_contact

import torch
from torch import nn
from torch.nn.modules import Linear
from torch.nn import functional as F


class LuongAttention(nn.Module):
    def __init__(self, decoder_dims, encoder_dims):
        super(LuongAttention, self).__init__()
        self.linear = Linear(encoder_dims, decoder_dims)

    def forward(self, decoder_hidden, encoder_outputs):
        aligns = torch.bmm(self.linear(encoder_outputs), decoder_hidden.transpose(-1, -2))
        attn_score = F.softmax(aligns, dim=-1)

        return attn_score

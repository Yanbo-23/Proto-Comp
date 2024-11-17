import torch.nn as nn


class PartialCond(nn.Module):
    def __init__(self, inChannel, outChannel=512):
        super().__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel

    self.ln = nn.Linear(inChannel, outChannel)


def forward(self, x):
    return self.ln(x)

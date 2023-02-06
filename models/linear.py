from .base import LinearBase
import torch.nn as nn

class Linear(LinearBase):
    def __init__(self, in_channels, num_classes, architecture):
        super(Linear, self).__init__(in_channels, num_classes, architecture)
        self.fcs = self.create_fcs()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        x = x.view(-1, x.size(0))
        x = self.softmax(x)
        return x

    def create_fcs(self):
        fcs = []
        in_channel = self.in_channels
        for out_channel in self.architecture:
            fcs.append(nn.Linear(in_channel, out_channel))
            fcs.append(nn.ReLU())
            in_channel = out_channel
        fcs.append(nn.Linear(in_channel, self.num_classes))
        return nn.Sequential(*fcs)
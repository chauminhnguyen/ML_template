import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu', batch_norm=True, pool=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.pool:
            x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, residual_blocks):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.relu = nn.ReLU()
        self.residual_blocks = residual_blocks
        self.layers = []
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0) if in_channel != out_channel else None
        for _ in range(self.residual_blocks):
            self.layers += [CNNBlock(in_channel, out_channel, kernel_size, stride, padding, activation='relu', batch_norm=True, pool=False)]
            in_channel = out_channel

    def forward(self, x):
        resudual = x
        for layer in self.layers:
            x = layer(x)
        if self.conv1x1:
            resudual = self.conv1x1(resudual)
        x = self.relu(x + resudual)
        return x


class CNNBase(nn.Module):
    def __init__(self, in_channels, num_classes, architecture):
        super(CNNBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = architecture

    def forward(self, x):
        raise NotImplementedError

    def create_conv_layers(self, architecture):
        raise NotImplementedError
    
    def create_fcs(self):
        raise NotImplementedError


class RNNBase(nn.Module):
    def __init__(self, in_channels, num_classes, architecture):
        super(RNNBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = architecture

    def forward(self, x):
        raise NotImplementedError

    def create_rnn_layers(self, architecture):
        raise NotImplementedError
    
    def create_fcs(self):
        raise NotImplementedError
    
    def init_hidden(self, batch_size, seq_len, hidden_dim):
        raise NotImplementedError


class LinearBase(nn.Module):
    def __init__(self, in_channels, num_classes, architecture):
        super(LinearBase, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = architecture

    def forward(self, x):
        raise NotImplementedError

    def create_fcs(self):
        raise NotImplementedError
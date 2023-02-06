import torch
import torch.nn as nn
from .base import CNNBlock, CNNBase, ResidualBlock

class ResNet(CNNBase):
    def __init__(self, in_channels=3, num_classes=1000, 
            architecture=[(64, 2, 3), (128, 2, 4), (256, 2, 6), (512, 2, 3)]):
        super(ResNet, self).__init__(in_channels, num_classes, architecture)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = architecture
        self.conv_layers = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        conv1 = CNNBlock(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), activation='relu', batch_norm=True, pool=False)
        in_channels = 64
        for x in architecture:
            out_channels, num_blocks, num_repeats = x

            for _ in range(num_repeats):
                layers += [ResidualBlock(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), residual_blocks=num_blocks)]
                in_channels = out_channels

        return nn.Sequential(
            conv1,
            *layers
        )

    def create_fcs(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, self.num_classes)
        )
    
    
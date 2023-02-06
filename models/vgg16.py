import torch
import torch.nn as nn
from .base import CNNBlock, CNNBase

class VGG16(CNNBase):
    def __init__(self, in_channels=3, num_classes=1000, 
                architecture=[(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]):
        super(VGG16, self).__init__(in_channels, num_classes, architecture)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = architecture
        self.conv_layers = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs()
        

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        '''
        architecture: list of tuples (out_channels, num_repeats)
        
        Example:
        [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
        '''
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            out_channels, num_repeats = x

            for _ in range(num_repeats):
                layers += [CNNBlock(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), pool=False)]
                in_channels = out_channels

            layers += [CNNBlock(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))]
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    def create_fcs(self):
        return nn.Sequential(
                nn.Linear(512*7*7, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, self.num_classes)
            )
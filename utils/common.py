import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3), activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                        kernel_size=kernel)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = F.relu(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, kernel=(2,2)):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=kernel)

    def forward(self, x1, x2):
        x_temp = self.deconv(x1)
        x =  torch.cat([x_temp, x2], dim=1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self, in_channels, kernel, size, stride=1, activation=True):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, kernel_size=kernel, stride=stride, padding=size//2)
        self.bnorm = nn.BatchNorm2d(kernel)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        
        if self.activation:
            x = F.relu(x)
        
        return x

class deconvBlock(nn.Module):
    def __init__(self, in_channels, kernel, stride=1):
        super(deconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, kernel, stride)
    
    def forward(self, x1, x2):
        x_temp = self.up(x1)
        x = torch.cat([x_temp, x2], dim=1)
        return x
        
        

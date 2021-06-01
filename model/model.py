from model.config import N_CLASSES, START_FRAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.common import ConvBlock, DoubleConvBlock, ResidualBlock

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=N_CLASSES, start_fm=START_FRAME):
        super(UNet, self).__init__()
        # Input 1x128x128

        # Maxpool 
        self.pool = nn.MaxPool2d((2,2))

        # Transpose conv
        self.deconv_4  = nn.ConvTranspose2d(start_fm*16, start_fm*8, 2, 2)
        self.deconv_3  = nn.ConvTranspose2d(start_fm*8, start_fm*4, 2, 2)
        self.deconv_2  = nn.ConvTranspose2d(start_fm*4, start_fm*2, 2, 2)
        self.deconv_1  = nn.ConvTranspose2d(start_fm*2, start_fm, 2, 2)
        
        # Encoder 
        self.encoder_1 = DoubleConvBlock(in_channels, start_fm, kernel=3)
        self.encoder_2 = DoubleConvBlock(start_fm, start_fm*2, kernel=3)
        self.encoder_3 = DoubleConvBlock(start_fm*2, start_fm*4, kernel=3)
        self.encoder_4 = DoubleConvBlock(start_fm*4, start_fm*8, kernel=3)

        # Middle
        self.middle = DoubleConvBlock(start_fm*8, start_fm*16)
        
        # Decoder
        self.decoder_4 = DoubleConvBlock(start_fm*16, start_fm*8)
        self.decoder_3 = DoubleConvBlock(start_fm*8, start_fm*4)
        self.decoder_2 = DoubleConvBlock(start_fm*4, start_fm*2)
        self.decoder_1 = DoubleConvBlock(start_fm*2, start_fm)

        self.conv_last = nn.Conv2d(start_fm, n_classes, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.encoder_1(x)
        x     = self.pool(conv1)

        conv2 = self.encoder_2(x)
        x     = self.pool(conv2)

        conv3 = self.encoder_3(x)
        x     = self.pool(conv3)

        conv4 = self.encoder_4(x)
        x     = self.pool(conv4)

        # Middle
        x     = self.middle(x)

        # Decoder
        x     = self.deconv_4(x)
        x     = torch.cat([conv4, x], dim=1)
        x     = self.decoder_4(x)

        x     = self.deconv_3(x)
        x     = torch.cat([conv3, x], dim=1)
        x     = self.decoder_3(x)

        x     = self.deconv_2(x)
        x     = torch.cat([conv2, x], dim=1)
        x     = self.decoder_2(x)

        x     = self.deconv_1(x)
        x     = torch.cat([conv1, x], dim=1)
        x     = self.decoder_1(x)
        
        out   = self.conv_last(x)
        return out

class UNet_ResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=N_CLASSES, dropout=0.5, start_fm=START_FRAME):
        super(UNet_ResNet, self).__init__()

        # Encoder 
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, start_fm, 3, padding=(1,1)),
            ResidualBlock(start_fm),
            ResidualBlock(start_fm, batch_activation=True),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(dropout//2),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(start_fm, start_fm*2, 3, padding=(1,1)),
            ResidualBlock(start_fm*2),
            ResidualBlock(start_fm*2, batch_activation=True),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(dropout),
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(start_fm*2, start_fm*4, 3, padding=(1,1)),
            ResidualBlock(start_fm*4),
            ResidualBlock(start_fm*4, batch_activation=True),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(dropout),
        )
        
        self.encoder_4 = nn.Sequential(
            nn.Conv2d(start_fm*4, start_fm*8, 3, padding=(1,1)),
            ResidualBlock(start_fm*8),
            ResidualBlock(start_fm*8, batch_activation=True),
            nn.MaxPool2d((2,2)),
            nn.Dropout2d(dropout),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(start_fm*8, start_fm*16, 3, padding=3//2),
            ResidualBlock(start_fm*16),
            ResidualBlock(start_fm*16, batch_activation=True),
            nn.MaxPool2d((2,2))
        )
        
        # Transpose conv
        self.deconv_4  = nn.ConvTranspose2d(start_fm*16, start_fm*8, 2, 2)
        self.deconv_3  = nn.ConvTranspose2d(start_fm*8, start_fm*4, 2, 2)
        self.deconv_2  = nn.ConvTranspose2d(start_fm*4, start_fm*2, 2, 2)
        self.deconv_1  = nn.ConvTranspose2d(start_fm*2, start_fm, 2, 2)

        # Decoder 
        self.decoder_4 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(start_fm*16, start_fm*8, 3, padding=(1,1)),
            ResidualBlock(start_fm*8),
            ResidualBlock(start_fm*8, batch_activation=True),
        )

        self.decoder_3 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(start_fm*8, start_fm*4, 3, padding=(1,1)),
            ResidualBlock(start_fm*4),
            ResidualBlock(start_fm*4, batch_activation=True),
        )

        self.decoder_2 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(start_fm*4, start_fm*2, 3, padding=(1,1)),
            ResidualBlock(start_fm*2),
            ResidualBlock(start_fm*2, batch_activation=True),
        )

        self.decoder_1 = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(start_fm*2, start_fm, 3, padding=(1,1)),
            ResidualBlock(start_fm),
            ResidualBlock(start_fm, batch_activation=True),
            nn.ConvTranspose2d(start_fm, start_fm, 2, 2)
        )
            
        self.conv_last = nn.Conv2d(start_fm, n_classes, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.encoder_1(x)

        conv2 = self.encoder_2(conv1)

        conv3 = self.encoder_3(conv2)

        conv4 = self.encoder_4(conv3)

        # Middle
        x     = self.middle(conv4)

        # Decoder
        x     = self.deconv_4(x)
        x     = torch.cat([conv4, x], dim=1)
        x     = self.decoder_4(x)

        x     = self.deconv_3(x)
        x     = torch.cat([conv3, x], dim=1)
        x     = self.decoder_3(x)

        x     = self.deconv_2(x)
        x     = torch.cat([conv2, x], dim=1)
        x     = self.decoder_2(x)

        x     = self.deconv_1(x)
        x     = torch.cat([conv1, x], dim=1)
        x     = self.decoder_1(x)
        
        out   = (self.conv_last(x))
        return out
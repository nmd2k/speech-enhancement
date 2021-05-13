import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.common import ConvBlock, DeconvBlock

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Encoder Block 1
        self.encoder_1 = nn.Sequential(
            OrderedDict([
                ('conv1_1', ConvBlock(in_channels, 64)),
                ('conv1_2', ConvBlock(64, 64)),
                ('maxpool1', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 2
        self.encoder_2 = nn.Sequential(
            OrderedDict([
                ('enconv2_1', ConvBlock(64, 128)),
                ('enconv2_2', ConvBlock(128, 128)),
                ('maxpool2', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 3
        self.encoder_3 = nn.Sequential(
            OrderedDict([
                ('enconv3_1', ConvBlock(128, 256)),
                ('enconv3_2', ConvBlock(256, 256)),
                ('maxpool3', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 4
        self.encoder_4 = nn.Sequential(
            OrderedDict([
                ('enconv4_1', ConvBlock(256, 512)),
                ('enconv4_2', ConvBlock(512, 512)),
                ('maxpool4', nn.MaxPool2d((2, 2)))
            ])
        )

        # Encoder Block 5
        self.encoder_5 = nn.Sequential(
            OrderedDict([
                ('enconv5_1', ConvBlock(512, 1024)),
                ('enconv5_2', ConvBlock(1024, 1024))
            ])
        )

        self.upconv4 = nn.Sequential(OrderedDict([('upconv4', DeconvBlock(1024))]))

        # Decoder Block 4 
        self.decoder_4 = nn.Sequential(
            OrderedDict([
                ('deconv4_1', ConvBlock(1024, 512)),
                ('deconv4_2', ConvBlock(512, 512))
            ])
        )

        self.upconv3 = nn.Sequential(OrderedDict([('upconv3', DeconvBlock(512))]))

        # Decoder Block 3
        self.decoder_3 = nn.Sequential(
            OrderedDict([
                ('deconv3_1', ConvBlock(512, 256)),
                ('deconv3_2', ConvBlock(256, 256))
            ])
        )

        self.upconv2 = nn.Sequential(OrderedDict([('upconv2', DeconvBlock(256))]))

        # Decoder Block 2
        self.decoder_2 = nn.Sequential(
            OrderedDict([
                ('deconv2_1', ConvBlock(256, 128)),
                ('deconv2_2', ConvBlock(128, 128))
            ])
        )

        self.upconv1 = nn.Sequential(OrderedDict([('upconv1', DeconvBlock(128))]))

        # Decoder Block 1
        self.decoder_1 = nn.Sequential(
            OrderedDict([
                ('deconv1_1', ConvBlock(128, 64)),
                ('deconv1_2', ConvBlock(64, 64)),
                ('deconv1_3', ConvBlock(64, self.n_classes, (1,1), activation=False)),
            ])
        )

    def forward(self, x):
        # Encoder
        # 1-64
        encoder_1 = self.encoder_1(x)
        # 64-128
        encoder_2 = self.encoder_2(encoder_1)
        # 128-256
        encoder_3 = self.encoder_2(encoder_2)
        # 256-512
        encoder_4 = self.encoder_2(encoder_3)
        
        # 512-1024
        x = self.encoder_2(encoder_4)

        # Decoder
        # 1024-512
        x = self.upconv4(x, encoder_4)
        x = self.decoder_4(x)
        # 512-256
        x = self.upconv3(x, encoder_3)
        x = self.decoder_3(x)
        # 256-128
        x = self.upconv2(x, encoder_2)
        x = self.decoder_2(x)
        #128-64-2
        x = self.upconv1(x, encoder_1)
        x = self.decoder_1(x)

        return x

        
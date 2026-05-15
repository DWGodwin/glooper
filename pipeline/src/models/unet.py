"""Small U-Net for binary segmentation.

Architecture: 4 encoder blocks + bottleneck + 4 decoder blocks. Each block is
two 3x3 convs + BN + ReLU. Encoder downsamples by max-pool; decoder upsamples
by transposed conv and concatenates the matching skip connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = (base_channels * m for m in (1, 2, 4, 8, 16))

        self.enc1 = _conv_block(in_channels, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.enc4 = _conv_block(c3, c4)
        self.bottleneck = _conv_block(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = _conv_block(c5, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(c2, c1)

        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


MODEL_SIZE_TO_BASE_CHANNELS = {"small": 16, "medium": 32, "large": 64}


def build_unet(model_size: str = "small", in_channels: int = 3) -> UNet:
    """Construct a UNet from a model_size key (small | medium | large)."""
    if model_size not in MODEL_SIZE_TO_BASE_CHANNELS:
        raise ValueError(f"Unknown model_size '{model_size}'. Valid: {list(MODEL_SIZE_TO_BASE_CHANNELS)}")
    return UNet(in_channels=in_channels, base_channels=MODEL_SIZE_TO_BASE_CHANNELS[model_size])

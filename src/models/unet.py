import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class UNetConfig:
    n_freq_bins:          int   = 1025
    base_channels:        int   = 24
    depth:                int   = 4
    dropout:              float = 0.1
    pool_freq:            bool  = True
    use_grad_checkpoint:  bool  = False

    @property
    def channel_list(self) -> List[int]:
        return [self.base_channels * (2 ** i) for i in range(self.depth)]

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1, pool_freq: bool = True):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        pool_size = (2, 2) if pool_freq else (1, 2)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        downsampled = self.pool(skip)
        return downsampled, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout)
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MagnitudeUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        ch = config.channel_list
        self.encoders = nn.ModuleList()
        self.encoders.append(EncoderBlock(1, ch[0], config.dropout, config.pool_freq))
        for i in range(1, config.depth):
            self.encoders.append(EncoderBlock(ch[i-1], ch[i], config.dropout, config.pool_freq))
        self.bottleneck = ConvBlock(ch[-1], ch[-1], config.dropout)
        self.decoders = nn.ModuleList()
        for i in range(config.depth - 1, 0, -1):
            self.decoders.append(DecoderBlock(ch[i], ch[i], ch[i-1], config.dropout))
        self.decoders.append(DecoderBlock(ch[0], ch[0], ch[0], config.dropout))
        self.output_conv = nn.Conv2d(ch[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        return self.sigmoid(self.output_conv(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

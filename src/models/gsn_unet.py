"""
src/models/gsn_unet.py
========================

GSN Phase 2 Model: U-Net with Harmonic GCN Bottleneck.

Architecture:
    Input [B, 1, F, T]
        ↓
    Encoder 1: ConvBlock + Pool  → [B, 32,  F/2,  T/2]
    Encoder 2: ConvBlock + Pool  → [B, 64,  F/4,  T/4]
    Encoder 3: ConvBlock + Pool  → [B, 128, F/8,  T/8]
    Encoder 4: ConvBlock + Pool  → [B, 256, F/16, T/16]
        ↓
    Bottleneck: HarmonicGCN      → [B, 256, F/16, T/16]
    (replaces plain ConvBlock)
        ↓
    Decoder 4: Upsample + skip   → [B, 128, F/8,  T/8]
    Decoder 3: Upsample + skip   → [B, 64,  F/4,  T/4]
    Decoder 2: Upsample + skip   → [B, 32,  F/2,  T/2]
    Decoder 1: Upsample + skip   → [B, 32,  F,    T]
        ↓
    Output conv + sigmoid        → [B, 1, F, T]  mask in [0, 1]

The only change from Phase 1:
    bottleneck = ConvBlock(256, 256)       ← Phase 1
    bottleneck = HarmonicGCNBottleneck()   ← Phase 2 (GSN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from src.models.unet import (
    UNetConfig,
    ConvBlock,
    EncoderBlock,
    DecoderBlock,
)
from src.models.harmonic_graph import build_harmonic_edges, get_edge_stats
from src.models.gcn_bottleneck import HarmonicGCNBottleneck


class GSNUNet(nn.Module):
    """
    Graph-Semantic-Net Phase 2:
    Magnitude U-Net with Harmonic GCN Bottleneck.

    Usage:
        model = GSNUNet(unet_config, sample_rate=44100, n_fft=2048)
        mask  = model(mixture_magnitude)   # [B, 1, F, T] → [B, 1, F, T]
    """

    def __init__(
        self,
        config: UNetConfig,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        max_harmonic: int = 5,
    ):
        super().__init__()

        self.config = config
        ch = config.channel_list   # e.g. [32, 64, 128, 256]

        # ------------------------------------------------
        # Encoder (same as Phase 1)
        # ------------------------------------------------
        self.encoders = nn.ModuleList()

        self.encoders.append(
            EncoderBlock(1, ch[0], config.dropout, config.pool_freq)
        )
        for i in range(1, config.depth):
            self.encoders.append(
                EncoderBlock(ch[i-1], ch[i], config.dropout, config.pool_freq)
            )

        # ------------------------------------------------
        # Harmonic GCN Bottleneck (NEW in Phase 2)
        # ------------------------------------------------

        # The bottleneck receives feature maps that have been pooled
        # pool_freq=True means both freq and time are halved each layer
        # After 4 encoder layers: F/16 frequency bins
        if config.pool_freq:
            bottleneck_freq_bins = config.n_freq_bins // (2 ** config.depth)
        else:
            bottleneck_freq_bins = config.n_freq_bins

        # Make sure we have at least a few bins to work with
        bottleneck_freq_bins = max(bottleneck_freq_bins, 4)

        print(f"Bottleneck freq bins: {bottleneck_freq_bins}")

        # Build harmonic edges for the bottleneck scale
        # At bottleneck, freq resolution is coarser so we use n_fft scaled
        bottleneck_n_fft = n_fft // (2 ** config.depth) if config.pool_freq else n_fft

        edge_index = build_harmonic_edges(
            n_freq_bins=bottleneck_freq_bins,
            sample_rate=sample_rate,
            n_fft=max(bottleneck_n_fft, 64),
            max_harmonic=max_harmonic,
        )

        # Print graph stats
        stats = get_edge_stats(edge_index, bottleneck_freq_bins)
        print(f"Harmonic graph stats:")
        print(f"  Nodes (freq bins) : {stats['num_nodes']}")
        print(f"  Edges             : {stats['num_edges']}")
        print(f"  Avg degree        : {stats['avg_degree']:.1f}")
        print(f"  Isolated nodes    : {stats['isolated_nodes']}")

        # The actual GCN bottleneck
        self.bottleneck = HarmonicGCNBottleneck(
            channels=ch[-1],
            edge_index=edge_index,
            dropout=config.dropout,
        )

        # ------------------------------------------------
        # Decoder (same as Phase 1)
        # ------------------------------------------------
        self.decoders = nn.ModuleList()

        for i in range(config.depth - 1, 0, -1):
            self.decoders.append(
                DecoderBlock(
                    in_channels=ch[i],
                    skip_channels=ch[i],
                    out_channels=ch[i-1],
                    dropout=config.dropout,
                )
            )

        self.decoders.append(
            DecoderBlock(
                in_channels=ch[0],
                skip_channels=ch[0],
                out_channels=ch[0],
                dropout=config.dropout,
            )
        )

        # ------------------------------------------------
        # Output head (same as Phase 1)
        # ------------------------------------------------
        self.output_conv = nn.Conv2d(ch[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        print(f"GSNUNet ready | params: {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mixture magnitude [B, 1, F, T]

        Returns:
            mask: [B, 1, F, T]  values in [0, 1]
        """

        assert x.dim() == 4 and x.shape[1] == 1, \
            f"Expected [B, 1, F, T], got {x.shape}"

        # Encode
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Harmonic GCN bottleneck
        x = self.bottleneck(x)

        # Decode
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output mask
        x = self.output_conv(x)
        x = self.sigmoid(x)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def separate(
        self,
        mixture_magnitude: torch.Tensor,
        mixture_phase: torch.Tensor,
        processor,
        original_length: int,
    ) -> torch.Tensor:
        """Full pipeline: magnitude + phase → separated audio [B, C, T]"""

        B, C, F, T = mixture_magnitude.shape

        masks = []
        for ch in range(C):
            ch_mask = self(mixture_magnitude[:, ch:ch+1])
            masks.append(ch_mask)

        mask = torch.cat(masks, dim=1)
        separated_mag = mixture_magnitude * mask

        stft = processor.magnitude_phase_to_complex(separated_mag, mixture_phase)
        return processor.istft(stft, length=original_length)
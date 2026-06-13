"""
src/models/gsn_phase3.py
=========================


The only new part vs Phase 2:
    - CLAP text encoder added
    - Cross-attention block added after GCN
    - Model now takes text_prompt as optional input
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Optional, List

from src.models.unet import (
    UNetConfig,
    EncoderBlock,
    DecoderBlock,
)
from src.models.harmonic_graph import build_harmonic_edges, get_edge_stats
from src.models.gcn_bottleneck import HarmonicGCNBottleneck
from src.models.cross_attention import SemanticCrossAttention
from src.models.clap_encoder import CLAPTextEncoder, DEFAULT_PROMPTS


class GSNPhase3(nn.Module):
    """
    Graph-Semantic-Net Phase 3.
    Text-guided audio source separation.

    Usage:
        model = GSNPhase3(config)

        # With text prompt (semantic steering ON)
        mask = model(mixture_mag, text_prompt="singing voice")

        # Without text prompt (falls back to Phase 2 behavior)
        mask = model(mixture_mag)

        # Switch target at inference time
        vocals_mask = model(mixture_mag, text_prompt="singing voice")
        drums_mask  = model(mixture_mag, text_prompt="drums and percussion")
        bass_mask   = model(mixture_mag, text_prompt="bass guitar")
    """

    def __init__(
        self,
        config: UNetConfig,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        max_harmonic: int = 5,
        text_dim: int = 512,
        num_attention_heads: int = 4,
        clap_model_name: str = "laion/clap-htsat-fused",
        clap_cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        ch = config.channel_list

        # ---- Encoder (same as Phase 2) ----
        self.encoders = nn.ModuleList()
        self.encoders.append(
            EncoderBlock(1, ch[0], config.dropout, config.pool_freq)
        )
        for i in range(1, config.depth):
            self.encoders.append(
                EncoderBlock(ch[i-1], ch[i], config.dropout, config.pool_freq)
            )

        # ---- Harmonic GCN Bottleneck (same as Phase 2) ----
        if config.pool_freq:
            bottleneck_freq_bins = max(
                config.n_freq_bins // (2 ** config.depth), 4
            )
        else:
            bottleneck_freq_bins = config.n_freq_bins

        bottleneck_n_fft = max(
            n_fft // (2 ** config.depth) if config.pool_freq else n_fft,
            64,
        )

        edge_index = build_harmonic_edges(
            n_freq_bins=bottleneck_freq_bins,
            sample_rate=sample_rate,
            n_fft=bottleneck_n_fft,
            max_harmonic=max_harmonic,
        )

        self.bottleneck = HarmonicGCNBottleneck(
            channels=ch[-1],
            edge_index=edge_index,
            dropout=config.dropout,
        )

        # ---- NEW: Cross-Attention for semantic steering ----
        # audio_dim must be divisible by num_attention_heads
        audio_dim = ch[-1]

        # Adjust heads if needed
        while audio_dim % num_attention_heads != 0 and num_attention_heads > 1:
            num_attention_heads -= 1

        self.cross_attention = SemanticCrossAttention(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_heads=num_attention_heads,
            dropout=config.dropout,
        )

        # ---- NEW: CLAP text encoder (frozen) ----
        self.text_encoder = CLAPTextEncoder(
            model_name=clap_model_name,
            embedding_dim=text_dim,
            cache_dir=clap_cache_dir,
        )

        # ---- Decoder (same as Phase 2) ----
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

        # ---- Output head (same as Phase 2) ----
        self.output_conv = nn.Conv2d(ch[0], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        print(f"GSN Phase 3 ready")
        print(f"  Total params      : {self.count_parameters():,}")
        print(f"  Attention heads   : {num_attention_heads}")
        print(f"  Text dim          : {text_dim}")

    def forward(
        self,
        x: torch.Tensor,
        text_prompt: Optional[str] = None,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : mixture magnitude [B, 1, F, T]
            text_prompt    : e.g. "singing voice" (optional)
            text_embedding : pre-computed [B, text_dim] (optional, faster)

        Returns:
            mask : [B, 1, F, T]  in [0, 1]
        """

        assert x.dim() == 4 and x.shape[1] == 1, (
            f"Expected [B, 1, F, T], got {x.shape}"
        )

        B = x.shape[0]
        device = x.device

        # ---- Get text embedding ----
        if text_embedding is None and text_prompt is not None:
            single_emb = self.text_encoder.encode(text_prompt, device)
            text_embedding = single_emb.expand(B, -1)   # [B, text_dim]

        # ---- Encode ----
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # ---- Harmonic GCN bottleneck ----
        x = self.bottleneck(x)
        # x shape: [B, C, freq_bins, T]

        # ---- Semantic cross-attention ----
        if text_embedding is not None:
            B, C, F, T = x.shape

            # Reshape for attention: [B, C, F, T] -> [B*T, F, C]
            x_for_attn = x.permute(0, 3, 2, 1)      # [B, T, F, C]
            x_for_attn = x_for_attn.reshape(B * T, F, C)

            # Expand text embedding for each time frame
            text_expanded = text_embedding.unsqueeze(1).expand(-1, T, -1)
            text_expanded = text_expanded.reshape(B * T, -1)   # [B*T, text_dim]

            # Apply cross-attention
            x_steered = self.cross_attention(x_for_attn, text_expanded)

            # Reshape back: [B*T, F, C] -> [B, C, F, T]
            x_steered = x_steered.reshape(B, T, F, C)
            x = x_steered.permute(0, 3, 2, 1)

        # ---- Decode ----
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # ---- Output ----
        x = self.output_conv(x)
        x = self.sigmoid(x)

        return x

    def separate(
        self,
        mixture_magnitude: torch.Tensor,
        mixture_phase: torch.Tensor,
        processor,
        original_length: int,
        text_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Full pipeline: magnitude + phase + text → audio [B, C, T]"""

        B, C, F, T = mixture_magnitude.shape
        device = mixture_magnitude.device

        # Pre-compute text embedding once for all channels
        text_emb = None
        if text_prompt is not None:
            single_emb = self.text_encoder.encode(text_prompt, device)
            text_emb = single_emb.expand(B, -1)

        masks = []
        for ch in range(C):
            ch_mask = self(
                mixture_magnitude[:, ch:ch+1],
                text_embedding=text_emb,
            )
            masks.append(ch_mask)

        mask = torch.cat(masks, dim=1)
        separated_mag = mixture_magnitude * mask
        stft = processor.magnitude_phase_to_complex(separated_mag, mixture_phase)
        return processor.istft(stft, length=original_length)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self) -> dict:
        components = {}
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            components[name] = params
        return components
"""
src/models/gsn_complex.py
=========================
GSN Phase 4: Full Complex-Valued Text-Guided Separator.
"""
import torch
import torch.nn as nn
from src.models.gsn_phase3 import GSNPhase3

class GSNComplex(GSNPhase3):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Change first layer: 1 channel (mag) -> 2 channels (real/imag)
        ch0 = config.channel_list[0]
        self.encoders[0].conv.block[0] = nn.Conv2d(2, ch0, kernel_size=3, padding=1, bias=False)

        # Change output layer: 1 channel (mag mask) -> 2 channels (complex mask)
        self.output_conv = nn.Conv2d(ch0, 2, kernel_size=1)
        # We use Tanh for complex masks because they can be negative
        self.activation = nn.Tanh()

    def forward(self, x, text_embedding=None):
        # x: [B, 2, F, T] (Real, Imag)
        # The rest of the U-Net logic is inherited from GSNPhase3
        # We just override the final activation

        # ... logic from GSNPhase3 forward ...
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        if text_embedding is not None:
            B, C, F, T = x.shape
            x_for_attn = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
            text_expanded = text_embedding.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            x = self.cross_attention(x_for_attn, text_expanded).reshape(B, T, F, C).permute(0, 3, 2, 1)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        x = self.output_conv(x)
        return self.activation(x) # [B, 2, F, T]
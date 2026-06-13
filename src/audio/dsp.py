
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class STFTConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1

class STFTProcessor(nn.Module):
    def __init__(self, config: STFTConfig):
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.win_length))

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # audio: [B, C, T]
        B, C, T = audio.shape
        flat = audio.reshape(-1, T)
        spec = torch.stft(
            flat, self.config.n_fft, self.config.hop_length,
            self.config.win_length, self.window,
            center=True, return_complex=True
        )
        _, F, Tf = spec.shape
        spec = spec.reshape(B, C, F, Tf)
        return spec.abs(), torch.angle(spec), T

    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        # spec: [B, C, F, Tf] (complex)
        B, C, F, Tf = spec.shape
        flat = spec.reshape(-1, F, Tf)
        wav = torch.istft(
            flat, self.config.n_fft, self.config.hop_length,
            self.config.win_length, self.window,
            center=True, length=length, return_complex=False
        )
        return wav.reshape(B, C, -1)

    def magnitude_phase_to_complex(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

class AudioAugmenter(nn.Module):
    def __init__(self, gain_range=(0.7, 1.3), swap_prob=0.5, seed=42):
        super().__init__()
        self.gain_range = gain_range
        self.swap_prob = swap_prob
        import random
        self.rng = random.Random(seed)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, C, T] or [C, T]
        if audio.ndim == 2:
             audio = audio.unsqueeze(0)

        B, C, T = audio.shape

        # Gain augmentation
        gain = self.gain_range[0] + self.rng.random() * (self.gain_range[1] - self.gain_range[0])
        audio = audio * gain

        # Channel swap
        if C == 2 and self.rng.random() < self.swap_prob:
            audio = audio[:, [1, 0], :]

        return audio.squeeze(0) if B == 1 else audio

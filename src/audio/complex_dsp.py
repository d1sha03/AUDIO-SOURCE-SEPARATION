"""
src/audio/complex_dsp.py
========================
Handles Complex STFT processing for Phase 4.
Instead of Magnitude, we keep Real and Imaginary parts.
"""
import torch
import torch.nn as nn

class ComplexSTFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def transform(self, audio):
        # audio: [B, C, T]
        B, C, T = audio.shape
        # Flatten batch and channels for stft
        flat_audio = audio.reshape(-1, T)

        spec = torch.stft(
            flat_audio, self.n_fft, self.hop_length,
            window=self.window, center=True, return_complex=True
        ) # [B*C, F, Tf]

        # Reshape back to [B, C, F, Tf]
        _, F, Tf = spec.shape
        spec = spec.reshape(B, C, F, Tf)

        # Stack Real and Imaginary as two new channels
        # [B, C, F, Tf] -> [B, C*2, F, Tf]
        # For stereo (C=2), we get 4 channels: [L_real, L_imag, R_real, R_imag]
        complex_spec = torch.stack([spec.real, spec.imag], dim=2) # [B, C, 2, F, Tf]
        complex_spec = complex_spec.reshape(B, C*2, F, Tf)
        return complex_spec

    def inverse(self, complex_spec, length=None):
        # complex_spec: [B, C*2, F, Tf]
        B, C2, F, Tf = complex_spec.shape
        C = C2 // 2

        # Unstack Real and Imaginary
        spec_reshaped = complex_spec.reshape(B, C, 2, F, Tf)
        real = spec_reshaped[:, :, 0]
        imag = spec_reshaped[:, :, 1]

        complex_tensor = torch.complex(real, imag) # [B, C, F, Tf]
        flat_complex = complex_tensor.reshape(-1, F, Tf)

        audio = torch.istft(
            flat_complex, self.n_fft, self.hop_length,
            window=self.window, center=True, length=length
        )
        return audio.reshape(B, C, -1)

    def apply_complex_mask(self, mix_spec, mask):
        """
        mix_spec: [B, 4, F, T] (L_re, L_im, R_re, R_im)
        mask:     [B, 4, F, T] (M_re, M_im, M_re, M_im)

        Complex multiplication: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
        """
        B, _, F, T = mix_spec.shape
        m = mix_spec.reshape(B, 2, 2, F, T) # [B, Stereo, RI, F, T]
        mask = mask.reshape(B, 2, 2, F, T)

        mix_re = m[:, :, 0]
        mix_im = m[:, :, 1]
        mask_re = mask[:, :, 0]
        mask_im = mask[:, :, 1]

        res_re = mix_re * mask_re - mix_im * mask_im
        res_im = mix_re * mask_im + mix_im * mask_re

        out = torch.stack([res_re, res_im], dim=2) # [B, 2, 2, F, T]
        return out.reshape(B, 4, F, T)
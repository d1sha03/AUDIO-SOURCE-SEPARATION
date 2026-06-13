"""
src/training/losses.py
=======================
Loss functions for audio source separation.

WHAT IS A LOSS FUNCTION?
-------------------------
It measures "how wrong" the model is.
During training, PyTorch automatically adjusts the model's
weights to make the loss smaller.

Loss = 0.0  →  perfect prediction
Loss = big  →  terrible prediction

LOSSES WE USE IN PHASE 1:
--------------------------
1. L1 Loss (Mean Absolute Error)
   - Simple: average absolute difference between prediction and target
   - MAE(pred, target) = mean(|pred - target|)
   - Good for spectrograms: directly penalises wrong magnitudes

2. SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
   - Measures audio quality in dB
   - Higher is better (positive = good, negative = bad)
   - "Scale-invariant" means volume differences don't hurt the score
   - Used as a METRIC (we report this to know how good we are)
   - Also used as a loss: loss = -SI_SNR (minimise negative = maximise SI-SNR)

PHASE 4 will add:
   - Spectral Convergence Loss
   - Log Magnitude Loss
   - Combined multi-domain loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# =============================================================================
# L1 MAGNITUDE LOSS
# =============================================================================

class L1MagnitudeLoss(nn.Module):
    """
    L1 loss on magnitude spectrograms.

    Simple and effective for Phase 1.
    Compares predicted magnitude vs target magnitude directly.

    loss = mean( |predicted_magnitude - target_magnitude| )
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_magnitude:   torch.Tensor,   # [B, C, F, T]
        target_magnitude: torch.Tensor,   # [B, C, F, T]
    ) -> torch.Tensor:
        assert pred_magnitude.shape == target_magnitude.shape, (
            f"Shape mismatch: {pred_magnitude.shape} vs {target_magnitude.shape}"
        )
        return F.l1_loss(pred_magnitude, target_magnitude)


# =============================================================================
# SI-SNR LOSS
# =============================================================================

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.

    SI-SNR measures separation quality in decibels (dB).

    HOW IT WORKS:
    1. Project the estimate onto the target  (signal part)
    2. Compute the noise = estimate - signal_part
    3. SI-SNR = 10 * log10( ||signal||² / ||noise||² )

    Higher SI-SNR = better separation.
    As a LOSS we return -SI-SNR (so minimising loss = maximising quality).

    Typical values:
        < 0 dB   : worse than random
        0-5 dB   : poor (but audible separation)
        5-10 dB  : decent separation
        > 10 dB  : good separation
        > 15 dB  : excellent
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        estimate: torch.Tensor,   # [B, C, T]  predicted audio
        target:   torch.Tensor,   # [B, C, T]  ground truth audio
    ) -> torch.Tensor:
        """
        Returns -SI-SNR (negative, because we want to MINIMISE loss).
        """
        assert estimate.shape == target.shape, (
            f"Shape mismatch: {estimate.shape} vs {target.shape}"
        )

        # Flatten channels and time: [B, C, T] → [B, C*T]
        B, C, T   = estimate.shape
        est_flat  = estimate.reshape(B, -1)   # [B, C*T]
        tgt_flat  = target.reshape(B, -1)     # [B, C*T]

        # 1. Zero-mean (remove DC offset)
        est_flat  = est_flat - est_flat.mean(dim=1, keepdim=True)
        tgt_flat  = tgt_flat - tgt_flat.mean(dim=1, keepdim=True)

        # 2. Scale-invariant projection
        # s_target = <est, tgt> / <tgt, tgt> * tgt
        dot       = (est_flat * tgt_flat).sum(dim=1, keepdim=True)          # [B, 1]
        tgt_power = (tgt_flat * tgt_flat).sum(dim=1, keepdim=True) + self.eps
        s_target  = dot / tgt_power * tgt_flat                              # [B, C*T]

        # 3. Noise = residual
        e_noise   = est_flat - s_target                                      # [B, C*T]

        # 4. SI-SNR in dB
        signal_power = (s_target * s_target).sum(dim=1) + self.eps          # [B]
        noise_power  = (e_noise  * e_noise ).sum(dim=1) + self.eps          # [B]
        sisnr        = 10 * torch.log10(signal_power / noise_power)         # [B]

        # Return mean over batch, negated (loss to minimise)
        return -sisnr.mean()

    def compute_metric(
        self,
        estimate: torch.Tensor,
        target:   torch.Tensor,
    ) -> float:
        """
        Returns SI-SNR in dB as a Python float (positive = good).
        Use this for logging/display.
        """
        with torch.no_grad():
            return -self.forward(estimate, target).item()


# =============================================================================
# SI-SDR METRIC  (for validation reporting)
# =============================================================================

def compute_si_sdr(
    estimate: torch.Tensor,   # [B, C, T]
    target:   torch.Tensor,   # [B, C, T]
    eps:      float = 1e-8,
) -> float:
    """
    Compute SI-SDR (dB) between estimate and target.
    Returns a Python float (mean over batch).

    SI-SDR and SI-SNR are equivalent formulas;
    we use SI-SDR as the standard evaluation metric.
    """
    with torch.no_grad():
        B, C, T  = estimate.shape
        est      = estimate.reshape(B, -1)
        tgt      = target.reshape(B, -1)

        est      = est - est.mean(dim=1, keepdim=True)
        tgt      = tgt - tgt.mean(dim=1, keepdim=True)

        dot      = (est * tgt).sum(dim=1, keepdim=True)
        tgt_pow  = (tgt * tgt).sum(dim=1, keepdim=True) + eps
        s_tgt    = dot / tgt_pow * tgt

        noise    = est - s_tgt
        si_sdr   = 10 * torch.log10(
            (s_tgt * s_tgt).sum(dim=1) / ((noise * noise).sum(dim=1) + eps)
        )
        return si_sdr.mean().item()


# =============================================================================
# COMBINED PHASE 1 LOSS
# =============================================================================

class Phase1Loss(nn.Module):
    """
    Combined loss for Phase 1 training.

    L_total = w_l1 * L1(pred_mag, target_mag)
            + w_sisnr * (-SI-SNR(pred_audio, target_audio))

    Why combine them?
    - L1 on magnitude: easy to optimise, guides early training
    - SI-SNR on audio: directly measures perceived quality

    Default weights: 70% L1, 30% SI-SNR
    (SI-SNR weight increases in Phase 4)
    """

    def __init__(self, w_l1: float = 0.7, w_sisnr: float = 0.3):
        super().__init__()
        self.w_l1    = w_l1
        self.w_sisnr = w_sisnr
        self.l1_loss    = L1MagnitudeLoss()
        self.sisnr_loss = SISNRLoss()

    def forward(
        self,
        pred_magnitude:   torch.Tensor,   # [B, C, F, T]
        target_magnitude: torch.Tensor,   # [B, C, F, T]
        pred_audio:       torch.Tensor,   # [B, C, T]
        target_audio:     torch.Tensor,   # [B, C, T]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict so we can log each component separately.
        {
            "loss"    : total weighted loss  (use this for .backward())
            "l1"      : L1 magnitude loss
            "sisnr"   : SI-SNR loss (negative dB)
        }
        """
        l1    = self.l1_loss(pred_magnitude, target_magnitude)
        sisnr = self.sisnr_loss(pred_audio, target_audio)
        total = self.w_l1 * l1 + self.w_sisnr * sisnr

        return {
            "loss":   total,
            "l1":     l1,
            "sisnr":  sisnr,
        }
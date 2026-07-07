"""
src/training/trainer.py
========================
Training loop for Phase 1 U-Net.

WHAT HAPPENS EACH TRAINING STEP:
  1. Load batch: (mixture_audio, target_audio)  from DataLoader
  2. Compute STFT: audio → magnitude + phase
  3. Forward pass: mixture_magnitude → predicted mask
  4. Apply mask: predicted_vocals_magnitude = mixture_magnitude * mask
  5. Compute loss: compare predicted vs target magnitudes + audio
  6. Backward pass: compute gradients
  7. Optimiser step: update model weights
  8. Repeat!

WHAT IS AN EPOCH?
  One full pass through the training dataset.
  e.g., 1000 songs × 10 chunks each = 10,000 items
        with batch_size=4: 2,500 steps per epoch

WHAT IS A CHECKPOINT?
  A saved copy of the model weights at a point in time.
  We save the best checkpoint (lowest validation loss).
  If training crashes, we can resume from the last checkpoint.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    """All training hyperparameters."""

    # Optimiser
    learning_rate: float = 1e-3      # How big each weight update is
    weight_decay:  float = 1e-4      # L2 regularisation (prevents overfitting)

    # Scheduler: reduce LR when validation loss stops improving
    lr_patience:  int   = 5          # Wait this many epochs before reducing LR
    lr_factor:    float = 0.5        # Multiply LR by this when reducing
    min_lr:       float = 1e-6       # Never go below this LR

    # Training duration
    max_epochs:   int   = 100
    grad_clip:    float = 5.0        # Clip gradients to prevent explosion

    # Saving
    checkpoint_dir: str = "checkpoints"
    save_every:     int = 5          # Save checkpoint every N epochs

    # Logging
    log_every:    int   = 10         # Print loss every N batches

    # Early stopping
    patience:     int   = 20         # Stop if no improvement for N epochs


# =============================================================================
# METRIC TRACKER
# =============================================================================

class MetricTracker:
    """
    Tracks running average of metrics during an epoch.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            loss = compute_loss(batch)
            tracker.update({"loss": loss.item(), "si_sdr": si_sdr})
        print(tracker.averages())
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if k not in self._sums:
                self._sums[k]   = 0.0
                self._counts[k] = 0
            self._sums[k]   += float(v)
            self._counts[k] += 1

    def averages(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def average(self, key: str) -> float:
        if key not in self._sums or self._counts[key] == 0:
            return float("nan")
        return self._sums[key] / self._counts[key]


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    Manages the full training loop for Phase 1.

    Usage:
        trainer = Trainer(model, processor, loss_fn, config, device)
        trainer.fit(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model:      nn.Module,
        processor,                  # STFTProcessor
        loss_fn:    nn.Module,
        config:     TrainerConfig,
        device:     torch.device,
    ):
        self.model     = model.to(device)
        self.processor = processor.to(device)
        self.loss_fn   = loss_fn
        self.config    = config
        self.device    = device

        # Optimiser: Adam is the standard choice
        self.optimiser = Adam(
            model.parameters(),
            lr           = config.learning_rate,
            weight_decay = config.weight_decay,
        )

        # Scheduler: reduce LR when stuck
        self.scheduler = ReduceLROnPlateau(
            self.optimiser,
            mode      = "min",           # reduce when val_loss stops decreasing
            patience  = config.lr_patience,
            factor    = config.lr_factor,
            min_lr    = config.min_lr,
        )

        # State tracking
        self.best_val_loss    = float("inf")
        self.epochs_no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_si_sdr": [], "val_si_sdr": [],
            "lr": [],
        }

        # Make checkpoint dir
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       Optional[int] = None,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Args:
            train_loader : training DataLoader
            val_loader   : validation DataLoader
            epochs       : override max_epochs if provided

        Returns:
            history dict with loss curves
        """
        max_epochs = epochs or self.config.max_epochs

        print("="*60)
        print("PHASE 1 TRAINING START")
        print("="*60)
        print(f"  Model params : {self.model.count_parameters():,}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches  : {len(val_loader)}")
        print(f"  Max epochs   : {max_epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Device       : {self.device}")
        print("="*60)

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # ── Train ───────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validate ────────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader)

            elapsed = time.time() - t0

            # ── Scheduler step ──────────────────────────────────────
            self.scheduler.step(val_metrics["loss"])
            current_lr = self.optimiser.param_groups[0]["lr"]

            # ── Log ─────────────────────────────────────────────────
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_si_sdr"].append(train_metrics.get("si_sdr", 0))
            self.history["val_si_sdr"].append(val_metrics.get("si_sdr", 0))
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:03d}/{max_epochs} "
                f"| train_loss={train_metrics['loss']:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} "
                f"| val_SI-SDR={val_metrics.get('si_sdr', 0):+.2f}dB "
                f"| lr={current_lr:.1e} "
                f"| {elapsed:.1f}s"
            )

            # ── Save best checkpoint ─────────────────────────────────
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss     = val_metrics["loss"]
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ★ New best! val_loss={self.best_val_loss:.4f}")
            else:
                self.epochs_no_improve += 1

            # ── Periodic checkpoint ──────────────────────────────────
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # ── Early stopping ───────────────────────────────────────
            if self.epochs_no_improve >= self.config.patience:
                print(f"\
Early stopping at epoch {epoch} "
                      f"(no improvement for {self.config.patience} epochs)")
                break

        print("\
Training complete!")
        print(f"Best val_loss : {self.best_val_loss:.4f}")
        return self.history

    # ------------------------------------------------------------------
    def _train_epoch(
        self,
        loader: DataLoader,
        epoch:  int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()

        for batch_idx, (mix_audio, tgt_audio) in enumerate(loader):
            # Move to device
            mix_audio = mix_audio.to(self.device)   # [B, C, T]
            tgt_audio = tgt_audio.to(self.device)   # [B, C, T]

            # ── STFT ──────────────────────────────────────────────
            with torch.no_grad():   # STFT has no learnable params
                mix_mag, mix_phase, orig_len = self.processor(mix_audio)
                tgt_mag, _,         _        = self.processor(tgt_audio)
            # mix_mag: [B, C, F, T]

            # ── Forward pass ──────────────────────────────────────
            # Process each channel: [B, 1, F, T] → mask → [B, 1, F, T]
            B, C, F, T = mix_mag.shape
            pred_masks = []
            for ch in range(C):
                ch_mag  = mix_mag[:, ch:ch+1, :, :]   # [B, 1, F, T]
                ch_mask = self.model(ch_mag)            # [B, 1, F, T]
                pred_masks.append(ch_mask)

            pred_mask = torch.cat(pred_masks, dim=1)   # [B, C, F, T]

            # Apply mask to get predicted magnitude
            pred_mag  = mix_mag * pred_mask             # [B, C, F, T]

            # Reconstruct audio for SI-SNR loss
            pred_stft  = self.processor.magnitude_phase_to_complex(pred_mag, mix_phase)
            pred_audio = self.processor.istft(pred_stft, length=orig_len)   # [B, C, T]

            # ── Loss ──────────────────────────────────────────────
            loss_dict = self.loss_fn(pred_mag, tgt_mag, pred_audio, tgt_audio)
            loss      = loss_dict["loss"]

            # ── Backward ──────────────────────────────────────────
            self.optimiser.zero_grad()
            loss.backward()

            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

            self.optimiser.step()

            # ── Metrics ───────────────────────────────────────────
            from src.training.losses import compute_si_sdr
            si_sdr = compute_si_sdr(pred_audio.detach(), tgt_audio)

            tracker.update({
                "loss":   loss.item(),
                "l1":     loss_dict["l1"].item(),
                "sisnr":  loss_dict["sisnr"].item(),
                "si_sdr": si_sdr,
            })

            # ── Log ───────────────────────────────────────────────
            if batch_idx % self.config.log_every == 0:
                avgs = tracker.averages()
                print(
                    f"  [{epoch}] batch {batch_idx:04d}/{len(loader)} "
                    f"loss={avgs['loss']:.4f} "
                    f"SI-SDR={avgs['si_sdr']:+.2f}dB",
                    end="\\r"
                )

        print()  # newline after \\r
        return tracker.averages()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch (no gradients)."""
        self.model.eval()
        tracker = MetricTracker()

        for mix_audio, tgt_audio in loader:
            mix_audio = mix_audio.to(self.device)
            tgt_audio = tgt_audio.to(self.device)

            mix_mag, mix_phase, orig_len = self.processor(mix_audio)
            tgt_mag, _,         _        = self.processor(tgt_audio)

            B, C, F, T = mix_mag.shape
            pred_masks = []
            for ch in range(C):
                ch_mask = self.model(mix_mag[:, ch:ch+1, :, :])
                pred_masks.append(ch_mask)

            pred_mask  = torch.cat(pred_masks, dim=1)
            pred_mag   = mix_mag * pred_mask

            pred_stft  = self.processor.magnitude_phase_to_complex(pred_mag, mix_phase)
            pred_audio = self.processor.istft(pred_stft, length=orig_len)

            loss_dict  = self.loss_fn(pred_mag, tgt_mag, pred_audio, tgt_audio)

            from src.training.losses import compute_si_sdr
            si_sdr = compute_si_sdr(pred_audio, tgt_audio)

            tracker.update({
                "loss":   loss_dict["loss"].item(),
                "l1":     loss_dict["l1"].item(),
                "sisnr":  loss_dict["sisnr"].item(),
                "si_sdr": si_sdr,
            })

        return tracker.averages()

    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        epoch:      int,
        metrics:    Dict[str, float],
        is_best:    bool,
    ) -> None:
        """Save model weights + training state."""
        state = {
            "epoch":         epoch,
            "model_state":   self.model.state_dict(),
            "optim_state":   self.optimiser.state_dict(),
            "metrics":       metrics,
            "best_val_loss": self.best_val_loss,
            "history":       self.history,
        }

        if is_best:
            path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        else:
            path = os.path.join(
                self.config.checkpoint_dir, f"epoch_{epoch:03d}.pt"
            )

        torch.save(state, path)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint to resume training.

        Args:
            path : path to .pt file

        Returns:
            epoch number where training left off
        """
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state"])
        self.optimiser.load_state_dict(state["optim_state"])
        self.best_val_loss = state["best_val_loss"]
        self.history       = state.get("history", self.history)

        epoch = state["epoch"]
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        print(f"  Best val_loss so far: {self.best_val_loss:.4f}")
        return epoch
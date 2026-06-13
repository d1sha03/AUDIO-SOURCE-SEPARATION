"""
src/models/clap_encoder.py
===========================

CLAP Text Encoder for semantic steering.

CLAP = Contrastive Language-Audio Pretraining
  - Pretrained model that understands both audio and text
  - We use ONLY the text encoder part
  - Weights are FROZEN (we do not train them)
  - Given a text prompt, outputs a 512-dim embedding vector

Why freeze CLAP?
  - CLAP was trained on millions of audio-text pairs
  - It already knows what "vocals", "drums", "guitar" mean
  - Fine-tuning it would destroy this knowledge
  - We just use it as a fixed feature extractor

The embedding vector captures the MEANING of the text.
  "singing voice" and "vocal melody" → similar vectors
  "drums" and "percussion" → similar vectors
  "singing voice" and "drums" → very different vectors

This is what lets the model switch targets based on text.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import os


# Common text prompts for source separation
# We predefine these so users get consistent results
STEM_PROMPTS = {
    "vocals": [
        "singing voice",
        "lead vocals",
        "human voice singing",
        "vocal melody",
    ],
    "drums": [
        "drums and percussion",
        "drum kit",
        "rhythmic percussion",
        "kick snare and hi-hat",
    ],
    "bass": [
        "bass guitar",
        "low frequency bass",
        "bass line",
        "electric bass",
    ],
    "other": [
        "guitar and keyboards",
        "melodic instruments",
        "accompaniment without vocals bass and drums",
        "harmonic instruments",
    ],
}

# Use the first prompt as the default for each stem
DEFAULT_PROMPTS = {stem: prompts[0] for stem, prompts in STEM_PROMPTS.items()}


class CLAPTextEncoder(nn.Module):
    """
    Frozen CLAP text encoder.

    Takes a text string and returns a fixed-size embedding vector.
    The weights never change during training.

    Usage:
        encoder = CLAPTextEncoder()
        embedding = encoder.encode("singing voice")
        # embedding shape: [1, 512]
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        embedding_dim: int = 512,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.cache_dir = cache_dir

        self._model = None
        self._processor = None
        self._loaded = False

        # Cache for text embeddings
        # So we do not recompute the same prompt every batch
        self._embedding_cache: Dict[str, torch.Tensor] = {}

        print(f"CLAPTextEncoder initialized")
        print(f"  Model    : {model_name}")
        print(f"  Emb dim  : {embedding_dim}")
        print(f"  Note     : weights are FROZEN, loaded on first use")

    def _load_model(self, device: torch.device) -> None:
        """Load CLAP model on first use (lazy loading)."""
        if self._loaded:
            return

        print(f"Loading CLAP model: {self.model_name}")
        print("  (This downloads ~1GB on first run, then cached locally)")

        try:
            from transformers import ClapModel, ClapProcessor

            self._processor = ClapProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self._model = ClapModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            # Freeze ALL CLAP weights
            for param in self._model.parameters():
                param.requires_grad = False

            self._model = self._model.to(device)
            self._model.eval()

            self._loaded = True
            print(f"  ✓ CLAP loaded and frozen on {device}")

        except Exception as e:
            print(f"  ✗ CLAP load failed: {e}")
            print("  Falling back to random embeddings for testing")
            self._loaded = False

    @torch.no_grad()
    def encode(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a text prompt into a fixed-size embedding.

        Args:
            text   : text prompt e.g. "singing voice"
            device : which device to put the output on

        Returns:
            embedding : [1, embedding_dim] tensor
        """
        if device is None:
            device = torch.device("cpu")

        # Check cache first
        cache_key = f"{text}_{device}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Try to load and use CLAP
        if not self._loaded:
            self._load_model(device)

        if self._loaded and self._model is not None:
            try:
                inputs = self._processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                text_features = self._model.get_text_features(**inputs)
                # Normalize to unit sphere (standard for CLAP)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embedding = text_features.detach()

            except Exception as e:
                print(f"CLAP encode failed: {e}, using fallback")
                embedding = self._fallback_embedding(text, device)
        else:
            embedding = self._fallback_embedding(text, device)

        # Make sure embedding has the right size
        if embedding.shape[-1] != self.embedding_dim:
            # Project to correct size if needed
            embedding = embedding[:, :self.embedding_dim]

        # Cache it
        self._embedding_cache[cache_key] = embedding

        return embedding

    def _fallback_embedding(self, text: str, device: torch.device) -> torch.Tensor:
        """
        Deterministic fallback when CLAP is not available.
        Uses a hash of the text to create a consistent random embedding.
        Different texts get different embeddings.
        """
        seed = hash(text) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        embedding = torch.randn(1, self.embedding_dim, generator=gen)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.to(device)

    def encode_batch(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode a list of text prompts.

        Args:
            texts  : list of text prompts, one per batch item
            device : target device

        Returns:
            embeddings : [batch, embedding_dim]
        """
        embeddings = [self.encode(t, device) for t in texts]
        return torch.cat(embeddings, dim=0)

    def forward(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.encode(text, device)

    def clear_cache(self) -> None:
        self._embedding_cache.clear()
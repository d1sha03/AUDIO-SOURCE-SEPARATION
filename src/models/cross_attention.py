from typing import Optional
"""
src/models/cross_attention.py
==============================

Cross-attention between GCN frequency nodes and CLAP text embedding.

How cross-attention works here:

  Query  = frequency node features (what the audio model sees)
  Key    = text embedding (what the user asked for)
  Value  = text embedding (what to inject into the audio features)

In plain English:
  Each frequency node "asks" the text embedding:
  "Should I keep this frequency for the requested source?"

  The text embedding answers based on what it knows about
  the requested source (vocals, drums, bass, etc.)

  This steers the separation toward the requested source.

Why cross-attention and not just concatenation?
  - Concatenation treats all frequency bins the same
  - Cross-attention lets each frequency bin attend to the text
    differently based on its content
  - A 440 Hz bin (vocal fundamental) should attend very
    strongly to "singing voice"
  - A 60 Hz bin (bass fundamental) should attend less
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import math


class SemanticCrossAttention(nn.Module):
    """
    Cross-attention that injects text semantics into audio features.

    Input:
        audio_features : [B, num_nodes, audio_dim]  (from GCN)
        text_embedding : [B, text_dim]               (from CLAP)

    Output:
        steered_features : [B, num_nodes, audio_dim]  (same shape)

    The text embedding is used as Key and Value.
    The audio features are used as Query.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.head_dim = audio_dim // num_heads

        assert audio_dim % num_heads == 0, (
            f"audio_dim ({audio_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Project audio features to Query
        self.query_proj = nn.Linear(audio_dim, audio_dim, bias=False)

        # Project text embedding to Key and Value
        # Text dim (512) may differ from audio dim (256)
        self.key_proj   = nn.Linear(text_dim, audio_dim, bias=False)
        self.value_proj = nn.Linear(text_dim, audio_dim, bias=False)

        # Output projection
        self.output_proj = nn.Linear(audio_dim, audio_dim, bias=False)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(audio_dim)

        self.dropout = nn.Dropout(dropout)

        # Scale factor for dot-product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features : [B, num_nodes, audio_dim]
            text_embedding : [B, text_dim]

        Returns:
            output : [B, num_nodes, audio_dim]
        """

        B, N, D = audio_features.shape

        # Expand text embedding to [B, 1, text_dim]
        # The "1" means there is one text token
        text = text_embedding.unsqueeze(1)   # [B, 1, text_dim]

        # Project to Q, K, V
        Q = self.query_proj(audio_features)  # [B, N, audio_dim]
        K = self.key_proj(text)              # [B, 1, audio_dim]
        V = self.value_proj(text)            # [B, 1, audio_dim]

        # Reshape for multi-head attention
        # [B, N, audio_dim] -> [B, num_heads, N, head_dim]
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Q: [B, heads, N, head_dim]
        # K: [B, heads, 1, head_dim]
        # scores: [B, heads, N, 1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = func.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention weights to values
        # weights: [B, heads, N, 1]
        # V:       [B, heads, 1, head_dim]
        # output:  [B, heads, N, head_dim]
        attended = torch.matmul(weights, V)

        # Merge heads back
        # [B, heads, N, head_dim] -> [B, N, audio_dim]
        attended = attended.transpose(1, 2).reshape(B, N, D)

        # Output projection
        attended = self.output_proj(attended)

        # Residual connection + layer norm
        output = self.layer_norm(audio_features + attended)

        return output   # [B, N, audio_dim]


class SemanticInjectionBlock(nn.Module):
    """
    Full semantic injection block for the GSN bottleneck.

    This wraps:
    1. The Harmonic GCN (processes audio graph)
    2. Cross-attention (injects text semantics)

    So the flow is:
        audio features
            → GCN (harmonic reasoning)
            → Cross-attention (text steering)
            → output

    Can be used with or without text prompt.
    When no text is provided, behaves like plain GCN.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attention = SemanticCrossAttention(
            audio_dim=audio_dim,
            text_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        audio_features: torch.Tensor,     # [B, num_nodes, audio_dim]
        text_embedding: Optional[torch.Tensor] = None,  # [B, text_dim]
    ) -> torch.Tensor:
        """
        Args:
            audio_features : [B, num_nodes, audio_dim]
            text_embedding : [B, text_dim] or None

        Returns:
            output : [B, num_nodes, audio_dim]
        """

        if text_embedding is None:
            return audio_features

        return self.cross_attention(audio_features, text_embedding)


# Need to import Optional
from typing import Optional
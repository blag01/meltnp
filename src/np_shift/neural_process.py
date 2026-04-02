from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NeuralProcessOutput:
    mean: Tensor
    variance: Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """One transformer block: self-attn + MLP with pre-norm and residual connections."""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention + MLP with pre-norm and residual connections."""

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        # Cross-attention with residual
        h = self.norm_q(query)
        kv = self.norm_kv(context)
        h, _ = self.cross_attn(h, kv, kv, need_weights=False)
        query = query + h
        # MLP with residual
        query = query + self.mlp(self.norm2(query))
        return query


class AttentionNeuralProcess(nn.Module):
    """Transformer Neural Process with stacked self-attention and cross-attention."""

    def __init__(
        self,
        x_dim: int = 1,
        y_dim: int = 1,
        hidden_dim: int = 128,
        representation_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.representation_dim = representation_dim

        # Input embeddings
        self.context_embed = MLP(x_dim + y_dim, hidden_dim, representation_dim)
        self.target_embed = MLP(x_dim, hidden_dim, representation_dim)

        # Stacked transformer layers
        mlp_dim = representation_dim * 2
        self.context_self_attn_layers = nn.ModuleList([
            TransformerLayer(representation_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(representation_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        # Final norm + decoder
        self.final_norm = nn.LayerNorm(representation_dim)
        self.decoder = MLP(representation_dim + x_dim, hidden_dim, 2 * y_dim)

    def forward(
        self,
        context_x: Tensor,
        context_y: Tensor,
        target_x: Tensor,
        latent_value_shift: Tensor | None = None,
        context_weights: Tensor | None = None,
    ) -> NeuralProcessOutput:
        """Run stacked self-attention on context, then cross-attention from target.

        Args:
            latent_value_shift: additive offset to context embeddings (for TTA).
            context_weights: multiplicative scaling of context embeddings (for TTA).
        """
        if context_x.ndim != 3 or context_y.ndim != 3 or target_x.ndim != 3:
            raise ValueError("expected tensors shaped [batch, points, features]")
        if context_x.shape[:2] != context_y.shape[:2]:
            raise ValueError("context_x and context_y must align on batch and point dimensions")
        if context_x.size(-1) != self.x_dim or target_x.size(-1) != self.x_dim:
            raise ValueError("x tensors have wrong feature dimension")
        if context_y.size(-1) != self.y_dim:
            raise ValueError("context_y has wrong feature dimension")

        # Embed inputs
        ctx = self.context_embed(torch.cat([context_x, context_y], dim=-1))
        tgt = self.target_embed(target_x)

        # TTA hooks (applied before the transformer stack)
        if latent_value_shift is not None:
            ctx = ctx + latent_value_shift
        if context_weights is not None:
            ctx = ctx * context_weights

        # Stacked layers: context self-attention, then target cross-attention
        for self_attn, cross_attn in zip(self.context_self_attn_layers, self.cross_attn_layers):
            ctx = self_attn(ctx)
            tgt = cross_attn(tgt, ctx)

        # Decode
        tgt = self.final_norm(tgt)
        decoder_input = torch.cat([tgt, target_x], dim=-1)
        decoder_output = self.decoder(decoder_input)
        mean, raw_scale = torch.split(decoder_output, self.y_dim, dim=-1)
        variance = nn.functional.softplus(raw_scale) + 1e-4
        return NeuralProcessOutput(mean=mean, variance=variance)

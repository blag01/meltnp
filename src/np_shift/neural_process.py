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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionNeuralProcess(nn.Module):
    """A small attention-based Neural Process with learned embeddings."""

    def __init__(
        self,
        x_dim: int = 1,
        y_dim: int = 1,
        hidden_dim: int = 128,
        representation_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.representation_dim = representation_dim

        self.context_encoder = MLP(
            input_dim=x_dim + y_dim,
            hidden_dim=hidden_dim,
            output_dim=representation_dim,
        )
        self.query_encoder = MLP(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            output_dim=representation_dim,
        )
        self.key_encoder = MLP(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            output_dim=representation_dim,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=representation_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.decoder = MLP(
            input_dim=representation_dim + x_dim,
            hidden_dim=hidden_dim,
            output_dim=2 * y_dim,
        )

    def forward(
        self,
        context_x: Tensor,
        context_y: Tensor,
        target_x: Tensor,
    ) -> NeuralProcessOutput:
        if context_x.ndim != 3 or context_y.ndim != 3 or target_x.ndim != 3:
            raise ValueError("expected tensors shaped [batch, points, features]")
        if context_x.shape[:2] != context_y.shape[:2]:
            raise ValueError("context_x and context_y must align on batch and point dimensions")
        if context_x.size(-1) != self.x_dim or target_x.size(-1) != self.x_dim:
            raise ValueError("x tensors have wrong feature dimension")
        if context_y.size(-1) != self.y_dim:
            raise ValueError("context_y has wrong feature dimension")

        context_features = torch.cat([context_x, context_y], dim=-1)
        value_embeddings = self.context_encoder(context_features)
        key_embeddings = self.key_encoder(context_x)
        query_embeddings = self.query_encoder(target_x)

        attended, _ = self.cross_attention(
            query=query_embeddings,
            key=key_embeddings,
            value=value_embeddings,
            need_weights=False,
        )

        decoder_input = torch.cat([attended, target_x], dim=-1)
        decoder_output = self.decoder(decoder_input)
        mean, raw_scale = torch.split(decoder_output, self.y_dim, dim=-1)
        variance = torch.nn.functional.softplus(raw_scale) + 1e-4
        return NeuralProcessOutput(mean=mean, variance=variance)

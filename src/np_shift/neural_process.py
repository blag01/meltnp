from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NeuralProcessOutput:
    mean: Tensor
    variance: Tensor
    prior_mu: Tensor | None = None
    prior_log_sigma: Tensor | None = None
    posterior_mu: Tensor | None = None
    posterior_log_sigma: Tensor | None = None


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


class LatentEncoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.encoder = MLP(x_dim + y_dim, hidden_dim, hidden_dim)
        self.penultimate = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        xy = torch.cat([x, y], dim=-1)
        r_i = self.encoder(xy)
        r = r_i.mean(dim=1)
        r = torch.relu(self.penultimate(r))
        return self.mu(r), self.log_sigma(r)


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
        z_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.representation_dim = representation_dim

        # Input embeddings
        self.context_embed = MLP(x_dim + y_dim, hidden_dim, representation_dim)
        self.target_embed = MLP(x_dim, hidden_dim, representation_dim)

        # Latent Path (Optional)
        self.z_dim = z_dim
        if z_dim is not None:
            self.latent_encoder = LatentEncoder(x_dim, y_dim, hidden_dim, z_dim)
            self.z_proj = nn.Linear(z_dim, representation_dim)

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
        target_y: Tensor | None = None,
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
            
        prior_m, prior_s, post_m, post_s = None, None, None, None
        
        # Inject latent z if active
        if self.z_dim is not None:
            prior_m, prior_s = self.latent_encoder(context_x, context_y)
            if target_y is not None:
                full_x = torch.cat([context_x, target_x], dim=1)
                full_y = torch.cat([context_y, target_y], dim=1)
                post_m, post_s = self.latent_encoder(full_x, full_y)
                mu, log_sigma = post_m, post_s
            else:
                mu, log_sigma = prior_m, prior_s
                
            # Reparameterization trick
            std = torch.exp(log_sigma)
            z = mu + torch.randn_like(std) * std
            
            # Embed and prepend z to context sequence
            z_embedded = self.z_proj(z).unsqueeze(1)
            ctx = torch.cat([z_embedded, ctx], dim=1)

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
        return NeuralProcessOutput(
            mean=mean, variance=variance,
            prior_mu=prior_m, prior_log_sigma=prior_s,
            posterior_mu=post_m, posterior_log_sigma=post_s
        )

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass(frozen=True)
class NPBatch:
    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor | None = None  # Optional for inference
    corruption_label: str = "clean"


def add_gaussian_noise(y: Tensor, std: float = 0.5) -> Tensor:
    return y + torch.randn_like(y) * std


def apply_bias_shift(y: Tensor, shift_range: tuple[float, float] = (-2.0, 2.0)) -> Tensor:
    shift = torch.empty(y.size(0), 1, 1, device=y.device).uniform_(*shift_range)
    return y + shift


def rbf_kernel(x1: Tensor, x2: Tensor, length_scale: float = 0.5) -> Tensor:
    """Compute RBF kernel between x1 and x2 [B, N, D] and [B, M, D] -> [B, N, M]."""
    dist_sq = torch.cdist(x1, x2).pow(2)
    return torch.exp(-0.5 * dist_sq / length_scale**2)


class GPData:
    def __init__(
        self,
        batch_size: int = 16,
        num_context: int = 10,
        num_target: int = 20,
        x_range: tuple[float, float] = (-2.0, 2.0),
        length_scale: float = 0.5,
    ) -> None:
        self.batch_size = batch_size
        self.num_context = num_context
        self.num_target = num_target
        self.x_range = x_range
        self.length_scale = length_scale

    def generate_batch(self, corruption_fn: callable | None = None) -> NPBatch:
        total_points = self.num_context + self.num_target
        x = torch.empty(self.batch_size, total_points, 1).uniform_(*self.x_range)

        K = rbf_kernel(x, x, length_scale=self.length_scale)
        K = K + torch.eye(total_points).unsqueeze(0) * 1e-4
        L = torch.linalg.cholesky(K)
        y = torch.matmul(L, torch.randn(self.batch_size, total_points, 1))

        context_x, target_x = x[:, : self.num_context, :], x[:, self.num_context :, :]
        context_y, target_y = y[:, : self.num_context, :], y[:, self.num_context :, :]

        if corruption_fn is not None:
            context_y = corruption_fn(context_y)

        return NPBatch(context_x, context_y, target_x, target_y)


class SinusoidData:
    def __init__(
        self,
        batch_size: int = 16,
        num_context: int = 10,
        num_target: int = 20,
        x_range: tuple[float, float] = (-5.0, 5.0),
        amp_range: tuple[float, float] = (0.1, 1.0),
        phase_range: tuple[float, float] = (0.0, torch.pi),
        freq_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        self.batch_size = batch_size
        self.num_context = num_context
        self.num_target = num_target
        self.x_range = x_range
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.freq_range = freq_range

    def generate_batch(self, corruption_fn: callable | None = None) -> NPBatch:
        total_points = self.num_context + self.num_target
        x = torch.empty(self.batch_size, total_points, 1).uniform_(*self.x_range)

        # Sample task parameters once per batch
        amp = torch.empty(self.batch_size, 1, 1).uniform_(*self.amp_range)
        phase = torch.empty(self.batch_size, 1, 1).uniform_(*self.phase_range)
        freq = torch.empty(self.batch_size, 1, 1).uniform_(*self.freq_range)

        y = amp * torch.sin(freq * x + phase)

        context_x, target_x = x[:, : self.num_context, :], x[:, self.num_context :, :]
        context_y, target_y = y[:, : self.num_context, :], y[:, self.num_context :, :]

        if corruption_fn is not None:
            context_y = corruption_fn(context_y)

        return NPBatch(context_x, context_y, target_x, target_y)

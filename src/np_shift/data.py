from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable

@dataclass(frozen=True)
class NPBatch:
    """A single batch of Neural Process data (context + target)."""
    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor | None = None
    context_y_clean: Tensor | None = None
    corruption_label: str = "clean"


def add_gaussian_noise(x: Tensor, y: Tensor, std: float = 0.5) -> Tensor:
    """Add i.i.d. Gaussian noise to y."""
    return y + torch.randn_like(y) * std


def apply_bias_shift(x: Tensor, y: Tensor, shift_range: tuple[float, float] = (-2.0, 2.0)) -> Tensor:
    """Add a uniform constant bias per batch element."""
    shift = torch.empty(y.size(0), 1, 1, device=y.device).uniform_(*shift_range)
    return y + shift


def heteroskedastic_noise(x: Tensor, y: Tensor, scale_factor: float = 0.5) -> Tensor:
    """Noise magnitude increases proportionately with |x|."""
    noise_std = scale_factor * torch.abs(x)
    return y + torch.randn_like(y) * noise_std


def apply_warp_shift(x: Tensor, y: Tensor, warp_power: float = 3.0) -> Tensor:
    """Non-linear warping of the target signal (e.g., y^3)."""
    # Preserve sign for odd/even stability
    return torch.sign(y) * (torch.abs(y) ** warp_power)


def inject_outliers(x: Tensor, y: Tensor, fraction: float = 0.3, magnitude: float = 5.0) -> Tensor:
    """Corrupt a random fraction of context points with extreme values."""
    mask = torch.rand_like(y) < fraction
    outlier_vals = torch.randn_like(y) * magnitude
    return torch.where(mask, y + outlier_vals, y)


def rbf_kernel(x1: Tensor, x2: Tensor, length_scale: float = 0.5) -> Tensor:
    """Compute RBF kernel between x1 and x2 [B, N, D] and [B, M, D] -> [B, N, M]."""
    dist_sq = torch.cdist(x1, x2).pow(2)
    return torch.exp(-0.5 * dist_sq / length_scale**2)


class NPDataset(ABC):
    @abstractmethod
    def generate_batch(self, corruption_fn: Callable[[Tensor, Tensor], Tensor] | None = None) -> NPBatch:
        """Returns a batch of context and target points."""
        pass

class GPData(NPDataset):
    """Generates regression tasks from a GP with RBF kernel."""
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

    def generate_batch(self, corruption_fn: callable | None = None,
                       context_x_range: tuple[float, float] | None = None) -> NPBatch:
        # Sample target x from the standard range
        target_x = torch.empty(self.batch_size, self.num_target, 1).uniform_(*self.x_range)
        
        # Sample context x from a potentially shifted range
        cx_range = context_x_range if context_x_range is not None else self.x_range
        context_x = torch.empty(self.batch_size, self.num_context, 1).uniform_(*cx_range)
        
        # Evaluate the GP over all points jointly for consistency
        x = torch.cat([context_x, target_x], dim=1)
        total_points = self.num_context + self.num_target

        K = rbf_kernel(x, x, length_scale=self.length_scale)
        K = K + torch.eye(total_points).unsqueeze(0) * 1e-4
        L = torch.linalg.cholesky(K)
        y = torch.matmul(L, torch.randn(self.batch_size, total_points, 1))

        context_y = y[:, : self.num_context, :]
        target_y = y[:, self.num_context :, :]

        context_y_clean = context_y.clone()
        if corruption_fn is not None:
            context_y = corruption_fn(context_x, context_y)
            corruption_label = "shifted"
        else:
            corruption_label = "clean"

        return NPBatch(
            context_x=context_x,
            context_y=context_y,
            target_x=target_x,
            target_y=target_y,
            context_y_clean=context_y_clean,
            corruption_label=corruption_label
        )


class SinusoidData(NPDataset):
    """Generates regression tasks from random sinusoids."""
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

    def generate_batch(self, corruption_fn: callable | None = None,
                       context_x_range: tuple[float, float] | None = None) -> NPBatch:
        # Sample target x from the standard range
        target_x = torch.empty(self.batch_size, self.num_target, 1).uniform_(*self.x_range)
        
        # Sample context x from a potentially shifted range
        cx_range = context_x_range if context_x_range is not None else self.x_range
        context_x = torch.empty(self.batch_size, self.num_context, 1).uniform_(*cx_range)
        
        x = torch.cat([context_x, target_x], dim=1)

        # Sample task parameters once per batch
        amp = torch.empty(self.batch_size, 1, 1).uniform_(*self.amp_range)
        phase = torch.empty(self.batch_size, 1, 1).uniform_(*self.phase_range)
        freq = torch.empty(self.batch_size, 1, 1).uniform_(*self.freq_range)

        y = amp * torch.sin(freq * x + phase)

        context_y = y[:, : self.num_context, :]
        target_y = y[:, self.num_context :, :]

        context_y_clean = context_y.clone()
        if corruption_fn is not None:
            context_y = corruption_fn(context_x, context_y)
            corruption_label = "shifted"
        else:
            corruption_label = "clean"

        return NPBatch(
            context_x=context_x,
            context_y=context_y,
            target_x=target_x,
            target_y=target_y,
            context_y_clean=context_y_clean,
            corruption_label=corruption_label
        )


class UCIData(NPDataset):
    """Loads a real UCI regression dataset and samples random rows to form NP batches."""
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 16,
        num_context: int = 10,
        num_target: int = 20,
    ) -> None:
        self.batch_size = batch_size
        self.num_context = num_context
        self.num_target = num_target
        self.dataset_name = dataset_name
        
        from sklearn.datasets import fetch_california_housing
        import numpy as np
        
        # Load the dataset
        X, y = fetch_california_housing(return_X_y=True)
            
        # Normalize to zero mean, unit variance
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = (y - y.mean()) / (y.std() + 1e-8)
        
        self.X = torch.tensor(X, dtype=torch.float32)
        # Add dimension so y is [N, 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.num_samples = len(self.X)
        self.x_range = (-3.0, 3.0)

    def generate_batch(self, corruption_fn=None, context_x_range=None) -> NPBatch:
        total_points = self.num_context + self.num_target
        
        c_x_list = []
        c_y_list = []
        t_x_list = []
        t_y_list = []
        
        for b in range(self.batch_size):
            # Sample random indices from the dataset for this batch element
            idx = torch.randperm(self.num_samples)[:total_points]
            
            ctx_idx = idx[:self.num_context]
            tgt_idx = idx[self.num_context:]
            
            if context_x_range is not None:
                # If covariate shift is requested, we need context points whose feature 0 is in range
                valid_mask = (self.X[:, 0] >= context_x_range[0]) & (self.X[:, 0] <= context_x_range[1])
                valid_idx = torch.where(valid_mask)[0]
                if len(valid_idx) >= self.num_context:
                    ctx_idx = valid_idx[torch.randperm(len(valid_idx))[:self.num_context]]
            
            c_x_list.append(self.X[ctx_idx])
            c_y_list.append(self.y[ctx_idx])
            t_x_list.append(self.X[tgt_idx])
            t_y_list.append(self.y[tgt_idx])
            
        context_x = torch.stack(c_x_list)
        context_y = torch.stack(c_y_list)
        target_x = torch.stack(t_x_list)
        target_y = torch.stack(t_y_list)
        
        context_y_clean = context_y.clone()
        if corruption_fn is not None:
            context_y = corruption_fn(context_x, context_y)
            corruption_label = "shifted"
        else:
            corruption_label = "clean"
            
        return NPBatch(
            context_x=context_x,
            context_y=context_y,
            target_x=target_x,
            target_y=target_y,
            context_y_clean=context_y_clean,
            corruption_label=corruption_label
        )

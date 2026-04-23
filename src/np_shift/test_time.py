"""Test-time adaptation methods for Neural Processes.

Each adapt_and_predict_* function takes a frozen model and a corrupted batch,
optimises lightweight parameters via pseudo-likelihood on split context,
and returns (mean, variance) predictions on the target set.
"""
import torch
import torch.nn as nn
from torch import optim
import math
from .data import NPBatch

class DenoisingMLP(nn.Module):
    """Small MLP that maps (x, y) -> learned noise shift."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # Zero-init output layer so denoiser starts as identity
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=-1)
        return self.net(inputs)

def nll_loss(mean, var, target_y):
    """Gaussian NLL."""
    return (0.5 * torch.log(2 * torch.pi * var) + 0.5 * (target_y - mean)**2 / var).mean()


class GaussianNoisePrior(nn.Module):
    """A mathematical prior mapping a shift to its Gaussian log-probability."""
    def __init__(self, stddev: float = 1.0):
        super().__init__()
        self.stddev = stddev

    def log_prob(self, shift: torch.Tensor) -> torch.Tensor:
        # Puts highest probability on 0, drops quadratically.
        return -0.5 * (shift / self.stddev)**2 - math.log(self.stddev * math.sqrt(2 * math.pi))


class EmpiricalNoisePrior(nn.Module):
    """A purely data-driven, differentiable generative model built using Kernel Density Estimation (KDE)."""
    def __init__(self, data_path: str, bandwidth: float = 0.5):
        super().__init__()
        import numpy as np
        
        # Load empirical noise samples collected by the researcher
        if data_path.endswith('.npy'):
            samples = np.load(data_path)
        else:
            samples = np.loadtxt(data_path)
            
        samples = torch.tensor(samples, dtype=torch.float32)
        if samples.ndim == 1:
            samples = samples.unsqueeze(-1) # [N, 1]
            
        # Register as a buffer so it moves to GPU automatically if the model is moved
        self.register_buffer("samples", samples) # [N, D]
        self.bandwidth = bandwidth
        self.D = samples.shape[-1]

    def log_prob(self, shift: torch.Tensor) -> torch.Tensor:
        """Evaluates the non-parametric log probability of a shift and allows backpropagation."""
        # shift: [Batch, Context, D] -> flattened to [M, D]
        flat_shift = shift.reshape(-1, self.D)
        
        # Calculate squared L2 distances against all empirical samples [M, N]
        diff_sq = torch.sum((flat_shift.unsqueeze(1) - self.samples.unsqueeze(0))**2, dim=-1)
        
        # Evaluate Multivariate Gaussian kernel for KDE
        exponent = -0.5 * diff_sq / (self.bandwidth**2)
        norm_const = (self.bandwidth ** self.D) * math.sqrt((2 * math.pi) ** self.D)
        
        # LogSumExp gives numerical stability to the sum of kernel probabilities
        log_prob = torch.logsumexp(exponent, dim=1) - math.log(len(self.samples)) - math.log(norm_const)
        
        # Return probability per spatial point: [Batch, Context]
        return log_prob.view(shift.shape[:-1])


def adapt_and_predict_mlp(model, batch: NPBatch, num_steps: int = 100, sgld_noise_scale: float = 0.0, noise_prior: nn.Module = None):
    """Adapt via a learned denoising network. Returns (mean, variance)."""
    model.eval()
    
    with torch.enable_grad():
        denoiser = DenoisingMLP()
        optimizer = optim.Adam(denoiser.parameters(), lr=0.01)

        num_ctx = batch.context_x.size(1)
        split_idx = num_ctx // 2
        
        ctx_x_A = batch.context_x[:, :split_idx, :]
        ctx_y_A = batch.context_y[:, :split_idx, :]
        ctx_x_B = batch.context_x[:, split_idx:, :]
        ctx_y_B = batch.context_y[:, split_idx:, :]

        num_samples = 20 if sgld_noise_scale > 0.0 else 0
        total_steps = num_steps + num_samples
        
        collected_means = []
        collected_vars = []

        for step in range(total_steps):
            optimizer.zero_grad()
            shift_A = denoiser(ctx_x_A, ctx_y_A)
            shift_B = denoiser(ctx_x_B, ctx_y_B)
            
            denoised_y_A = ctx_y_A - shift_A
            denoised_y_B = ctx_y_B - shift_B
            
            out = model(ctx_x_A, denoised_y_A, ctx_x_B)
            loss = nll_loss(out.mean, out.variance, denoised_y_B)
            
            # Maximum A Posteriori (MAP) Inference: Inject explicit prior knowledge of the noise distribution
            if noise_prior is not None:
                # Maximize posterior = minimize NLL - log_prob(prior)
                prior_penalty_A = -noise_prior.log_prob(shift_A).mean()
                prior_penalty_B = -noise_prior.log_prob(shift_B).mean()
                loss = loss + (prior_penalty_A + prior_penalty_B)
                
            loss.backward()
            optimizer.step()
            
            if sgld_noise_scale > 0.0:
                with torch.no_grad():
                    for param in denoiser.parameters():
                        param.add_(torch.randn_like(param) * sgld_noise_scale)
                        
            # If we are in the collection phase, harvest the predictions
            if step >= num_steps and sgld_noise_scale > 0.0:
                with torch.no_grad():
                    final_shift = denoiser(batch.context_x, batch.context_y)
                    final_denoised_context = batch.context_y - final_shift
                    out_after = model(batch.context_x, final_denoised_context, batch.target_x)
                    collected_means.append(out_after.mean)
                    collected_vars.append(out_after.variance)

    # 1) Standard Optimization (Point Estimate)
    if num_samples == 0:
        with torch.no_grad():
            final_shift = denoiser(batch.context_x, batch.context_y)
            final_denoised_context = batch.context_y - final_shift
            out_after = model(batch.context_x, final_denoised_context, batch.target_x)
            
        return out_after.mean, out_after.variance
        
    # 2) Bayesian MCMC (Mixture of Gaussians / Empirical Distribution)
    else:
        mu_stack = torch.stack(collected_means, dim=0) # [K, Batch, N, D]
        var_stack = torch.stack(collected_vars, dim=0) # [K, Batch, N, D]
        
        # Law of Total Variance for a Mixture of equal-weight Gaussians
        final_mean = mu_stack.mean(dim=0)
        final_var = (var_stack + mu_stack**2).mean(dim=0) - final_mean**2
        return final_mean, final_var


def adapt_and_predict_reweight(model, batch: NPBatch, num_steps: int = 100, sgld_noise_scale: float = 0.0):
    """Adapt via per-point attention weights. Returns (mean, variance)."""
    model.eval()
    num_ctx = batch.context_x.size(1)
    split_idx = num_ctx // 2
    
    with torch.enable_grad():
        # Initialize weights near 1.0 (logit=3.0 -> sigmoid=0.95)
        logit_w = torch.full((batch.context_x.size(0), num_ctx, 1), 3.0, requires_grad=True)
        optimizer = optim.Adam([logit_w], lr=0.1)
        
        ctx_x_A = batch.context_x[:, :split_idx, :]
        ctx_y_A = batch.context_y[:, :split_idx, :]
        ctx_x_B = batch.context_x[:, split_idx:, :]
        ctx_y_B = batch.context_y[:, split_idx:, :]

        for step in range(num_steps):
            optimizer.zero_grad()
            w = torch.sigmoid(logit_w)
            w_A = w[:, :split_idx, :]
            w_B = w[:, split_idx:, :]
            
            # Bidirectional cross-prediction
            out_B = model(ctx_x_A, ctx_y_A, ctx_x_B, context_weights=w_A)
            out_A = model(ctx_x_B, ctx_y_B, ctx_x_A, context_weights=w_B)
            
            loss = nll_loss(out_B.mean, out_B.variance, ctx_y_B) + nll_loss(out_A.mean, out_A.variance, ctx_y_A)
            loss.backward()
            optimizer.step()
            
            if sgld_noise_scale > 0.0:
                with torch.no_grad():
                    logit_w.add_(torch.randn_like(logit_w) * sgld_noise_scale)

    with torch.no_grad():
        final_w = torch.sigmoid(logit_w)
        out_after = model(batch.context_x, batch.context_y, batch.target_x, context_weights=final_w)
        return out_after.mean, out_after.variance


def adapt_and_predict_latent(model, batch: NPBatch, num_steps: int = 100, sgld_noise_scale: float = 0.0):
    """Adapt via learnable offset in representation space. Returns (mean, variance)."""
    model.eval()
    num_ctx = batch.context_x.size(1)
    split_idx = num_ctx // 2
    
    with torch.enable_grad():
        latent_shift = torch.zeros(batch.context_x.size(0), num_ctx, model.representation_dim, requires_grad=True)
        optimizer = optim.Adam([latent_shift], lr=0.05)
        
        ctx_x_A = batch.context_x[:, :split_idx, :]
        ctx_y_A = batch.context_y[:, :split_idx, :]
        ctx_x_B = batch.context_x[:, split_idx:, :]
        ctx_y_B = batch.context_y[:, split_idx:, :]

        for step in range(num_steps):
            optimizer.zero_grad()
            shift_A = latent_shift[:, :split_idx, :]
            shift_B = latent_shift[:, split_idx:, :]
            
            out_B = model(ctx_x_A, ctx_y_A, ctx_x_B, latent_value_shift=shift_A)
            out_A = model(ctx_x_B, ctx_y_B, ctx_x_A, latent_value_shift=shift_B)
            
            loss = nll_loss(out_B.mean, out_B.variance, ctx_y_B) + nll_loss(out_A.mean, out_A.variance, ctx_y_A)
            loss.backward()
            optimizer.step()
            
            if sgld_noise_scale > 0.0:
                with torch.no_grad():
                    latent_shift.add_(torch.randn_like(latent_shift) * sgld_noise_scale)

    with torch.no_grad():
        out_after = model(batch.context_x, batch.context_y, batch.target_x, latent_value_shift=latent_shift)
        return out_after.mean, out_after.variance

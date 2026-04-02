import torch
from torch import optim
import importlib
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add src to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from np_shift import AttentionNeuralProcess, SinusoidData, heteroskedastic_noise
from np_shift.viz import plot_np_task

class DenoisingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # map (x, corrupted_y) -> predicted noise shift
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # Initialize output layer to emit zeros initially (identity mapping)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=-1)
        return self.net(inputs)

def nll_loss(mean, var, target_y):
    """Gaussian NLL."""
    return (0.5 * torch.log(2 * torch.pi * var) + 0.5 * (target_y - mean)**2 / var).mean()

def run_test_time_adaptation():
    print("--- Test-Time Parameterized Denoising Prototype ---")
    weights_path = Path("results/sinusoid_10_vanilla/weights.pt")
    if not weights_path.exists():
        print(f"Error: {weights_path} not found. Run 'uv run python scripts/sweep.py' first.")
        return

    # Load "Vanilla" Model (only knows clean data)
    model = AttentionNeuralProcess()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    # Generate a batch with severe Heteroskedastic Noise (x-dependent)
    data_gen = SinusoidData(batch_size=1, num_context=20, num_target=50)
    batch = data_gen.generate_batch(corruption_fn=lambda x, y: heteroskedastic_noise(x, y, scale_factor=2.0))

    # 1. Prediction BEFORE Adaptation
    with torch.no_grad():
        out_before = model(batch.context_x, batch.context_y, batch.target_x)

    # 2. Test-Time Adaptation Loop
    denoiser = DenoisingMLP()
    optimizer = optim.Adam(denoiser.parameters(), lr=0.01)

    # Split the corrupted context for pseudo-likelihood calculation
    num_ctx = batch.context_x.size(1)
    split_idx = num_ctx // 2
    ctx_x_A = batch.context_x[:, :split_idx, :]
    ctx_y_A = batch.context_y[:, :split_idx, :]
    ctx_x_B = batch.context_x[:, split_idx:, :]
    ctx_y_B = batch.context_y[:, split_idx:, :]

    print("Starting adaptation loop with Parameterized Denoising Network...")
    
    losses = []
    shift_magnitudes = []
    
    for step in range(200):
        optimizer.zero_grad()
        
        # Denosie context using the MLP
        shift_A = denoiser(ctx_x_A, ctx_y_A)
        shift_B = denoiser(ctx_x_B, ctx_y_B)
        
        denoised_y_A = ctx_y_A - shift_A
        denoised_y_B = ctx_y_B - shift_B
        
        # Ask NP to predict B from A
        out = model(ctx_x_A, denoised_y_A, ctx_x_B)
        
        # Score the prediction relative to denoised B
        loss = nll_loss(out.mean, out.variance, denoised_y_B)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        shift_magnitudes.append(shift_A.abs().mean().item())
        
        if (step + 1) % 40 == 0:
            print(f"Step {step+1:3d} | Loss: {loss.item():.4f} | Avg Shift Magnitude: {shift_A.abs().mean().item():.4f}")

    # 3. Prediction AFTER Adaptation
    with torch.no_grad():
        final_shift = denoiser(batch.context_x, batch.context_y)
        final_denoised_context = batch.context_y - final_shift
        
        out_after = model(batch.context_x, final_denoised_context, batch.target_x)
        
        # Reproject predictions back into the corrupted space so they align with the true noisy targets
        # The true final targets to predict should ideally be the real observations, not clean targets
        target_shift = denoiser(batch.target_x, out_after.mean)
        final_mean = out_after.mean + target_shift

    # Save visual comparison
    out_dir = Path("results/test_time_adaptation")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot Before
    plot_np_task(
        batch.context_x, batch.context_y, batch.target_x, batch.target_y,
        out_before.mean, out_before.variance, context_y_clean=batch.context_y_clean,
        title=f"BEFORE Adaptation (Vanilla Model, Hetero Noise)",
        save_path=str(out_dir / "before_mlp.png")
    )
    
    # Plot After
    plot_np_task(
        batch.context_x, batch.context_y, batch.target_x, batch.target_y,
        final_mean, out_after.variance, context_y_clean=batch.context_y_clean,
        title=f"AFTER Adaptation (Parameterized Denoising)",
        save_path=str(out_dir / "after_mlp.png")
    )
    
    # Plot learning curve
    plt.figure()
    plt.plot(shift_magnitudes, label="Avg Predicted Shift Magnitude")
    plt.xlabel("Optimization Step")
    plt.ylabel("Magnitude")
    plt.title("Test-Time Optimization of Parameterized Denoising Network")
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_dir / "optimization_curve_mlp.png"))
    plt.close()

    print(f"\nSaved 'before_mlp.png', 'after_mlp.png', and 'optimization_curve_mlp.png' to {out_dir}")

if __name__ == "__main__":
    run_test_time_adaptation()

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import Tensor

def plot_np_task(
    context_x: Tensor,
    context_y: Tensor,
    target_x: Tensor,
    target_y_true: Tensor | None,
    pred_mean: Tensor,
    pred_var: Tensor,
    context_y_clean: Tensor | None = None,
    title: str = "Neural Process Prediction",
    save_path: str | None = None,
):
    """Plot NP prediction for a single batch element (index 0)."""
    # Convert to numpy for plotting
    cx = context_x[0, :, 0].cpu().numpy()
    cy = context_y[0, :, 0].cpu().numpy()
    tx = target_x[0, :, 0].cpu().numpy()
    pm = pred_mean[0, :, 0].cpu().numpy()
    ps = torch.sqrt(pred_var[0, :, 0]).cpu().numpy()
    
    # Sort target points for a clean line plot
    sort_idx = tx.argsort()
    tx, pm, ps = tx[sort_idx], pm[sort_idx], ps[sort_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Plot true target if available
    if target_y_true is not None:
        ty = target_y_true[0, :, 0].cpu().numpy()[sort_idx]
        plt.plot(tx, ty, 'k--', alpha=0.5, label='Ground Truth')
        
    # Plot uncertainty band (2 sigma)
    plt.fill_between(tx, pm - 2*ps, pm + 2*ps, color='C0', alpha=0.2, label='2σ Uncertainty')
    plt.plot(tx, pm, 'C0', lw=2, label='Predicted Mean')
    
    # Plot clean context points as ghosts if they differ from corrupted ones
    if context_y_clean is not None:
        cy_clean = context_y_clean[0, :, 0].cpu().numpy()
        if not np.allclose(cy, cy_clean):
            plt.scatter(cx, cy_clean, c='black', marker='o', alpha=0.3, s=30, label='Clean Context (unseen)')

    # Plot corrupted context points
    plt.scatter(cx, cy, c='red' if context_y_clean is not None else 'black', 
                marker='x', s=50, zorder=10, label='Observed Context (Shifted)' if context_y_clean is not None else 'Context Points')
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

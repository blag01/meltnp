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
    tx_unsorted = tx.copy()
    sort_idx = tx.argsort()
    tx, pm, ps = tx[sort_idx], pm[sort_idx], ps[sort_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Plot true target if available
    if target_y_true is not None:
        ty = target_y_true[0, :, 0].cpu().numpy()
        # Combine target and context points so the line interpolates through both
        true_cx = cx
        true_cy = context_y_clean[0, :, 0].cpu().numpy() if context_y_clean is not None else cy
        
        all_x = np.concatenate([tx_unsorted, true_cx])
        all_y = np.concatenate([ty, true_cy])
        
        sort_idx_all = all_x.argsort()
        plt.plot(all_x[sort_idx_all], all_y[sort_idx_all], 'k--', alpha=0.5, label='Ground Truth')
        
    # Plot uncertainty band (2 sigma)
    plt.fill_between(tx, pm - 2*ps, pm + 2*ps, color='C0', alpha=0.2, label='2σ Uncertainty')
    plt.plot(tx, pm, 'C0', lw=2, label='Predicted Mean')
    
    # Determine if context was actually corrupted
    is_corrupted = (context_y_clean is not None and not np.allclose(cy, context_y_clean[0, :, 0].cpu().numpy()))
    
    # Plot clean context points as ghosts if corrupted
    if is_corrupted:
        cy_clean = context_y_clean[0, :, 0].cpu().numpy()
        plt.scatter(cx, cy_clean, c='black', marker='o', alpha=0.3, s=30, label='Clean Context (unseen)')

    # Plot context points
    plt.scatter(cx, cy, c='red' if is_corrupted else 'black', 
                marker='x', s=50, zorder=10, label='Observed Context (Shifted)' if is_corrupted else 'Context Points')
    
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
        # Save corresponding CSV for comparisons without reloading the model
        import csv
        from pathlib import Path
        csv_path = Path(save_path).with_suffix('.csv')
        
        all_points = {}
        ty_unsorted = target_y_true[0, :, 0].cpu().numpy() if target_y_true is not None else [None] * len(tx_unsorted)
        pm_unsorted = pred_mean[0, :, 0].cpu().numpy()
        ps_unsorted = torch.sqrt(pred_var[0, :, 0]).cpu().numpy()
        
        for x, y_true, m, s in zip(tx_unsorted, ty_unsorted, pm_unsorted, ps_unsorted):
            all_points[x] = {
                "is_context": 0, "cy_clean": "", "cy_obs": "",
                "ty": y_true if y_true is not None else "", "pm": m, "ps": s
            }
            
        cy_clean_arr = context_y_clean[0, :, 0].cpu().numpy() if context_y_clean is not None else cy
        for x, y_clean, y_obs in zip(cx, cy_clean_arr, cy):
            if x not in all_points:
                all_points[x] = {"ty": "", "pm": "", "ps": ""}
            all_points[x]["is_context"] = 1
            all_points[x]["cy_clean"] = y_clean
            all_points[x]["cy_obs"] = y_obs
            
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "is_context", "ground_truth", "observed_context", "pred_mean", "pred_std"])
            
            for x in sorted(all_points.keys()):
                p = all_points[x]
                gt = p["cy_clean"] if p["is_context"] else p["ty"]
                writer.writerow([x, p["is_context"], gt, p["cy_obs"], p["pm"], p["ps"]])
    else:
        plt.show()
    plt.close()

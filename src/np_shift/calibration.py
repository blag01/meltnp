import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_calibration_comparison(det_results: dict, lat_results: dict, save_path: str):
    """
    Plots ECE vs Corruption Intensity for Deterministic TNP vs Latent TNP.
    Produces a 6-panel figure (one for each corruption type).
    """
    shift_types = ["noise", "bias", "hetero", "warp", "outlier", "covariate"]
    
    # We only care about the intersection of shifts present in both
    shifts = [s for s in shift_types if s in det_results and s in lat_results]
    if not shifts:
        print("[Calibration] Warning: No matching corruption data found for comparison.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, st in enumerate(shifts):
        ax = axes[i]
        
        # Extract data
        # Results structure: results[shift_type][model_name]["x" / "ece"]
        # Find the robust or vanilla model (we just grab the first one we find)
        det_models = list(det_results[st].keys())
        lat_models = list(lat_results[st].keys())
        
        if not det_models or not lat_models:
            continue
            
        # We prefer the robust model, fallback to vanilla
        det_target = next((m for m in det_models if "robust" in m), det_models[0])
        lat_target = next((m for m in lat_models if "robust" in m), lat_models[0])
        
        det_data = det_results[st][det_target]
        lat_data = lat_results[st][lat_target]
        
        x = det_data["x"]
        
        # ECE is stored as a list of (mean, std)
        det_ece_m, det_ece_s = zip(*det_data["ece"])
        lat_ece_m, lat_ece_s = zip(*lat_data["ece"])
        
        ax.errorbar(x, det_ece_m, yerr=det_ece_s, label="Deterministic TNP", capsize=3, fmt='-o', color='C0')
        ax.errorbar(x, lat_ece_m, yerr=lat_ece_s, label="Latent TNP", capsize=3, fmt='-s', color='C1')
        
        ax.set_title(f"Corruption: {st.capitalize()}")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Expected Calibration Error (ECE)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
            
    plt.suptitle("Uncertainty Calibration under Distribution Shift", fontsize=16)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[Calibration] Comparison plot saved to {save_path}")

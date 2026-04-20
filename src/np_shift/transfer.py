import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .benchmark import run_stress_test


def run_transfer_matrix(models_by_train_corruption: dict, dataset_name: str, num_context: int, save_path: str):
    """
    Evaluates each model (trained on a specific corruption) against all test corruptions.
    Builds a heat map showing how well robustness transfers across domain shifts.
    """
    test_shifts = ["noise", "bias", "hetero", "warp", "outlier", "covariate"]
    train_shifts = list(models_by_train_corruption.keys())

    # Build matrix [Train Shift, Test Shift]
    matrix = np.zeros((len(train_shifts), len(test_shifts)))

    print("\n[Transfer Matrix] Starting evaluation sweep...")
    for i, train_corr in enumerate(train_shifts):
        model = models_by_train_corruption[train_corr]
        model.eval()
        
        print(f"  Evaluating model trained on: {train_corr}")
        for j, test_corr in enumerate(test_shifts):
            # Evaluate across the full intensity range and take the mean NLL (or AUC)
            # The metric we plot is the average NLL across the entire corruption intensity spectrum
            results = run_stress_test(model, dataset_name, test_corr, num_context=num_context)
            
            # Extract mean NLL across all intensities
            nll_mean = np.mean([m for m, s in results["nll"]])
            matrix[i, j] = nll_mean

    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    
    plt.colorbar(label="Mean Negative Log Likelihood (NLL)")
    plt.xticks(ticks=np.arange(len(test_shifts)), labels=test_shifts, rotation=45)
    plt.yticks(ticks=np.arange(len(train_shifts)), labels=train_shifts)
    
    plt.xlabel("Test-Time Corruption")
    plt.ylabel("Train-Time Augmentation")
    plt.title(f"Cross-Corruption Transferability ({dataset_name}, N={num_context})")
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[Transfer Matrix] Saved to {save_path}")

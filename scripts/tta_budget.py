"""
TTA Budget Curve: How many optimization steps does Test-Time Adaptation need?

Plots NLL vs number of TTA optimization steps for each adaptation method,
answering the compute-quality tradeoff question.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from np_shift import AttentionNeuralProcess, GPData, SinusoidData
from np_shift.benchmark import evaluate_model
from np_shift.data import add_gaussian_noise, heteroskedastic_noise


def run_budget_analysis(z_dim=None):
    print("--- TTA Budget Curve Analysis ---")

    # We test on both datasets with a moderate corruption
    configs = [
        ("sinusoid", lambda x, y: add_gaussian_noise(x, y, std=1.0), "Gaussian Noise (σ=1.0)"),
        ("sinusoid", lambda x, y: heteroskedastic_noise(x, y, scale_factor=1.0), "Heteroskedastic (s=1.0)"),
    ]

    step_budgets = [0, 5, 10, 20, 50, 100, 200]
    tta_methods = ["mlp", "reweight", "latent", "mlp_sgld_0.01", "mlp_sgld_0.05", "mlp_sgld_0.1"]

    root = "results/tnp" if z_dim is None else f"results/z{z_dim}tnp"
    out_dir = Path(f"{root}/10/tta_budget")
    out_dir.mkdir(exist_ok=True, parents=True)

    for dataset_name, corruption_fn, corruption_label in configs:
        weights_path = Path(f"{root}/10/{dataset_name}_vanilla/weights.pt")
        if not weights_path.exists():
            print(f"Skipping {dataset_name}: {weights_path} not found. Run sweep.py first.")
            continue

        model = AttentionNeuralProcess(z_dim=z_dim)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()

        DataClass = GPData if dataset_name == "gp" else SinusoidData
        data_gen = DataClass(batch_size=16, num_context=10)

        # Collect results: method -> list of (steps, nll_mean, nll_std)
        method_results = {m: {"steps": [], "nll_mean": [], "nll_std": []} for m in tta_methods}

        # Also get vanilla baseline (no adaptation)
        baseline = evaluate_model(model, data_gen, corruption_fn=corruption_fn, num_tasks=20)
        baseline_nll = baseline["nll"][0]

        for method in tta_methods:
            print(f"  [{dataset_name} / {corruption_label}] Testing {method}...")
            for n_steps in step_budgets:
                if n_steps == 0:
                    metrics = baseline
                else:
                    # Temporarily patch num_steps via a wrapper
                    from np_shift.test_time import adapt_and_predict_mlp, adapt_and_predict_reweight, adapt_and_predict_latent
                    
                    adapt_fns = {
                        "mlp": adapt_and_predict_mlp,
                        "reweight": adapt_and_predict_reweight,
                        "latent": adapt_and_predict_latent,
                    }
                    
                    parts = method.split("_")
                    base_method = parts[0]
                    noise_scale = 0.0
                    for p in parts[1:]:
                        try:
                            noise_scale = float(p)
                        except ValueError:
                            if p == "sgld" and noise_scale == 0.0:
                                noise_scale = 0.05

                    # Run evaluation manually with custom step count
                    losses = []
                    for _ in range(10):
                        batch = data_gen.generate_batch(corruption_fn=corruption_fn)
                        mean, var = adapt_fns[base_method](model, batch, num_steps=n_steps, sgld_noise_scale=noise_scale)
                        y = batch.target_y
                        log_p = -0.5 * torch.log(2 * np.pi * var) - 0.5 * (y - mean)**2 / var
                        losses.append(-log_p.mean().item())
                    
                    metrics = {"nll": (np.mean(losses), np.std(losses))}

                method_results[method]["steps"].append(n_steps)
                method_results[method]["nll_mean"].append(metrics["nll"][0])
                method_results[method]["nll_std"].append(metrics["nll"][1])

        # Plot
        plt.figure(figsize=(10, 6))
        plt.axhline(baseline_nll, color='gray', linestyle='--', label='Vanilla (no TTA)', alpha=0.7)
        
        for method, data in method_results.items():
            plt.errorbar(
                data["steps"], data["nll_mean"], yerr=data["nll_std"],
                label=f"TTA: {method}", capsize=3, fmt='-o'
            )

        plt.title(f"TTA Budget Curve — {dataset_name} / {corruption_label}")
        plt.xlabel("Number of TTA Optimization Steps")
        plt.ylabel("Negative Log Likelihood")
        plt.legend()
        plt.grid(True, alpha=0.3)

        safe_label = corruption_label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        save_path = out_dir / f"budget_{dataset_name}_{safe_label}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\nAll budget curves saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--z-dim", type=int, default=None)
    args = parser.parse_args()
    
    run_budget_analysis(z_dim=args.z_dim)

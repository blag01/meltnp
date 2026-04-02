import subprocess
import sys
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to PYTHONPATH so we can import np_shift
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from np_shift import AttentionNeuralProcess, run_stress_test, plot_robustness_curves


def run_training_phase(experiments):
    """Phase 1: Train all models."""
    for dataset, robust, num_context in experiments:
        mode = "robust" if robust else "vanilla"
        output_dir = Path(f"results/{dataset}_{num_context}_{mode}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = output_dir / "weights.pt"
        
        cmd = [
            sys.executable, "scripts/train.py",
            "--dataset", dataset,
            "--num-context", str(num_context),
            "--epochs", "1000",
            "--output", str(weights_path)
        ]
        if robust:
            cmd.append("--robust")
            
        print(f"\n>>> [Train] {dataset}_{num_context} ({mode})")
        subprocess.run(cmd, check=True)

def run_benchmarking_phase(experiments):
    """Phase 2 & 3: Benchmark and Report."""
    print("\nStarting scientific benchmarking phase...")
    shift_types = ["noise", "bias", "hetero", "warp", "outlier", "covariate"]
    groups = sorted(set((d, c) for d, _, c in experiments))
    
    # Structure: {(dataset, num_context): {shift_type: {model_name: results}}}
    all_results = {g: {st: {} for st in shift_types} for g in groups}
    
    for dataset, robust, num_context in experiments:
        mode = "robust" if robust else "vanilla"
        model_name = f"{dataset}_{num_context}_{mode}"
        weights_path = Path(f"results/{dataset}_{num_context}_{mode}/weights.pt")
        
        if not weights_path.exists():
            continue
            
        model = AttentionNeuralProcess()
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        
        print(f"Stress-testing model: {model_name}...")
        for st in shift_types:
            all_results[(dataset, num_context)][st][model_name] = run_stress_test(
                model, dataset, st, num_context=num_context)
            
        # Add TTA tracks for vanilla models to compare against explicitly robust ones
        if not robust:
            for tta_method in ["mlp", "reweight", "latent"]:
                tta_name = f"{model_name}_tta_{tta_method}"
                print(f"Stress-testing model: {tta_name} (with inference-time optimization)...")
                for st in shift_types:
                    all_results[(dataset, num_context)][st][tta_name] = run_stress_test(
                        model, dataset, st, adapt_method=tta_method, num_context=num_context)

    # Generate comparative plots
    print("Generating Comparative Robustness Curves...")
    plot_dir = Path("results/plots")
    for ds, ctx in groups:
        for st in shift_types:
            if all_results[(ds, ctx)][st]:
                st_dir = plot_dir / st
                plot_robustness_curves(all_results[(ds, ctx)][st], str(st_dir), file_prefix=f"{ds}_{ctx}")
    print(f"All plots saved to {plot_dir}/")

def main():
    datasets = ["gp", "sinusoid"]
    robust_flags = [False, True]
    context_sizes = [10, 20, 40]
    
    experiments = []
    for ctx in context_sizes:
        for ds in datasets:
            for r in robust_flags:
                experiments.append((ds, r, ctx))
    
    run_training_phase(experiments)
    run_benchmarking_phase(experiments)

    print("\nSweep Complete! Results including master COMPARISON are in 'results/'")


if __name__ == "__main__":
    main()

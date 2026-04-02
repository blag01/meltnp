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
    for dataset, robust in experiments:
        mode = "robust" if robust else "vanilla"
        output_dir = Path(f"results/{dataset}_{mode}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = output_dir / "weights.pt"
        
        cmd = [
            sys.executable, "scripts/train.py",
            "--dataset", dataset,
            "--epochs", "1000",
            "--output", str(weights_path)
        ]
        if robust:
            cmd.append("--robust")
            
        print(f"\n>>> [Train] {dataset} ({mode})")
        subprocess.run(cmd, check=True)

def run_benchmarking_phase(experiments):
    """Phase 2 & 3: Benchmark and Report."""
    print("\nStarting scientific benchmarking phase...")
    shift_types = ["noise", "bias", "hetero", "warp", "outlier", "covariate"]
    datasets = sorted(set(d for d, _ in experiments))
    
    # Structure: {dataset: {shift_type: {model_name: results}}}
    all_results = {ds: {st: {} for st in shift_types} for ds in datasets}
    
    for dataset, robust in experiments:
        mode = "robust" if robust else "vanilla"
        model_name = f"{dataset}_{mode}"
        weights_path = Path(f"results/{dataset}_{mode}/weights.pt")
        
        if not weights_path.exists():
            continue
            
        model = AttentionNeuralProcess()
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        
        print(f"Stress-testing model: {model_name}...")
        for st in shift_types:
            all_results[dataset][st][model_name] = run_stress_test(model, dataset, st)
            
        # Add TTA tracks for vanilla models to compare against explicitly robust ones
        if not robust:
            for tta_method in ["mlp", "reweight", "latent"]:
                tta_name = f"{model_name}_tta_{tta_method}"
                print(f"Stress-testing model: {tta_name} (with inference-time optimization)...")
                for st in shift_types:
                    all_results[dataset][st][tta_name] = run_stress_test(model, dataset, st, adapt_method=tta_method)

    # Generate comparative plots — one per (dataset, shift_type), flat in results/plots/
    print("Generating Comparative Robustness Curves...")
    plot_dir = Path("results/plots")
    for ds in datasets:
        for st in shift_types:
            if all_results[ds][st]:
                plot_robustness_curves(all_results[ds][st], str(plot_dir), file_prefix=f"{ds}_{st}")
    print(f"All plots saved to {plot_dir}/")

def main():
    experiments = [
        ("gp", False),
        ("gp", True),
        ("sinusoid", False),
        ("sinusoid", True),
    ]
    
    run_training_phase(experiments)
    run_benchmarking_phase(experiments)

    print("\nSweep Complete! Results including master COMPARISON are in 'results/'")


if __name__ == "__main__":
    main()

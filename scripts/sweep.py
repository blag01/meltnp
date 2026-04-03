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


def run_training_phase(experiments, z_dim=None):
    """Phase 1: Train all models."""
    for dataset, robust, num_context in experiments:
        mode = "robust" if robust else "vanilla"
        if z_dim is not None:
            mode += f"_z{z_dim}"
        output_dir = Path(f"results/{num_context}/{dataset}_{mode}")
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
        if z_dim is not None:
            cmd.extend(["--z-dim", str(z_dim)])
            
        print(f"\n>>> [Train] {dataset}_{num_context} ({mode})")
        subprocess.run(cmd, check=True)

def run_benchmarking_phase(experiments, z_dim=None):
    """Phase 2 & 3: Benchmark and Report."""
    print("\nStarting scientific benchmarking phase...")
    shift_types = ["noise", "bias", "hetero", "warp", "outlier", "covariate"]
    groups = sorted(set((d, c) for d, _, c in experiments))
    
    # Structure: {(dataset, num_context): {shift_type: {model_name: results}}}
    all_results = {g: {st: {} for st in shift_types} for g in groups}
    
    for dataset, robust, num_context in experiments:
        mode = "robust" if robust else "vanilla"
        if z_dim is not None:
            mode += f"_z{z_dim}"
        model_name = f"{dataset}_{mode}_{num_context}"
        weights_path = Path(f"results/{num_context}/{dataset}_{mode}/weights.pt")
        
        if not weights_path.exists():
            continue
            
        model = AttentionNeuralProcess(z_dim=z_dim)
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

    print("Generating Comparative Robustness Curves...")
    for ds, ctx in groups:
        plot_dir = Path(f"results/{ctx}/plots")
        for st in shift_types:
            if all_results[(ds, ctx)][st]:
                st_dir = plot_dir / st
                st_dir.mkdir(parents=True, exist_ok=True)
                plot_robustness_curves(all_results[(ds, ctx)][st], str(st_dir), file_prefix=ds)
    print("All plots saved locally for each context size!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the full benchmarking sweep.")
    parser.add_argument("--no-train", action="store_true", help="Blacklist: skip the training phase.")
    parser.add_argument("--no-bench", action="store_true", help="Blacklist: skip the benchmarking phase.")
    parser.add_argument("--no-extra", action="store_true", help="Blacklist: skip the extra TTA scripts (budget and visual prototypes).")
    parser.add_argument("--z-dim", type=int, default=None, help="Dimension of latent variable z (None = Deterministic TNP).")
    args = parser.parse_args()

    run_train = not args.no_train
    run_bench = not args.no_bench
    run_extra = not args.no_extra

    datasets = ["gp", "sinusoid"]
    robust_flags = [False, True]
    context_sizes = [10, 20, 40]
    
    experiments = []
    for ctx in context_sizes:
        for ds in datasets:
            for r in robust_flags:
                experiments.append((ds, r, ctx))
    
    if run_train:
        run_training_phase(experiments, z_dim=args.z_dim)
    
    if run_bench:
        run_benchmarking_phase(experiments, z_dim=args.z_dim)

    if run_extra:
        extra_args = ["--z-dim", str(args.z_dim)] if args.z_dim is not None else []
        print("\n>>> [Extra] Running Test-Time Adaptation Prototypes (scripts/test_time_adapt.py)")
        subprocess.run([sys.executable, "scripts/test_time_adapt.py"] + extra_args, check=True)
        
        print("\n>>> [Extra] Running TTA Budget Curves (scripts/tta_budget.py)")
        subprocess.run([sys.executable, "scripts/tta_budget.py"] + extra_args, check=True)

    print("\nSweep Complete! Results including master COMPARISON are in 'results/'")


if __name__ == "__main__":
    main()

import subprocess
import sys
from pathlib import Path

def run_experiment(dataset: str, robust: bool, epochs: int = 1000):
    mode = "robust" if robust else "vanilla"
    output_dir = Path(f"results/{dataset}_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = output_dir / "weights.pt"
    
    cmd = [
        sys.executable, "scripts/train.py",
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--output", str(weights_path)
    ]
    if robust:
        cmd.append("--robust")
        
    print(f"\n>>> Starting Experiment: {dataset} ({mode})")
    subprocess.run(cmd, check=True)
    print(f">>> Finished {dataset} ({mode}). Results in {output_dir}/")


def main():
    experiments = [
        ("gp", False),
        ("gp", True),
        ("sinusoid", False),
        ("sinusoid", True),
    ]
    
    print(f"Starting sweep of {len(experiments)} experiments...")
    
    for dataset, robust in experiments:
        try:
            run_experiment(dataset, robust, epochs=1000)
        except Exception as e:
            print(f"!!! Experiment failed: {dataset} (robust={robust}): {e}")

    print("\nSweep complete! Check the 'results/' directory for weights and plots.")


if __name__ == "__main__":
    main()

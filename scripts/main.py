from pathlib import Path
import sys

PROJECT_NAME = "neural-processes-distribution-shift"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from np_shift import AttentionNeuralProcess, GPData, SinusoidData, add_gaussian_noise


def main() -> None:
    # Initialize model and data generator
    model = AttentionNeuralProcess()
    # Try to load weights if they exist (from train.py)
    try:
        model.load_state_dict(torch.load("model_weights.pt", weights_only=True))
        print("Loaded trained model weights.")
    except FileNotFoundError:
        print("No trained weights found. Using random initialization.")

    # Choose dataset here (GPData or SinusoidData)
    data_gen = GPData(batch_size=1, num_context=5, num_target=10)
    # data_gen = SinusoidData(batch_size=1, num_context=5, num_target=10)

    # Scenarios using simple functions (lambdas for custom parameters)
    scenarios = [
        ("Clean", None),
        ("Gaussian Noise (std=0.8)", lambda y: add_gaussian_noise(y, std=0.8)),
    ]

    for label, corruption_fn in scenarios:
        batch = data_gen.generate_batch(corruption_fn=corruption_fn)
        
        with torch.no_grad():
            output = model(batch.context_x, batch.context_y, batch.target_x)

        print(f"\n--- Scenario: {label} ---")
        if batch.target_y is not None:
            for i in range(batch.target_x.size(1)):
                x = batch.target_x[0, i, 0].item()
                true_y = batch.target_y[0, i, 0].item()
                pred_y = output.mean[0, i, 0].item()
                std_y = torch.sqrt(output.variance[0, i, 0]).item()
                print(f"x={x:5.2f} | true={true_y:6.3f} | pred={pred_y:6.3f} ± {std_y:.3f}")


if __name__ == "__main__":
    main()

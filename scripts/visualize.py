import argparse
from pathlib import Path
import torch

from np_shift import AttentionNeuralProcess, GPData, SinusoidData, add_gaussian_noise, apply_bias_shift
from np_shift.viz import plot_np_task


def main():
    parser = argparse.ArgumentParser(description="Visualize NP predictions.")
    parser.add_argument("--dataset", type=str, default="gp", choices=["gp", "sinusoid"])
    parser.add_argument("--shift", type=str, default="none", choices=["none", "noise", "bias"])
    parser.add_argument("--weights", type=str, default="model_weights.pt")
    args = parser.parse_args()

    # Create plots directory
    Path("plots").mkdir(exist_ok=True)

    # Initialize model
    model = AttentionNeuralProcess()
    try:
        model.load_state_dict(torch.load(args.weights, weights_only=True))
        print(f"Loaded weights from {args.weights}")
    except FileNotFoundError:
        print("Using random weights (no checkpoint found).")

    # Initialize data generator
    DataClass = GPData if args.dataset == "gp" else SinusoidData
    data_gen = DataClass(batch_size=1)

    # Define corruption function
    corr_fn = None
    if args.shift == "noise":
        corr_fn = lambda y: add_gaussian_noise(y, std=0.8)
    elif args.shift == "bias":
        corr_fn = lambda y: apply_bias_shift(y, shift_range=(1.5, 2.5))

    # Generate task and predict
    batch = data_gen.generate_batch(corruption_fn=corr_fn)
    model.eval()
    with torch.no_grad():
        output = model(batch.context_x, batch.context_y, batch.target_x)

    # Plot
    filename = f"plots/{args.dataset}_{args.shift}.png"
    plot_np_task(
        context_x=batch.context_x,
        context_y=batch.context_y,
        target_x=batch.target_x,
        target_y_true=batch.target_y,
        pred_mean=output.mean,
        pred_var=output.variance,
        title=f"NP Task: {args.dataset} (shift={args.shift})",
        save_path=filename
    )
    print(f"Success! qualitative visualization generated at {filename}")


if __name__ == "__main__":
    main()

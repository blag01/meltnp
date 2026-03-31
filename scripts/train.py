import argparse
from pathlib import Path
import sys
import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from np_shift import AttentionNeuralProcess, GPData, SinusoidData, add_gaussian_noise, apply_bias_shift
from np_shift.viz import plot_np_task


def nll_loss(mean: torch.Tensor, variance: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_p = -0.5 * torch.log(2 * torch.pi * variance) - 0.5 * (y - mean)**2 / variance
    return -log_p.mean()


def train(args):
    model = AttentionNeuralProcess()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Select dataset
    DataClass = GPData if args.dataset == "gp" else SinusoidData
    data_gen = DataClass(batch_size=args.batch_size)
    
    # Configure potential corruptions
    corruptions = [None]
    if args.robust:
        corruptions += [add_gaussian_noise, apply_bias_shift]

    print(f"Training on {args.dataset} (robust={args.robust}) for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        corr_fn = corruptions[torch.randint(0, len(corruptions), (1,)).item()]
        batch = data_gen.generate_batch(corruption_fn=corr_fn)
        
        model.train()
        optimizer.zero_grad()
        
        output = model(batch.context_x, batch.context_y, batch.target_x)
        
        if batch.target_y is not None:
            loss = nll_loss(output.mean, output.variance, batch.target_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    # Save weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Weights saved to {output_path}")

    # Integrated Visualization
    print("Generating post-training visualization...")
    model.eval()
    with torch.no_grad():
        # Evaluate on a single clean task for visualization
        batch = data_gen.generate_batch(corruption_fn=None)
        output = model(batch.context_x, batch.context_y, batch.target_x)
        
        plot_path = output_path.with_suffix(".png")
        plot_np_task(
            context_x=batch.context_x,
            context_y=batch.context_y,
            target_x=batch.target_x,
            target_y_true=batch.target_y,
            pred_mean=output.mean,
            pred_var=output.variance,
            title=f"Result: {args.dataset} (robust={args.robust})",
            save_path=str(plot_path)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Process.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="gp", choices=["gp", "sinusoid"])
    parser.add_argument("--robust", action="store_true", help="Enable random corruptions during training.")
    parser.add_argument("--output", type=str, default="model_weights.pt", help="File to save weights.")
    
    args = parser.parse_args()
    train(args)

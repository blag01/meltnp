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
    # Select dataset
    if args.dataset == "gp":
        data_gen = GPData(batch_size=args.batch_size, num_context=args.num_context)
    elif args.dataset == "sinusoid":
        data_gen = SinusoidData(batch_size=args.batch_size, num_context=args.num_context)
    elif args.dataset == "uci":
        from np_shift.data import UCIData
        data_gen = UCIData(dataset_name="california", batch_size=args.batch_size, num_context=args.num_context)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
        
    # Get arbitrary feature dimension (needed for UCI where x_dim=8)
    dummy_batch = data_gen.generate_batch()
    x_dim = dummy_batch.context_x.shape[-1]
    
    model = AttentionNeuralProcess(x_dim=x_dim, z_dim=args.z_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Configure potential corruptions
    if not args.robust:
        args.robust_type = "clean"
        
    corruptions = []
    if args.robust_type == "clean":
        corruptions = [None]
    elif args.robust_type == "noise":
        corruptions = [lambda x, y: add_gaussian_noise(x, y, std=1.0)]
    elif args.robust_type == "bias":
        corruptions = [lambda x, y: apply_bias_shift(x, y, shift_range=(1.0, 1.0))]
    elif args.robust_type == "hetero":
        from np_shift.data import heteroskedastic_noise
        corruptions = [lambda x, y: heteroskedastic_noise(x, y, scale_factor=1.0)]
    elif args.robust_type == "warp":
        from np_shift.data import apply_warp_shift
        corruptions = [lambda x, y: apply_warp_shift(x, y, warp_power=2.0)]
    elif args.robust_type == "outlier":
        from np_shift.data import inject_outliers
        corruptions = [lambda x, y: inject_outliers(x, y, fraction=0.3, magnitude=5.0)]
    elif args.robust_type == "all":
        import np_shift.data as dt
        corruptions = [
            None,
            lambda x, y: dt.add_gaussian_noise(x, y, std=1.0),
            lambda x, y: dt.apply_bias_shift(x, y, shift_range=(1.0, 1.0)),
        ]
    else:
        raise ValueError(f"Unknown robust-type {args.robust_type}")

    print(f"Training on {args.dataset} (robust_type={args.robust_type}) for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        corr_fn = corruptions[torch.randint(0, len(corruptions), (1,)).item()]
        batch = data_gen.generate_batch(corruption_fn=corr_fn)
        
        model.train()
        optimizer.zero_grad()
        
        output = model(batch.context_x, batch.context_y, batch.target_x, target_y=batch.target_y)
        
        if batch.target_y is not None:
            loss = nll_loss(output.mean, output.variance, batch.target_y)
            
            # Add KL Divergence for ELBO if using Latent TNP
            if output.posterior_mu is not None:
                post_mu, post_s = output.posterior_mu, output.posterior_log_sigma
                prior_mu, prior_s = output.prior_mu, output.prior_log_sigma
                
                post_var = torch.exp(2 * post_s)
                prior_var = torch.exp(2 * prior_s)
                kl = prior_s - post_s + (post_var + (post_mu - prior_mu)**2) / (2 * prior_var) - 0.5
                kl = kl.sum(dim=-1).mean()
                
                loss = loss + kl / batch.target_y.shape[1]

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
        
        # Ensure target set includes context set so predictions span all observations
        full_tgt_x = torch.cat([batch.context_x, batch.target_x], dim=1)
        full_tgt_y = torch.cat([batch.context_y, batch.target_y], dim=1)
        
        output = model(batch.context_x, batch.context_y, full_tgt_x)
        
        plot_path = output_path.with_suffix(".png")
        plot_np_task(
            context_x=batch.context_x,
            context_y=batch.context_y,
            target_x=full_tgt_x,
            target_y_true=full_tgt_y,
            pred_mean=output.mean,
            pred_var=output.variance,
            context_y_clean=batch.context_y_clean,
            title=f"Result: {args.dataset} (robust={args.robust})",
            save_path=str(plot_path)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neural Process.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dataset", type=str, default="gp", choices=["gp", "sinusoid", "uci"])
    parser.add_argument("--num-context", type=int, default=10, help="Number of context points.")
    parser.add_argument("--z-dim", type=int, default=None, help="Dimension of latent variable z (None = Deterministic TNP).")
    parser.add_argument("--robust", action="store_true", help="Enable random corruptions during training.")
    parser.add_argument("--robust-type", type=str, default="all", choices=["clean", "noise", "bias", "hetero", "warp", "outlier", "all"], help="Specific corruption to train against.")
    parser.add_argument("--output", type=str, default="model_weights.pt", help="File to save weights.")
    
    args = parser.parse_args()
    train(args)

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .data import GPData, SinusoidData, add_gaussian_noise, apply_bias_shift, heteroskedastic_noise, apply_warp_shift

from .test_time import adapt_and_predict_mlp, adapt_and_predict_reweight, adapt_and_predict_latent

def evaluate_model(model, data_gen, corruption_fn=None, num_tasks=50, adapt_method=None):
    """Calculate mean metrics over multiple tasks."""
    model.eval()
    losses = []
    mses = []
    eces = []
    
    with torch.no_grad():
        for _ in range(num_tasks):
            batch = data_gen.generate_batch(corruption_fn=corruption_fn)
            
            if adapt_method == "mlp":
                mean, var = adapt_and_predict_mlp(model, batch, num_steps=50)
            elif adapt_method == "reweight":
                mean, var = adapt_and_predict_reweight(model, batch, num_steps=50)
            elif adapt_method == "latent":
                mean, var = adapt_and_predict_latent(model, batch, num_steps=50)
            else:
                output = model(batch.context_x, batch.context_y, batch.target_x)
                mean, var = output.mean, output.variance
                
            y = batch.target_y
            std = torch.sqrt(var)
            
            # 1. NLL calculation
            log_p = -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (y - mean)**2 / var
            losses.append(-log_p.mean().item())
            
            # 2. MSE calculation
            mse = ((y - mean)**2).mean().item()
            mses.append(mse)
            
            # 3. ECE calculation (Regression via standard normal CDF)
            z = (y - mean) / std
            cdf = torch.distributions.Normal(0, 1).cdf(z).flatten().cpu().numpy()
            counts, _ = np.histogram(cdf, bins=np.linspace(0, 1, 11))
            fractions = counts / len(cdf)
            ece = np.abs(fractions - 0.1).mean()
            eces.append(ece)
            
    return {
        "nll": (np.mean(losses), np.std(losses)),
        "mse": (np.mean(mses), np.std(mses)),
        "ece": (np.mean(eces), np.std(eces)),
    }

def run_stress_test(model, dataset_name, shift_type, adapt_method=None):
    """Measure metrics vs corruption intensity."""
    intensity_range = np.linspace(0.0, 2.0, 10)
    results = {"x": [], "nll": [], "mse": [], "ece": []}
    
    DataClass = GPData if dataset_name == "gp" else SinusoidData
    data_gen = DataClass(batch_size=16)
    
    for val in intensity_range:
        if shift_type == "noise":
            fn = lambda x, y: add_gaussian_noise(x, y, std=val)
        elif shift_type == "bias":
            fn = lambda x, y: apply_bias_shift(x, y, shift_range=(val, val))
        elif shift_type == "hetero":
            fn = lambda x, y: heteroskedastic_noise(x, y, scale_factor=val)
        elif shift_type == "warp":
            fn = lambda x, y: apply_warp_shift(x, y, warp_power=1.0 + val)
        else:
            fn = None
            
        metrics = evaluate_model(model, data_gen, corruption_fn=fn, num_tasks=10, adapt_method=adapt_method)
        results["x"].append(val)
        results["nll"].append(metrics["nll"])
        results["mse"].append(metrics["mse"])
        results["ece"].append(metrics["ece"])
        
    return results

def plot_robustness_curves(sweep_results, save_dir):
    """
    Generate plots comparing different models across multiple metrics.
    sweep_results: dict mapping model_name -> results dict
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "nll": "Negative Log Likelihood",
        "mse": "Mean Squared Error",
        "ece": "Expected Calibration Error"
    }
    
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10, 6))
        for name, data in sweep_results.items():
            x = data["x"]
            # data[metric_key] is a list of tuples (mean, std)
            m, s = zip(*data[metric_key])
            plt.errorbar(x, m, yerr=s, label=name, capsize=3, fmt='-o')
            
        plt.title(f"Robustness to Distribution Shift ({metric_name})")
        plt.xlabel("Noise Intensity (std)")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(Path(save_dir) / f"robustness_curve_{metric_key}.png")
        plt.close()

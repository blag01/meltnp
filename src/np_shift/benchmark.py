import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from .data import GPData, SinusoidData, add_gaussian_noise, apply_bias_shift, heteroskedastic_noise, apply_warp_shift, inject_outliers

from .test_time import adapt_and_predict_mlp, adapt_and_predict_reweight, adapt_and_predict_latent

def evaluate_model(model, data_gen, corruption_fn=None, num_tasks=50, adapt_method=None, context_x_range=None):
    """Calculate mean metrics over multiple tasks."""
    model.eval()
    losses = []
    mses = []
    eces = []
    
    with torch.no_grad():
        for _ in range(num_tasks):
            batch = data_gen.generate_batch(corruption_fn=corruption_fn, context_x_range=context_x_range)
            
            if adapt_method is not None and adapt_method.replace("_sgld", "") in ["mlp", "reweight", "latent"]:
                use_sgld = adapt_method.endswith("_sgld")
                base_method = adapt_method.replace("_sgld", "")
                noise_scale = 0.05 if use_sgld else 0.0
                
                if base_method == "mlp":
                    mean, var = adapt_and_predict_mlp(model, batch, num_steps=50, sgld_noise_scale=noise_scale)
                elif base_method == "reweight":
                    mean, var = adapt_and_predict_reweight(model, batch, num_steps=50, sgld_noise_scale=noise_scale)
                elif base_method == "latent":
                    mean, var = adapt_and_predict_latent(model, batch, num_steps=50, sgld_noise_scale=noise_scale)
            else:
                output = model(batch.context_x, batch.context_y, batch.target_x)
                mean, var = output.mean, output.variance
                
            y = batch.target_y
            std = torch.sqrt(var)
            
            # NLL
            log_p = -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (y - mean)**2 / var
            losses.append(-log_p.mean().item())
            
            # MSE
            mse = ((y - mean)**2).mean().item()
            mses.append(mse)
            
            # ECE (regression calibration via standard normal CDF)
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

def run_stress_test(model, dataset_name, shift_type, adapt_method=None, num_context=10):
    """Measure metrics vs corruption intensity."""
    intensity_range = np.linspace(0.0, 2.0, 10)
    results = {"x": [], "nll": [], "mse": [], "ece": []}
    
    DataClass = GPData if dataset_name == "gp" else SinusoidData
    data_gen = DataClass(batch_size=16, num_context=num_context)
    
    for val in intensity_range:
        if shift_type == "noise":
            fn = lambda x, y: add_gaussian_noise(x, y, std=val)
        elif shift_type == "bias":
            fn = lambda x, y: apply_bias_shift(x, y, shift_range=(val, val))
        elif shift_type == "hetero":
            fn = lambda x, y: heteroskedastic_noise(x, y, scale_factor=val)
        elif shift_type == "warp":
            fn = lambda x, y: apply_warp_shift(x, y, warp_power=1.0 + val)
        elif shift_type == "outlier":
            fn = lambda x, y: inject_outliers(x, y, fraction=0.3, magnitude=val * 5.0)
        elif shift_type == "covariate":
            # Shift context x-range progressively away from [-2, 2]
            cx_range = (data_gen.x_range[0] + val, data_gen.x_range[1] + val)
            fn = None
        else:
            fn = None
        
        cx = cx_range if shift_type == "covariate" else None
        metrics = evaluate_model(model, data_gen, corruption_fn=fn, num_tasks=10, adapt_method=adapt_method, context_x_range=cx)
        results["x"].append(val)
        results["nll"].append(metrics["nll"])
        results["mse"].append(metrics["mse"])
        results["ece"].append(metrics["ece"])
        
    return results

def plot_robustness_curves(sweep_results, save_dir, file_prefix="robustness"):
    """
    Generate plots comparing different models across multiple metrics.
    sweep_results: dict mapping model_name -> results dict
    file_prefix: prefix for output filenames (e.g. "gp_noise" -> "gp_noise_nll.png")
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
            m, s = zip(*data[metric_key])
            plt.errorbar(x, m, yerr=s, label=name, capsize=3, fmt='-o')
            
        plt.title(f"Robustness to Distribution Shift ({metric_name})")
        plt.xlabel("Corruption Intensity")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(Path(save_dir) / f"{file_prefix}_{metric_key}.png")
        plt.close()
        
        # Save corresponding CSV
        csv_path = Path(save_dir) / f"{file_prefix}_{metric_key}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            x_vals = list(sweep_results.values())[0]["x"]
            
            headers = ["Intensity"]
            for name in sweep_results.keys():
                headers.extend([f"{name}_mean", f"{name}_std"])
            writer.writerow(headers)
            
            for i, x in enumerate(x_vals):
                row = [x]
                for name in sweep_results.keys():
                    m, s = sweep_results[name][metric_key][i]
                    row.extend([m, s])
                writer.writerow(row)

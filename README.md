# Neural Processes under Distribution Shift

Benchmarking Neural Process robustness and inference-time adaptation under structured distribution shift.

## Setup

```bash
uv sync
```

## Layout

- `src/np_shift/` — model, data generators, benchmarking, and TTA modules
- `scripts/` — runnable experiments
- `docs/` — research notes and methodology

## Commands

### Train a single model

```bash
# Train on GP data (vanilla, clean)
uv run python scripts/train.py --dataset gp

# Train on sinusoid data with corruption augmentation
uv run python scripts/train.py --dataset sinusoid --robust
```

Options: `--dataset {gp,sinusoid}`, `--robust`, `--epochs N`, `--lr F`
Outputs: `results/<dataset>_<mode>/weights.pt` and `weights.png`

### Run the full experiment suite

```bash
uv run python scripts/sweep.py
```

Trains all 4 model variants, then stress-tests each across **6 corruption types** (noise, bias, heteroskedastic, warp, outlier, covariate) and **3 TTA methods** (MLP denoiser, context reweighting, latent reprojection). Generates robustness curves for NLL, MSE, and ECE.

Outputs: `results/plots/{shift}/{dataset}_{metric}.png` (e.g. `results/plots/noise/gp_nll.png`)

### TTA budget analysis

```bash
uv run python scripts/tta_budget.py
```

Plots NLL vs number of TTA optimization steps (0–200) for each adaptation method.  
Requires trained weights from `sweep.py`.

Outputs: `results/tta_budget/budget_*.png`

### TTA prototype (standalone)

```bash
uv run python scripts/test_time_adapt.py
```

Quick visual demo of the parameterized denoising MLP on a single corrupted sinusoid task. Generates before/after plots.

Outputs: `results/test_time_adaptation/{before,after}_mlp.png`

### Tests

```bash
uv run pytest
```


# Architecture Overview

## Core Modules (`src/np_shift/`)

### `neural_process.py`
Attention-based Neural Process with cross-attention between context and target. Supports two optional hooks for test-time adaptation:
- `latent_value_shift` — additive offset to value embeddings
- `context_weights` — multiplicative scaling of value embeddings

### `data.py`
Data generation interface with an `NPDataset` abstract base class.

**Generators:**
- `GPData` — Gaussian Process with RBF kernel
- `SinusoidData` — random amplitude/phase/frequency sinusoids

Both support `corruption_fn(x, y)` for applying distribution shifts and `context_x_range` for covariate shift.

**Corruption functions:**
| Function | Type | Description |
|---|---|---|
| `add_gaussian_noise` | Additive | Uniform noise across all points |
| `apply_bias_shift` | Additive | Constant shift per batch |
| `heteroskedastic_noise` | Structured | Noise scales with \|x\| |
| `apply_warp_shift` | Non-linear | $y \to \text{sign}(y) \cdot \|y\|^p$ |
| `inject_outliers` | Sparse | Random fraction of points get extreme values |

### `benchmark.py`
Quantitative evaluation computing NLL, MSE, and ECE. `run_stress_test` sweeps corruption intensity from 0 to 2 across all shift types.

### `test_time.py`
Three inference-time adaptation methods. See `docs/test_time_adaptation.md`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train.py` | Train a single model with CLI args |
| `scripts/sweep.py` | Full pipeline: train → benchmark → plot |
| `scripts/tta_budget.py` | TTA quality vs optimisation steps |
| `scripts/test_time_adapt.py` | Visual TTA prototype |

## Output Structure

```
results/
├── {dataset}_{mode}/
│   ├── weights.pt          # model checkpoint
│   └── weights.png         # post-training visualisation
├── plots/{shift_type}/
│   └── {dataset}_{metric}.png   # robustness curves
└── tta_budget/
    └── budget_*.png        # compute-quality tradeoff
```

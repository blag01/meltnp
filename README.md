# Neural Processes under Distribution Shift

Benchmarking Neural Process robustness and inference-time adaptation under structured distribution shift.

## Setup

```bash
uv sync
```

## Reproducing all results

```bash
uv run python scripts/run.py all     # full run: train → benchmark → TTA → budget curves
uv run python scripts/run.py plot     # re-run benchmark + extras on existing weights
uv run python scripts/run.py present  # generate assets/Neural_Processes_Robustness_Presentation.pptx
uv run python scripts/run.py help     # show all options and sweep flags
```

Outputs land in `results/tnp/` (deterministic) and `results/z16tnp/` (latent).

## Layout

- `src/np_shift/` — model, data generators, benchmarking, and TTA modules
- `scripts/` — runnable experiments
- `docs/` — research notes and methodology

## What the sweep produces

For each model variant × corruption type × context size:
- `results/{root}/{ctx}/plots/{shift}/{dataset}_{metric}.png` — robustness curves (NLL, MSE, ECE)
- `results/{root}/{ctx}/test_time_adaptation/` — before/after TTA qualitative plots
- `results/{root}/{ctx}/tta_budget/` — NLL vs gradient step budget curves

## Sweep CLI flags

| Flag | Effect |
|---|---|
| `--z-dims none 16` | Which model variants to run (`none` = deterministic TNP, `16` = Latent TNP) |
| `--clean` | Wipe `results/` before starting (ensures no stale artifacts) |
| `--plots-only` | Skip training, re-run benchmark + extras on existing weights |
| `--no-train` | Skip training phase only |
| `--no-bench` | Skip benchmark phase only |
| `--no-extra` | Skip TTA visual and budget scripts |

## Generate presentation slides

```bash
uv run --group dev python scripts/generate_presentation.py
# Output: assets/Neural_Processes_Robustness_Presentation.pptx
```

## Tests

```bash
uv run pytest
```

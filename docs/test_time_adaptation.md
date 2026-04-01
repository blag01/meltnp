# Test-Time Adaptation (TTA) Methodology

## The Problem
A "Vanilla" Neural Process trained strictly on clean data will fail catastrophically when presented with corrupted context points (distribution shift) at test time.

## The Algorithmic Solution
Instead of retraining the `vanilla` model explicitly on corrupted data (Track A), we can leave the model weights frozen and perform an optimization loop *at inference time* to denoise the context points.

### The Mechanism: Pseudo-Likelihood Optimization
1. **Latent Discrepancy Variable**: We introduce a learnable, latent discrepancy parameter (e.g., a scalar bias `c`).
2. **Context Splitting**: At test time, we take the corrupted context batch and split it into two halves: Context A and Context B.
3. **Denoising**: We apply the current guess for `c` to "denoise" both halves: $\tilde{y}_A = y_A - c$ and $\tilde{y}_B = y_B - c$.
4. **Scoring**: We feed $\tilde{y}_A$ into the NP as context and ask it to predict $\tilde{y}_B$.
5. **Gradient Descent**: We calculate the Negative Log Likelihood (NLL). Because the NP expects clean data (zero-mean signals), the NLL will be minimized precisely when `c` successfully removes the corruption, driving $\tilde{y}_B$ back to the clean manifold. We backpropagate this NLL to update `c`.

## Development Strategy & Ultimate Goal

### Phase 1: Prototyping (Current State)
We are currently evaluating the TTA loop using a standalone script (`scripts/test_time_adapt.py`).
* **Why**: The optimization loop involves sensitive inference-time hyperparameters (learning rate, optimization steps, context splitting logic). A standalone script runs quickly and exposes qualitative learning curves and before/after plots that are essential for debugging and tuning the algorithm itself.

### Phase 2: Sweep Integration (The Ultimate Goal)
* **Goal**: Once the mathematical stability and hyperparameter tuning of the adaptation algorithm are confirmed via the standalone script, it must be integrated into the master research suite (`scripts/sweep.py`).
* **Execution**: We will add a new evaluation track in the `run_benchmarking_phase` of `sweep.py`: `"Vanilla Model + TTA"`.
* **Output**: This will firmly place the inference-time adaptation baseline on the exact same quantitative robustness curve as the explicitly retrained `"Robust Model"`, allowing us to definitively answer the core research question: *Does amortized inference-time adaptation match or beat explicit structural robustness training?*

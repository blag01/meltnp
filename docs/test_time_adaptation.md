# Test-Time Adaptation (TTA) Methodology

## The Problem
A "Vanilla" Neural Process trained strictly on clean data will fail catastrophically when presented with corrupted context points at test time.

## The Solution: Pseudo-Likelihood Optimization
Instead of retraining the model on corrupted data, we leave the model weights frozen and optimise lightweight parameters *at inference time* to denoise the context.

### How it works
1. **Context Splitting**: Split the corrupted context into two halves: A and B.
2. **Denoising**: Apply the current adaptation parameters to denoise both halves.
3. **Cross-Prediction**: Feed denoised A into the NP and predict B (and vice versa for bidirectional methods).
4. **Gradient Descent**: Minimise the NLL of the cross-prediction. Because the NP expects clean data, the loss is minimised when the adaptation parameters successfully remove the corruption.

## Three Adaptation Methods

All methods are implemented in `src/np_shift/test_time.py`.

### 1. Parameterized Denoising (`adapt_and_predict_mlp`)
A small MLP $f_\theta(x, y) \to \text{shift}$ learns an input-dependent correction. Handles heteroskedastic and structured noise since the shift can vary per point.

### 2. Context Reweighting (`adapt_and_predict_reweight`)
Optimises a sigmoid weight $w_i \in [0,1]$ per context point, applied multiplicatively to the NP's value embeddings via `context_weights`. Effectively teaches the cross-attention to ignore corrupted outliers. Uses bidirectional cross-prediction.

### 3. Latent Reprojection (`adapt_and_predict_latent`)
Adds a learnable offset directly to the NP encoder's value embeddings via `latent_value_shift`. Operates in representation space rather than input space. Uses bidirectional cross-prediction.

## Integration

All three methods are fully integrated into the benchmarking suite:
- `scripts/sweep.py` evaluates them as `{dataset}_vanilla_tta_{mlp,reweight,latent}` alongside vanilla and robust baselines.
- `scripts/tta_budget.py` measures the compute-quality tradeoff (NLL vs number of optimisation steps).
- `scripts/test_time_adapt.py` provides a standalone visual prototype.

## Model Hooks

The `AttentionNeuralProcess.forward()` method accepts two optional parameters to support TTA:
- `latent_value_shift`: additive offset to the context encoder's value embeddings.
- `context_weights`: multiplicative scaling of value embeddings.

These hooks allow TTA methods to intervene in the NP's internals without modifying its weights.

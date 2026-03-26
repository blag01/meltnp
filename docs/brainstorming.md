# Brainstorming

## High-Level View

The core idea is strong, but it naturally splits into two separate projects:

1. Training-time robustness, where the corruption family is known and the Neural Process amortizes over it during training.
2. Test-time adaptation, where the model is trained on clean data and must respond to misspecification that only becomes apparent from the observed context.

These are related, but they ask different questions and should not be treated as equally central at the start.

## Main Recommendation

The training-time direction is the cleaner and more defensible primary project.

It gives a more controlled setup:

- define a corruption family,
- define a prior over that family,
- measure the tradeoff between robustness and clean-regime performance.

That framing supports both empirical work and possible theory.

## Why Track A Looks Stronger First

The most interesting question is not just whether corruption-aware augmentation helps, but whether amortized inference can represent robustness without paying much in the well-specified regime.

That suggests a sharper thesis:

Neural Processes can be made robust to distribution shift by explicitly amortizing over corruption families, and the benefit depends on whether the corruption is identifiable from context without destroying clean-task specialization.

## Concerns About Track B

The test-time idea is ambitious and potentially novel, but also methodologically harder.

The central issue is that Neural Processes do not provide a clean marginal likelihood, so a fully Bayesian adaptation story is not immediate. A more practical starting point would be:

- train an NP on clean tasks,
- introduce a latent corruption or discrepancy variable at test time,
- infer that variable from context using a surrogate score or pseudo-likelihood.

That is easier to operationalize than starting from a full Kennedy-O'Hagan style formulation.

## Concrete Research Directions

### Training-Time Misspecification

- Treat corruption as a latent task variable rather than just generic noise augmentation.
- Make the clean-regime tradeoff explicit and measurable.
- Start with additive Gaussian corruption only as a baseline.
- Move quickly to more structured corruptions such as bias shifts, heteroskedastic noise, or warped outputs.

### Test-Time Misspecification

- Focus first on practical adaptation mechanisms rather than a full likelihood-based Bayesian story.
- Explore latent denoising, discrepancy variables, or context reweighting.
- Only later ask whether those adaptation rules admit a principled Bayesian interpretation.

## Important Narrowing Decision

The current framing still uses "distribution shift" too broadly. It would help to choose early which of the following is the main object of study:

- observation corruption,
- simulator discrepancy,
- latent task shift,
- covariate shift in inputs,
- or a mixture.

These are not equivalent, and Neural Processes may behave very differently across them.

## Suggested Project Shape

- Make training-time misspecification the main project.
- Treat test-time adaptation as an extension.
- Build one benchmark family where the corruption variable is sometimes identifiable from context and sometimes not.

That setup is likely to produce the clearest and most informative results.

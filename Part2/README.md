
---

## Modules

### `GenerateData.py`
Synthetic data generation for discrete choice experiments.

- **`SimParams`** — dataclass holding all DGP hyperparameters
- **`generate_data_tf(dgp_type, sim)`** — generates market-level choice data
  under four DGP types (1–4) that vary in the sparsity and endogeneity
  structure. Returns a raw numpy dict for the MCMC runner, a
  `ChoiceDataset` for choice-learn, and a dict of true parameters.

---

### `BLP.py`
Berry–Levinsohn–Pakes (BLP) demand estimation with random coefficients.

- **`_rc_logit_draw_probs`** — computes logit choice probabilities for a single
  market across Monte Carlo draws of the price coefficient.
- **`blp_build_ivs`** — constructs instrument matrices (cost shifters or
  polynomial cost IVs) for BLP.
- **`FastBLP`** — class implementing the BLP contraction mapping and two-stage
  GMM objective. 
- **`blp_estimator_fast`** — convenience wrapper that runs the full two-stage
  BLP estimator.

---

### `DeepHalo.py`
Context-dependent neural encoder based on Zhang et al. (2025).

- **`NonlinearMap`** — a single residual block with layer normalisation, used
  as a building block for the encoder.
- **`DeepHaloEncoder`** — permutation-equivariant encoder that maps product
  characteristics and availability masks to latent embeddings.

---

### `LuSparseRandomLogit.py`
Bayesian Sparse Random Logit MCMC sampler, following Lu (2025).

- **`BayesianSparseRandomLogit`** — main model class. Implements a full
  posterior sampler of the Lu(2025) method. Supports
  `beta_method="rwmh"` (Random Walk MH) and `beta_method="tmh"` (Tailored MH
  using a Newton–Raphson mode-finding proposal).
- **`mh_update_beta_cl`** — RWMH step for β̄; proposal scale controlled by
  `mcmc_params.step_beta`.
- **`tmh_update_beta_cl`** — TMH step for β̄; proposal covariance scale
  controlled by `mcmc_params.kappa_beta`.
- **`mh_update_xi_cl`** — MH step for market-level mean quality ξ̄.
- **`mh_update_r_cl`** — MH step for random coefficient scales r.
- **`gibbs_update_gamma_phi_tf`** — Gibbs step for spike-and-slab inclusion
  indicators γ and sparsity probability φ.
- **`adapt_step_size`** — Robbins–Monro step-size adaptation based on
  empirical acceptance rate.
- **`calibrate_stepsizes_cl`** — runs a short pilot chain and adapts all step
  sizes before the main MCMC run.

---

### `DeepHalo_MCEM_Core.py`
Monte Carlo EM outer loop that jointly trains the DeepHalo encoder and the
Bayesian Sparse logit model.

- **`SparseDeepHaloMCEM`** — main class coordinating the EM loop. Each outer
  iteration runs an E-step (MCMC posterior draws over the latent utility
  parameters given the current encoder embeddings Z) followed by an M-step
  (gradient update of the encoder to maximise the expected log-likelihood).
  Exposes `run()`, `e_step()`, and `m_step()`.
- **`build_choice_dataset_from_market_counts`** — converts market-level count
  arrays `q` and encoder embeddings `Z` into a
  `ChoiceDataset` suitable for the MCMC sampler.
- **`compute_probs_and_ll_batch_masked`** — vectorised TF computation of
  choice probabilities and log-likelihood across all markets, handling
  availability masks and the outside option.

---

## Notebooks

### `Lu_Bayesian_Sparse_rwmh.ipynb`
Replicates Section 4 of Lu (2025). Runs the Bayesian Sparse Random Logit MCMC
sampler on simulated data under the four DGP types, comparing the two beta
update methods:
- **RWMH** (`beta_method="rwmh"`) — random walk Metropolis-Hasting scaled by
  `step_beta`.
- **TMH** (`beta_method="tmh"`) — tailored Metropolis-Hasting scaled by `kappa_beta`.

Produces MCMC trace plots, posterior summaries, and acceptance rate diagnostics
for each method. Corresponds to **answer (b)** of the assignment.

---

### `Joint_Bayesian_Sparse_DeepHalo.ipynb`
Implements the joint model combining Lu (2025) and Zhang et al. (2025).
The DeepHalo encoder replaces raw product features with
context-dependent latent features Z learned end-to-end via the MCEM loop in
`SparseDeepHaloMCEM`. Demonstrates how the neural encoder and the Bayesian
sparse logit interact across outer EM iterations, and reports in-sample fit and
posterior parameter estimates. Corresponds to **answer (d)** of the assignment.

---

### `Tests.ipynb`
Interactive test runner for the `BayesianSparseDeepHalo` package. Executes the
full test suite in `tests/Bayesian_Sparse_Deephalo_Tests.py` (92 tests across
16 classes) directly from a Jupyter cell using:

```python
!python -m pytest tests/Bayesian_Sparse_Deephalo_Tests.py -v

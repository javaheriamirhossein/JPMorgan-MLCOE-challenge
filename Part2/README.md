
---

## Modules

### `GenerateData.py`
Synthetic data generation for discrete choice experiments.

- **`SimParams`** — dataclass holding all DGP hyperparameters
- **`MCMCParams`** — dataclass holding MCMC sampler hyperparameters
- **`generate_data_tf(dgp_type, sim)`** — generates market-level choice data
  under four DGP types (1–4) that vary in the sparsity and endogeneity
  structure. Returns a raw numpy dict for the MCMC runner, a
  `ChoiceDataset` for choice-learn, and a dict of true parameters.
- **`generate_teacher_dgp(sim, encoder_kwarg, beta_star, r_star)`** — generates context-dependent
  market-level choice data based on the given encoder, simulation, and heterogeneous taste parameters.
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
Bayesian Sparse Random Logit MCMC sampler implemented with [`tfp.mcmcm`](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc), following Lu (2025).
We use Random Walk Metropolis-Hastings (RWMH) for beta_bar, r_vec, xi_bar, eta, phi (continuous variables) 
and exact Gibbs sampling with `tfmcmc.TransitionKernel` for gamma.   
- **`BayesianSparseRandomLogit`** — main model class. Implements a full
  posterior sampler of the Lu(2025) method. 
- **`GammaGibbsKernel`** — Exact Gibbs sampler for the binary spike-and-slab inclusion matrix gamma.
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

### `Lu_Bayesian_Sparse_Demo.ipynb`
Replicates Section 4 of Lu (2025). Runs the Bayesian Sparse Random Logit MCMC
sampler on simulated data under the four DGP types.

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
Comprehensive test suite for the `BayesianSparseDeepHalo` package. Executes the
full test suite for all the modules in `tests` folder. The tests can also be directly run from a terminal as:

```python
python -m pytest tests -v

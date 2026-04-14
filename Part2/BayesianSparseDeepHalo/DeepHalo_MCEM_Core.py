"""
Hybrid Monte Carlo EM for Sparse Deep Halo Logit.

Algorithm Overview
------------------
This module implements a Monte Carlo Expectation-Maximisation (MCEM) loop that
jointly trains a deep neural network encoder and a Bayesian sparse random logit model.

The key insight is a division of labour between two parameter types:

  Structural parameters (beta_bar, r_vec):
      Estimated via gradient descent (Adam SGD) in the M-step.
      These are the mean and log-std of the random coefficient distribution.

  Random effects (xi_bar, eta, gamma, phi):
      Inferred via MCMC (Bayesian Sparse Random Logit) in the E-step.
      These capture unobserved market- and product-level heterogeneity.

MCEM Loop (one outer iteration)
---------------------------------
  E-step: Run MCMC to obtain posterior samples of the random effects
          {xi_bar, eta, gamma, phi}, conditioning on the current encoder Z = f(X).
          The chain is "warm-started" from the previous iteration's final state
          to avoid burn-in overhead in later iterations.

  M-step: Fix the posterior means from the E-step and maximise the expected
          complete-data log-likelihood with respect to the encoder weights
          and structural parameters (beta_bar, r_vec) using Adam SGD.

Hybrid Design
-------------
This is NOT a pure EM algorithm — the M-step uses stochastic gradient ascent
(rather than closed-form maximisation), making this a Generalised EM / MCEM.
The deep encoder and the logit model share the latent representation Z = f(X),
so the encoder learns to produce market embeddings that best explain choices.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple

from choice_learn.data import ChoiceDataset
from .LuSparseRandomLogit import BayesianSparseRandomLogit


# ============================================================
# SparseDeepHaloMCEM
# ============================================================

class SparseDeepHaloMCEM:
    """
    Sparse Deep Halo Monte Carlo EM model.

    Combines a deep neural network encoder with a Bayesian sparse random logit
    model. The encoder maps raw product feature matrices X into a latent
    embedding Z = f(X). SParsity is promoted via a spike-and-slab prior on the product-level
    random effects eta.

    Parameters
    ----------
    X_np    : np.ndarray, shape [T, M, F]
              Raw product features for T markets, M inside products, F raw features.
    q_np    : np.ndarray, shape [T, M+1]
              Choice count matrix. q_np[t, j] = number of consumers in market t
              who chose product j (j=0 is the outside option).
    avail_np: np.ndarray, shape [T, M]
              Binary availability mask (1 = available, 0 = unavailable).
    encoder : tf.keras.Model
              Deep neural network that maps (X [T, M, F], avail [T, M]) -> Z [T, M, Dz].
    sim_params  : Namespace with T, J, D fields (market / product / feature counts).
    mcmc_params : Namespace with MCMC settings (G, burn, taus, step sizes, R0, etc.).
    em_params   : Namespace with EM settings (outer_iters, nn_steps, nn_lr, nn_l2, thin, burn, etc.).
    seed        : int, master random seed.
    verbose     : bool, print diagnostics.
    """

    def __init__(
        self,
        X_np: np.ndarray,          # [T, M, F]: raw product features
        q_np: np.ndarray,          # [T, M+1]: choice counts including outside option
        avail_np: np.ndarray,      # [T, M]: product availability mask
        encoder,                   # tf.keras.Model: X -> Z embeddings
        sim_params,
        mcmc_params,
        em_params,
        seed: int = 2026,
        verbose: bool = False
    ) -> None:
        # Store raw inputs as numpy for later dataset construction
        self.X_np = np.asarray(X_np)          # [T, M, F]
        self.q_np = np.asarray(q_np)          # [T, M+1]
        self.avail_np = np.asarray(avail_np)  # [T, M]
        self.encoder = encoder

        self.sim_params = sim_params
        self.mcmc_params = mcmc_params
        self.em_params = em_params
        self.seed = int(seed)
        self.verbose = verbose

        # Keep TF constant copies for graph execution
        self.X = tf.constant(self.X_np, tf.float64)       # [T, M, F]
        self.avail = tf.constant(self.avail_np, tf.float64)  # [T, M]

        # Infer dimensions: T markets, M products, F features
        self.T, self.M, _ = self.X_np.shape  # int, int, int

        # Probe the encoder to determine the output embedding dimension Dz
        dummy_Z = self.encoder(self.X[:2], self.avail[:2], training=False)
        self.Dz = int(dummy_Z.shape[-1])  # int: latent embedding dimension

        # Adam optimiser for the M-step (updates encoder weights + structural params)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(em_params.nn_lr))

        # Structural parameters: mean coefficients [Dz] and log-std [Dz] of random effects.
        # These are updated by SGD in the M-step (NOT by MCMC).
        self.beta_var = tf.Variable(tf.zeros((self.Dz,), tf.float64))  # [Dz]: mean coefs
        self.r_var = tf.Variable(tf.zeros((self.Dz,), tf.float64))     # [Dz]: log-std coefs

        # E-step state storage for warm-starting the MCMC chain across outer iterations
        self.posterior_means = None      # dict: latest posterior means of random effects
        self.last_mcmc_state = None      # dict: last MCMC draw (used to resume chain)
        self.initial_mcmc_state = None   # dict: externally provided starting state (optional)

    def _Z_np(self, training: bool = False) -> np.ndarray:
        """
        Run a forward pass of the encoder to get current market embeddings Z.

        Parameters
        ----------
        training : bool, whether to run in training mode (affects dropout/BN).

        Returns
        -------
        Z : np.ndarray, shape [T, M, Dz]
        """
        Z = self.encoder(self.X, self.avail, training=training)
        return Z.numpy()  # [T, M, Dz]

    # ─────────────────────────────────────────────
    # M-Step: Neural Network + Structural Parameter Update
    # ─────────────────────────────────────────────

    @tf.function(reduce_retracing=True)
    def _mstep_loss(
        self,
        xi_mean: tf.Tensor,   # [T_unique]: posterior mean of market quality shifter
        eta_mean: tf.Tensor,  # [T_unique, J]: posterior mean of product deviations
        v_draws: tf.Tensor    # [R0, Dz]: fixed Monte Carlo draws for integration
    ) -> tf.Tensor:
        """
        Compute the M-step objective: negative expected complete-data log-likelihood + L2 penalty.

        The expected log-likelihood is approximated by plugging in the E-step posterior
        means of the random effects (xi_bar, eta), then maximising over the encoder
        and structural parameters (beta_var, r_var) via gradient descent.

        The L2 regularisation is applied ONLY to the encoder network weights,
        not to beta_var or r_var, to avoid shrinking the structural parameters.

        Returns
        -------
        loss : tf.Tensor, scalar — the M-step loss to be minimised.
        """
        # Encode the raw product features to get the current latent embedding
        Z = self.encoder(self.X, self.avail, training=True)  # [T, M, Dz]
        q = tf.cast(self.q_np, tf.float64)                   # [T, M+1]

        # Use the SGD-updated structural parameters, NOT the MCMC posterior means
        beta = self.beta_var   # [Dz]
        r = self.r_var         # [Dz]

        xi = tf.cast(xi_mean, tf.float64)    # [T_unique]
        eta = tf.cast(eta_mean, tf.float64)  # [T_unique, J]

        # Combine market mean quality (xi_bar) with product deviations (eta)
        xi_inside = xi[:, None] + eta  # [T_unique, J]

        # Evaluate the masked simulated log-likelihood
        _, _, ll_vec = compute_probs_and_ll_batch_masked(
            Z, q, self.avail, xi_inside, beta, r, v_draws
        )  # ll_vec: [T]

        # Normalised NLL: divide by total number of observations for scale invariance
        nll = -tf.reduce_sum(ll_vec) / (tf.reduce_sum(q) + 1e-12)  # scalar

        # L2 regularisation on encoder weights only (prevents encoder overfitting)
        l2 = tf.add_n([tf.reduce_sum(w ** 2) for w in self.encoder.trainable_weights])
        return nll + tf.cast(self.em_params.nn_l2, tf.float64) * l2  # scalar

    def _mstep(self, post: Dict[str, np.ndarray]) -> float:
        """
        M-Step: Maximise the expected complete-data log-likelihood.

        Runs `em_params.nn_steps` Adam gradient steps jointly updating:
          - self.encoder weights (the deep network)
          - self.beta_var (mean coefficients of random effects)
          - self.r_var (log-std of random effects)

        The random effects (xi_bar, eta) are FIXED at their E-step posterior means
        and treated as observed data for the purpose of this optimisation.

        Parameters
        ----------
        post : dict with keys 'xi_bar' [T_unique] and 'eta' [T_unique, J] — E-step posterior means.

        Returns
        -------
        last_loss : float — the final M-step loss value.
        """
        # Fix simulation draws for this M-step (reproducible Monte Carlo integration)
        tf.random.set_seed(self.seed + 777)
        v_draws = tf.random.normal((self.mcmc_params.R0, self.Dz), dtype=tf.float64)  # [R0, Dz]

        # Joint variable list: encoder weights + structural scalar parameters
        trainable_vars = self.encoder.trainable_weights + [self.beta_var, self.r_var]

        last = None
        for _ in range(int(self.em_params.nn_steps)):
            with tf.GradientTape() as tape:
                # Compute M-step loss using E-step random effects as fixed inputs
                loss = self._mstep_loss(post["xi_bar"], post["eta"], v_draws)

            # Compute and apply gradients to encoder + structural parameters
            grads = tape.gradient(loss, trainable_vars)
            self.opt.apply_gradients(zip(grads, trainable_vars))
            last = float(loss.numpy())

        return last  # float: final M-step loss

    # ─────────────────────────────────────────────
    # E-Step: MCMC Posterior Sampling of Random Effects
    # ─────────────────────────────────────────────

    def _estep(self) -> Tuple[Dict[str, np.ndarray], BayesianSparseRandomLogit]:
        """
        E-Step: Approximate the posterior distribution of the random effects via MCMC.

        What happens here:
        1. Encode X -> Z using the current (frozen) encoder weights.
        2. Build a ChoiceDataset from Z and the choice counts q.
        3. Instantiate a BayesianSparseRandomLogit model and warm-start it from
           the previous iteration's MCMC chain state (or the provided initial state).
        4. Run the MCMC chain for `mcmc_per_outer` iterations.
        5. Thin and average the post-burn draws to get posterior means.

        Key design decisions:
        - beta_var and r_var are frozen in the E-step (step sizes set to ~0).
          They are treated as fixed parameters, not sampled. This avoids conflating
          MCMC uncertainty in beta with the SGD estimate from the M-step.
        - The MCMC chain is warm-started: the last draw from the previous E-step
          is used as the initial state, making the chain continuous across outer iterations.
        - In early EM iterations (outer_iter < 3), a full burn-in is run to allow
          the step sizes to adapt. Later iterations use burn=0 to avoid discarding draws.

        Returns
        -------
        post  : dict with posterior means of all random effects + locked beta/r values.
        model : the fitted BayesianSparseRandomLogit instance (for diagnostics).
        """
        # Encode current X -> Z (frozen encoder, no gradient)
        Z_np = self._Z_np(training=False)  # [T, M, Dz]

        # Build the ChoiceDataset from encoded features and choice counts
        ds = build_choice_dataset_from_market_counts(Z_np, self.q_np, self.avail_np)

        # Set MCMC chain length for this E-step
        self.mcmc_params.G = int(self.em_params.mcmc_per_outer)
        self.mcmc_params.burn = int(self.em_params.burn)

        # Lock beta and r: set their RWMH step sizes to essentially zero so
        # the Metropolis proposal never moves them — they stay fixed at M-step values.
        self.mcmc_params.step_beta = 1e-12
        self.mcmc_params.step_r = 1e-12
        # Disable random coefficients in the MCMC model (handled by SGD instead)
        self.mcmc_params.random_coef_mask = np.zeros(self.Dz, dtype=np.float64)

        # Early EM: use full burn-in to allow step sizes to adapt via Robbins-Monro
        # Later EM: skip burn-in to record all draws immediately (chain already warm)
        is_early_em = getattr(self, 'outer_iter_count', 0) < 3
        if is_early_em or self.em_params.recalibrate_each_outer:
            self.mcmc_params.burn = int(self.em_params.burn)  # e.g. 50: adapt then discard
        else:
            self.mcmc_params.burn = 0  # All draws are post-burn; record immediately

        # Instantiate a fresh BayesianSparseRandomLogit for this E-step
        model = BayesianSparseRandomLogit(
            sim_params=self.sim_params,
            mcmc_params=self.mcmc_params,
            random_coef_mask=self.mcmc_params.random_coef_mask,
            seed=self.seed + 123,
            verbose=self.verbose
        )
        model._initialize_from_dataset(ds)

        # Choose which state to warm-start from:
        # Prefer the last MCMC state (continuous chain), fall back to external initial state
        state_to_load = (
            self.last_mcmc_state if self.last_mcmc_state is not None
            else self.initial_mcmc_state
        )

        if state_to_load is not None:
            try:
                # Inject M-step SGD values of beta and r into the MCMC model
                # These are treated as FIXED parameters during the E-step
                current_beta = self.beta_var.numpy()  # [Dz]
                current_r = self.r_var.numpy()        # [Dz]

                if hasattr(model, 'beta_bar') and model.beta_bar is not None:
                    model.beta_bar.assign(tf.cast(current_beta, tf.float64))

                if hasattr(model, 'r_vec') and model.r_vec is not None:
                    model.r_vec.assign(tf.cast(current_r, tf.float64))

                # Restore the random effects state to continue the MCMC chain
                if hasattr(model, 'xi_bar') and model.xi_bar is not None:
                    model.xi_bar.assign(tf.cast(state_to_load['xi_bar'], tf.float64))  # [T_unique]
                if hasattr(model, 'eta') and model.eta is not None:
                    model.eta.assign(tf.cast(state_to_load['eta'], tf.float64))        # [T_unique, J]
                if hasattr(model, 'gamma') and model.gamma is not None:
                    model.gamma.assign(tf.cast(state_to_load['gamma'], tf.int32))      # [T_unique, J]
                if hasattr(model, 'phi') and model.phi is not None:
                    model.phi.assign(tf.cast(state_to_load['phi'], tf.float64))        # [T_unique]
            except Exception as e:
                print(f"Warning: Could not restore MCMC state: {e}")

        # Run the MCMC chain (G iterations, burn-in as configured above)
        _ = model.fit(ds, store_samples=True)
        draws = model.samples_  # dict of arrays, each shape [n_post_samples, ...]

        # Save the LAST draw of each random effect to warm-start the next E-step
        # This keeps the chain continuous across outer EM iterations
        self.last_mcmc_state = {
            'xi_bar': draws['xi_bar'][-1],   # [T_unique]: last xi draw
            'eta':    draws['eta'][-1],       # [T_unique, J]: last eta draw
            'gamma':  draws['gamma'][-1],     # [T_unique, J]: last gamma draw
            'phi':    draws['phi'][-1],       # [T_unique]: last phi draw
        }

        # Thinning: keep every `thin`-th sample to reduce autocorrelation in the posterior mean
        thin = int(self.em_params.thin)
        if thin > 1:
            draws = {k: v[::thin] for k, v in draws.items()}

        # Compute posterior means from the (thinned) post-burn draws
        post = {
            # beta and r come from SGD (M-step), not MCMC draws
            "beta_bar": self.beta_var.numpy(),         # [Dz]
            "r_vec":    self.r_var.numpy(),            # [Dz]
            # Random effects: Monte Carlo averages over post-burn MCMC draws
            "xi_bar":   draws['xi_bar'].mean(axis=0), # [T_unique]
            "eta":      draws["eta"].mean(axis=0),    # [T_unique, J]
            "phi":      draws["phi"].mean(axis=0),    # [T_unique]
            "gamma":    draws["gamma"].mean(axis=0),  # [T_unique, J]: inclusion probabilities
        }

        # Update posterior mean cache (used by M-step as fixed random effects)
        self.posterior_means = post

        # Track the outer iteration count for early-EM burn-in logic
        if not hasattr(self, 'outer_iter_count'):
            self.outer_iter_count = 0
        self.outer_iter_count += 1

        return post, model

    # ─────────────────────────────────────────────
    # Outer MCEM Loop
    # ─────────────────────────────────────────────

    def run(self, tqdm_outer=None) -> Dict[str, Any]:
        """
        Run the full MCEM outer loop for `em_params.outer_iters` iterations.

        Each iteration:
          1. E-step: MCMC posterior sampling of random effects (xi_bar, eta, gamma, phi).
          2. M-step: SGD update of the encoder and structural parameters (beta_var, r_var).

        Parameters
        ----------
        tqdm_outer : optional tqdm progress wrapper for the outer loop.

        Returns
        -------
        results : dict with final parameter estimates and training losses.
            - outer_losses  : list[float], M-step loss per outer iteration.
            - beta_last     : np.ndarray [Dz], final mean coefficients.
            - r_last        : np.ndarray [Dz], final log-std coefficients.
            - xi_bar_last   : np.ndarray [T_unique], final market quality posterior means.
            - eta_last      : np.ndarray [T_unique, J], final product deviation posterior means.
            - phi_last      : np.ndarray [T_unique], final inclusion probability posterior means.
            - step_*        : float, final adapted RWMH step sizes for each parameter.
        """
        # Build the outer iteration range, optionally wrapping with tqdm progress bar
        outer_iter = (
            range(int(self.em_params.outer_iters)) if tqdm_outer is None
            else tqdm_outer(range(int(self.em_params.outer_iters)), desc="Sparse DeepHalo MCEM outer")
        )

        outer_losses = []
        for _ in outer_iter:
            # E-step: sample posterior of random effects | current encoder + beta/r
            post, _ = self._estep()

            # M-step: update encoder + beta/r | fixed posterior means from E-step
            loss = self._mstep(post)
            outer_losses.append(loss)

        # Collect and return all final parameter estimates
        return {
            "outer_losses":  outer_losses,
            "beta_last":     self.beta_var.numpy(),               # [Dz]: final mean coefs
            "r_last":        self.r_var.numpy(),                  # [Dz]: final log-std coefs
            "xi_bar_last":   self.posterior_means["xi_bar"],      # [T_unique]
            "eta_last":      self.posterior_means["eta"],         # [T_unique, J]
            "phi_last":      self.posterior_means["phi"],         # [T_unique]
            "step_beta":     float(self.mcmc_params.step_beta),   # float: adapted RWMH step size
            "step_r":        float(self.mcmc_params.step_r),
            "step_xibar":    float(self.mcmc_params.step_xibar),
            "step_eta":      float(self.mcmc_params.step_eta),
        }


# ============================================================
# Masked Likelihood Helper
# ============================================================

@tf.function
def compute_probs_and_ll_batch_masked(
    Z: tf.Tensor,        # [T, M, Dz]: latent product embeddings from the encoder
    q: tf.Tensor,        # [T, M+1]: choice counts including outside option (j=0)
    avail: tf.Tensor,    # [T, M]: binary product availability mask
    xi: tf.Tensor,       # [T, M]: combined quality shifter xi_bar + eta (inside products)
    beta_bar: tf.Tensor, # [Dz]: mean coefficients of random effects
    r_vec: tf.Tensor,    # [Dz]: log-std of random effects
    v_draws: tf.Tensor   # [R0, Dz]: Monte Carlo simulation draws
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute simulated choice probabilities and log-likelihood over T markets.

    This is the core likelihood function used in the M-step. It integrates out
    consumer-level heterogeneity using Monte Carlo draws (simulated maximum likelihood).

    Utility specification for consumer r in market t choosing product j:
        U[t, r, j] = Z[t,j] · (beta_bar + sigma * v[r]) + xi[t,j]

    where sigma = exp(r_vec) is the standard deviation of the random coefficient
    distribution, and v[r] ~ N(0, I) are the simulation draws.

    Unavailable products receive utility -1e9 to be excluded from the denominator.

    Parameters
    ----------
    Z        : [T, M, Dz] encoder output (excludes outside option)
    q        : [T, M+1] choice counts (column 0 = outside option)
    avail    : [T, M] availability mask (1=available)
    xi       : [T, M] = xi_bar[:, None] + eta, combined quality shifters
    beta_bar : [Dz] mean utility coefficients
    r_vec    : [Dz] log-standard-deviations (sigma = exp(r_vec))
    v_draws  : [R0, Dz] fixed simulation draws for Monte Carlo integration

    Returns
    -------
    sigma_t  : tf.Tensor [T, M+1], average simulated choice probabilities
    probs    : tf.Tensor [T, R0, M+1], per-draw choice probabilities (for diagnostics)
    ll_vec   : tf.Tensor [T], market-level log-likelihoods sum_j q[t,j] * log(sigma_t[t,j])
    """
    T = tf.shape(Z)[0]   # int: number of markets
    M = tf.shape(Z)[1]   # int: number of inside products
    R = tf.shape(v_draws)[0]  # int: number of simulation draws

    # Convert log-std to standard deviations and generate per-draw coefficient perturbations
    sigma = tf.exp(r_vec)           # [Dz]: std devs
    resid = v_draws * sigma[None, :]  # [R, Dz]: individual-specific coefficient shifts

    # Mean utility component: Z[t,j] · beta_bar + xi[t,j]
    # delta[t, j] = sum_d Z[t,j,d] * beta_bar[d] + xi[t,j]
    delta = tf.linalg.matvec(Z, beta_bar) + xi  # [T, M]

    # Random utility component: for each draw r, market t, product j:
    # mu[t, r, j] = sum_d resid[r, d] * Z[t, j, d]
    mu = tf.einsum("rd,tmd->trm", resid, Z)  # [T, R, M]

    # Total inside-option utility: broadcast delta [T, 1, M] + mu [T, R, M]
    U_inside = delta[:, None, :] + mu  # [T, R, M]

    # Mask unavailable products by assigning large negative utility
    avail_f = tf.cast(avail, tf.float64)  # [T, M]
    U_inside_masked = U_inside + (-1e9) * (1.0 - avail_f)[:, None, :]  # [T, R, M]

    # Append the outside option utility (normalised to 0) as the first column
    zeros_outside = tf.zeros((T, R, 1), dtype=tf.float64)  # [T, R, 1]
    U_all = tf.concat([zeros_outside, U_inside_masked], axis=2)  # [T, R, M+1]

    # Compute softmax probabilities in numerically stable log-space
    denom = tf.reduce_logsumexp(U_all, axis=2, keepdims=True)  # [T, R, 1]
    probs = tf.exp(U_all - denom)   # [T, R, M+1]: per-draw choice probabilities

    # Average over simulation draws to get the simulated choice probability
    sigma_t = tf.reduce_mean(probs, axis=1)  # [T, M+1]

    # Market-level log-likelihood: sum_j q[t,j] * log(sigma_t[t,j])
    ll_vec = tf.reduce_sum(
        tf.cast(q, tf.float64) * tf.math.log(sigma_t + 1e-300), axis=1
    )  # [T]

    return sigma_t, probs, ll_vec


# ============================================================
# ChoiceDataset Builder
# ============================================================

def build_choice_dataset_from_market_counts(
    Z_np: np.ndarray,            # [T, M, Dz]: encoder embeddings
    q_np: np.ndarray,            # [T, M+1]: choice counts (j=0 is outside option)
    avail_np: np.ndarray,        # [T, M]: availability mask
    market_ids: Optional[np.ndarray] = None  # [T]: optional raw market IDs
) -> ChoiceDataset:
    """
    Convert aggregated market-level choice counts into a ChoiceDataset.

    The BayesianSparseRandomLogit model expects individual choice observations,
    not aggregate counts. This function expands q[t, j] into q[t,j] individual
    observations of consumer choosing product j in market t.

    The outside option (j=0) is prepended with a zero-valued embedding vector,
    so the resulting feature tensor has shape [T, M+1, Dz].

    Parameters
    ----------
    Z_np       : [T, M, Dz] encoder embeddings for inside products.
    q_np       : [T, M+1] choice counts (column 0 = outside option).
    avail_np   : [T, M] binary availability.
    market_ids : optional [T] raw integer market IDs; defaults to 0..T-1.

    Returns
    -------
    ds : ChoiceDataset ready for BayesianSparseRandomLogit.fit().
    """
    T, M, Dz = Z_np.shape  # int, int, int

    # Default market IDs: contiguous 0..T-1
    if market_ids is None:
        market_ids = np.arange(T, dtype=np.int32)  # [T]

    # Prepend a zero embedding for the outside option (j=0)
    Z_all = np.concatenate(
        [np.zeros((T, 1, Dz)), Z_np], axis=1
    )  # [T, M+1, Dz]

    # Prepend availability of 1 for the outside option (always available)
    avail_all = np.concatenate(
        [np.ones((T, 1)), avail_np], axis=1
    ).astype(np.float64)  # [T, M+1]

    # Expand counts into individual observations
    # For each market t and product j with q[t,j] > 0, add q[t,j] rows of choice j
    choices_list, market_list = [], []
    for t in range(T):
        for j in range(M + 1):
            c = int(q_np[t, j])
            if c > 0:
                choices_list.append(np.full(c, j, dtype=np.int32))       # [c]: choice indices
                market_list.append(np.full(c, market_ids[t], dtype=np.int32))  # [c]: market IDs

    # Concatenate all individual observations
    choices = np.concatenate(choices_list, axis=0)        # [N]: individual choices
    market_flat = np.concatenate(market_list, axis=0)     # [N]: market ID per observation

    # Build per-observation feature and availability matrices by indexing into market arrays
    items_feat = Z_all[market_flat]    # [N, M+1, Dz]: features for each observation
    avail_feat = avail_all[market_flat]  # [N, M+1]: availability per observation
    shared_feat = market_flat.reshape(-1, 1).astype(np.float32)  # [N, 1]: market ID column

    # Feature names for the Dz embedding dimensions
    names = [f"z{i}" for i in range(Dz)]

    ds = ChoiceDataset(
        choices=choices,                                    # [N]
        shared_features_by_choice=shared_feat,             # [N, 1]: market ID
        shared_features_by_choice_names=["market_id"],
        items_features_by_choice=items_feat,               # [N, M+1, Dz]
        items_features_by_choice_names=names,
        available_items_by_choice=avail_feat               # [N, M+1]
    )

    return ds

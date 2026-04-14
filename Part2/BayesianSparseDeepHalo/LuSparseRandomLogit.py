"""
Bayesian Sparse Random Logit Model — TFP MCMC

Model Overview
--------------
This model estimates a Bayesian sparse random-coefficient logit model
using a hybrid MCMC sampler. The sparsity prior is a spike-and-slab:
  - gamma[t,j] = 1  => feature j is "active" for market t (drawn from the Slab, variance tau1^2)
  - gamma[t,j] = 0  => feature j is "inactive" (drawn from the Spike, variance tau0^2, tau0 << tau1)

Sampler Strategy
----------------
  - beta_bar, r_vec, xi_bar, eta, phi: Random Walk Metropolis-Hastings (RWMH)
  - gamma: Exact Gibbs sampling (no MH accept/reject step needed)
  - All RWMH step sizes are auto-tuned using SimpleStepSizeAdaptation 
    to hit a target acceptance rate in [0.25, 0.45].
"""

import collections
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from choice_learn.models.base_model import ChoiceModel

# Use float64 globally for numerical stability in MCMC log-probability computations
tf.keras.backend.set_floatx('float64')

tfd = tfp.distributions
tfmcmc = tfp.mcmc

# Graceful fallback if tqdm is not installed
try:
    from tqdm import trange
except ImportError:
    def trange(n, desc=None, leave=True):
        for i in range(n):
            yield i


# =========================================================
# Gamma Gibbs Kernel (Custom TFP TransitionKernel)
# =========================================================

class GammaGibbsKernel(tfmcmc.TransitionKernel):
    """
    Exact Gibbs sampler for the binary spike-and-slab inclusion matrix `gamma`.

    Each element gamma[t, j] \in {0, 1} indicates whether product j's unobserved quality
    (eta[t, j]) in market t is drawn from the Slab (active) or the Spike (inactive).

    Parameters
    ----------
    phi_var : tf.Variable, shape [T_unique]
        Per-market prior inclusion probabilities P(gamma[t,j]=1).
    eta_var : tf.Variable, shape [T_unique, J]
        Current random coefficient values (the Slab draw for each market-product pair).
    tau0 : float
        Standard deviation of the Spike distribution (narrow, near-zero).
    tau1 : float
        Standard deviation of the Slab distribution (wide, informative).
    avail : tf.Variable, shape [J,]
        Availability mask
    """

    def __init__(
        self,
        phi_var: tf.Variable,
        eta_var: tf.Variable,
        tau0: float,
        tau1: float,
        avail_inside: tf.Variable,
    ) -> None:
        self.phi_var = phi_var   # shape: [T_unique]
        self.eta_var = eta_var   # shape: [T_unique, J]
        self.tau0 = tf.cast(tau0, tf.float64)  # Spike std dev (scalar)
        self.tau1 = tf.cast(tau1, tf.float64)  # Slab std dev (scalar)
        self.avail_inside = tf.cast(avail_inside, tf.float64) # availability mask

    @property
    def is_calibrated(self) -> bool:
        # Gibbs sampling draws directly from the target conditional, so it is
        # always perfectly calibrated (acceptance rate = 1.0 by definition).
        return True

    def one_step(
        self,
        current_gamma: tf.Tensor,   # shape: [T_unique, J], dtype=tf.int32
        previous_kernel_results,
        seed=None
    ):
        """
        Execute one Gibbs sweep over the full gamma matrix.

        For each (t, j), compute the posterior probability of gamma[t,j]=1:
            P(gamma=1 | eta, phi) ∝ phi[t] * N(eta[t,j]; 0, tau1^2)
            P(gamma=0 | eta, phi) ∝ (1-phi[t]) * N(eta[t,j]; 0, tau0^2)

        These log-probabilities are then normalised via sigmoid to give prob1.

        Returns
        -------
        new_gamma : tf.Tensor, shape [T_unique, J], dtype=tf.int32
        previous_kernel_results : unchanged (Gibbs needs no bookkeeping)
        """
        # Clip phi to avoid log(0) numerical errors
        phi_safe = tf.clip_by_value(self.phi_var.read_value(), 1e-12, 1.0 - 1e-12)
        eta = self.eta_var.read_value()  # shape: [T_unique, J]

        # Log-probability that feature is ACTIVE (Slab): includes phi prior and Gaussian likelihood
        # phi_safe[:, None] broadcasts [T_unique] -> [T_unique, J]
        log_p1 = (
            tf.math.log(phi_safe)[:, None]
            - 0.5 * (eta ** 2) / self.tau1 ** 2
            - tf.math.log(self.tau1)
        )  # shape: [T_unique, J]

        # Log-probability that feature is INACTIVE (Spike): narrow variance and (1-phi) prior
        log_p0 = (
            tf.math.log(1.0 - phi_safe)[:, None]
            - 0.5 * (eta ** 2) / self.tau0 ** 2
            - tf.math.log(self.tau0)
        )  # shape: [T_unique, J]

        # Posterior probability of being active using numerically stable sigmoid
        # sigmoid(log_p1 - log_p0) = exp(log_p1) / (exp(log_p1) + exp(log_p0))
        prob1 = tf.math.sigmoid(log_p1 - log_p0)  # shape: [T_unique, J]

        # Force probability of inclusion to 0 if item is structurally unavailable
        prob1 = prob1 * self.avail_inside
        
        # Draw uniform noise and compare to prob1 to get a Bernoulli sample
        if seed is None:
            rand_u = tf.random.uniform(tf.shape(prob1), dtype=tf.float64)
        else:
            # Use stateless variant for reproducibility when TFP provides a seed tensor
            rand_u = tf.random.stateless_uniform(tf.shape(prob1), seed=seed, dtype=tf.float64)

        # Cast boolean mask to int32: 1 = active (Slab), 0 = inactive (Spike)
        new_gamma = tf.cast(rand_u < prob1, tf.int32)  # shape: [T_unique, J]
        return new_gamma, previous_kernel_results

    def bootstrap_results(self, init_state):
        # Gibbs needs no stored state (no log-prob tracking, no step size)
        return collections.namedtuple("GammaKernelResults", [])()


# =========================================================
# Main Model
# =========================================================

class BayesianSparseRandomLogit(ChoiceModel):
    """
    Bayesian Sparse Random Logit Model estimated via hybrid MCMC.

    Utility specification for consumer i in market t choosing product j:
        U_{i,t,j} = sum_d beta_bar[d] * x[t,j,d]
                  + sum_d (sigma[d] * v_draw * x[t,j,d])   [random coefficients]
                  + xi_bar[t] + eta[t,j] * gamma[t,j]       [market intercept + market product sparse shocks ]
                  + eps_{i,t,j}                             [Gumbel noise -> Logit]

    Parameters
    ----------
    sim_params  : dataclass containing T, J, D (markets, products, features).
    mcmc_params : dataclass containing all priors and MCMC settings (burnin, G, taus, etc.).
    random_coef_mask : array, shape [D], binary mask indicating which features get random coefficients.
    seed        : int, random seed for reproducibility.
    verbose     : bool, print diagnostics after fitting.
    """

    def __init__(
        self,
        sim_params,
        mcmc_params,
        random_coef_mask: Optional[np.ndarray] = None,
        seed: int = 123,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer="adam")
        self.sim_params = sim_params
        self.mcmc_params = mcmc_params
        self.seed = seed
        self.verbose = verbose

        # Dimensions from sim_params
        self.T = sim_params.T    # int: total number of choice situations / markets
        self.J = sim_params.J    # int: number of inside products (excluding outside option)
        self.D = sim_params.D    # int: number of product feature dimensions

        # Determine which coefficients are random vs. fixed
        if random_coef_mask is None:
            random_coef_mask = mcmc_params.random_coef_mask
        self.random_coef_mask = tf.constant(random_coef_mask, dtype=tf.float64)  # shape: [D]
        self.n_random = int(np.sum(np.asarray(random_coef_mask)))                # int: number of random coefs
        self.random_indices = tf.where(self.random_coef_mask > 0.5)[:, 0]       # shape: [n_random]
        self.fixed_indices = tf.where(self.random_coef_mask < 0.5)[:, 0]        # shape: [D - n_random]

        # These are populated during _initialize_from_dataset()
        self._is_initialized = False
        self.unique_counts = None          # tf.Tensor [T_unique, J+1]: choice counts per market
        self.unique_items_features = None  # tf.Tensor [T_unique, J+1, D]: per-market feature matrices
        self.unique_avail = None           # tf.Tensor [T_unique, J+1]: per-market availability 
        self.avail_inside = None           # tf.Tensor [T_unique, J]: per-market availability for inside options 
        self._market_id_map = None         # tf.Tensor [max_market_id+1]: maps raw market id -> contiguous index
        self._T_unique = None              # int: number of unique markets

        # MCMC state variables (initialised in _initialize_state)
        self.beta_bar = self.r_vec = self.xi_bar = None
        self.eta = self.phi = self.gamma = None
        self.z_phi = self.v_draws = None

        # Posterior sample storage (filled after burn-in)
        self.samples_ = None
        self.kernel_diagnostics_ = {}

        # step sizes as plain tf.Variables — all adapted via Robbins-Monro
        self._step_vars = {}

        print(f"--- Starting Bayesian Shrinkage Random Logit ---")

    @property
    def trainable_weights(self):
        # Return all non-None MCMC state variables for ChoiceModel compatibility
        return [w for w in [self.beta_bar, self.r_vec, self.xi_bar,
                            self.eta, self.phi, self.gamma] if w is not None]

    # ─────────────────────────────────────────────
    # ChoiceModel API
    # ─────────────────────────────────────────────

    def compute_batch_utility(
        self,
        shared_features_by_choice,   # array [N] or [N, ...] with market_id in first column
        items_features_by_choice,    # array [N, J+1, D]
        available_items_by_choice,   # array [N, J+1] binary availability mask
        choices=None,
    ):
        """
        ChoiceModel-required utility computation for a batch of N observations.
        Uses posterior means for prediction.

        Returns
        -------
        util : tf.Tensor, shape [N, J+1], dtype=float64
            Utility scores including the outside option (index 0).
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call fit(choice_dataset) first.")

        X = tf.cast(items_features_by_choice, tf.float64)  # [N, J+1, D]
        N = tf.shape(X)[0]
        J_plus = tf.shape(X)[1]

        # Extract market IDs from shared features
        s = tf.convert_to_tensor(shared_features_by_choice)
        if len(s.shape) == 1:
            market_ids = tf.cast(s, tf.int32)
        else:
            market_ids = tf.cast(s[:, 0], tf.int32)

        # Map raw market IDs to contiguous indices 0..T_unique-1
        if isinstance(self._market_id_map, tf.Tensor):
            mid = tf.gather(self._market_id_map, market_ids)  # [N]
        else:
            mid = market_ids  # fallback if already contiguous

        # Fixed utility: sum over fixed-coefficient dimensions
        fixed_util = tf.zeros((N, J_plus), dtype=tf.float64)
        for idx in self.fixed_indices.numpy():
            fixed_util += self.beta_bar[idx] * X[:, :, idx]  # [N, J+1]

        # Random utility: use posterior mean beta_bar (ignores individual heterogeneity)
        random_mean_util = tf.zeros((N, J_plus), dtype=tf.float64)
        for idx in self.random_indices.numpy():
            random_mean_util += self.beta_bar[idx] * X[:, :, idx]  # [N, J+1]

        # Product-level quality shifters: xi = xi_bar[t] + eta[t,j] * gamma[t,j]
        xi_bar_n = tf.gather(self.xi_bar, mid)   # [N]
        eta_n = tf.gather(self.eta, mid)          # [N, J]
        xi_inside = xi_bar_n[:, None] + eta_n     # [N, J]
        # Pad with 0 for the outside option (j=0)
        xi_padded = tf.concat([tf.zeros((N, 1), tf.float64), xi_inside], axis=1)  # [N, J+1]

        util = fixed_util + random_mean_util + xi_padded  # [N, J+1]

        # Zero out unavailable products with a large negative utility
        if available_items_by_choice is not None:
            avail = tf.cast(available_items_by_choice, tf.float64)
            util = tf.where(avail > 0.5, util, tf.constant(-1e9, tf.float64))

        return util

    # ─────────────────────────────────────────────
    # Fit
    # ─────────────────────────────────────────────

    def fit(self, choice_dataset, store_samples: bool = True):
        """
        Run the hybrid MCMC sampler for G total iterations.
        After burn-in, posterior samples for all parameters are stored in self.samples_.

        Parameters
        ----------
        choice_dataset : ChoiceDataset with items_features, shared_features (market_id), availability, choices.
        store_samples  : bool, whether to store post-burn posterior draws.

        Returns
        -------
        losses_history : dict with key "train_loss" -> list of negative log-likelihoods per iteration.
        """
        assert self.mcmc_params.burn < self.mcmc_params.G, (
            f"burn ({self.mcmc_params.burn}) must be < G ({self.mcmc_params.G}). "
            f"No post-burn samples would be stored."
        )

        # Build unique-market aggregated representation from raw dataset
        self._initialize_from_dataset(choice_dataset)
        # Initialise all MCMC state tf.Variables to starting values
        self._initialize_state()
        # Construct all RWMH and Gibbs kernel objects
        self._build_kernels()

        losses_history = {"train_loss": []}

        # Storage for posterior samples (only post burn-in)
        samples = None
        if store_samples:
            samples = {k: [] for k in ["beta_bar", "r_vec", "xi_bar", "eta", "gamma", "phi"]}

        # Compute initial negative log-likelihood before any MCMC moves
        xi = self.xi_bar[:, None] + self.eta
        _, _, ll_vec = self.compute_log_likelihood_unique(xi, self.beta_bar, self.r_vec, self.v_draws)
        current_nll = float(-tf.reduce_sum(ll_vec).numpy())
        losses_history["train_loss"].append(current_nll)

        # Acceptance counters for monitoring chain health
        acc = dict(beta=0, r=0, xi=0, eta=0, phi=0, gamma=0)
        G = self.mcmc_params.G
        burnin = self.mcmc_params.burn

        # Bootstrap initial kernel result states (required by TFP's SSSA interface)
        kr_beta = self.beta_kernel.bootstrap_results(self.beta_bar.read_value())
        kr_r = self.r_kernel.bootstrap_results(self.r_vec.read_value())
        kr_xi = self.xi_kernel.bootstrap_results(self.xi_bar.read_value())
        kr_eta = self.eta_kernel.bootstrap_results(self.eta.read_value())
        kr_phi = self.phi_kernel.bootstrap_results(self.z_phi.read_value())

        for g in trange(G, desc="MCMC chain"):
            # One full Gibbs sweep: update all parameters in sequence
            kr_beta, kr_r, kr_xi, kr_eta, kr_phi, ll_vec, accepts = self._mcmc_step(
                kr_beta, kr_r, kr_xi, kr_eta, kr_phi
            )

            # Accumulate acceptance counts for each parameter block
            for k in acc:
                acc[k] += int(accepts[k].numpy())

            # Record current negative log-likelihood as the training loss proxy
            nll = float(-tf.reduce_sum(ll_vec).numpy())
            losses_history["train_loss"].append(nll)

            # Store posterior sample after burn-in phase
            if store_samples and g >= burnin:
                samples["beta_bar"].append(self.beta_bar.numpy())
                samples["r_vec"].append(self.r_vec.numpy())
                samples["xi_bar"].append(self.xi_bar.numpy())
                samples["eta"].append(self.eta.numpy())
                samples["gamma"].append(self.gamma.numpy())
                samples["phi"].append(self.phi.numpy())

        # Convert lists to stacked numpy arrays: shape [n_post_samples, ...]
        self.samples_ = ({k: np.asarray(v) for k, v in samples.items()} if store_samples else None)

        if self.verbose:
            T_u = self._T_unique
            print(f"Final beta: {self.beta_bar.numpy()}")
            print(f"Final sigma: {np.exp(self.r_vec.numpy())}")
            print(
                f"Accept beta={acc['beta']/G:.3f} r={acc['r']/G:.3f} "
                f"xi={acc['xi']/(G*T_u):.3f} "
                f"eta={acc['eta']/(G*T_u):.3f} "
                f"phi={acc['phi']/G:.3f} "
                f"gamma=1 (Gibbs) "
            )
            print("Step sizes:")
            for k, v in self._step_vars.items():
                if isinstance(v, list):
                    vals = [vv.numpy() for vv in v]
                    print(f"  step_{k} = avg {np.mean(vals):.6f}")
                else:
                    print(f"  step_{k} = {v.numpy():.6f}")

        return losses_history

    def _mcmc_step(self, kr_beta, kr_r, kr_xi, kr_eta, kr_phi):
        """
        Execute one complete hybrid MCMC sweep updating all parameter blocks.
        Order: beta_bar -> r_vec -> xi_bar -> eta -> phi -> gamma
        Returns updated kernel result objects, current log-likelihoods, and acceptance flags.
        """
        kr_beta, acc_beta = self._step_beta(kr_beta)
        kr_r, acc_r = self._step_r(kr_r)
        kr_xi, acc_xi = self._step_xi(kr_xi)

        # Update eta as a single joint block [T_unique, J] rather than element-wise
        kr_eta, acc_eta = self._step_eta(kr_eta)

        kr_phi, acc_phi = self._step_phi(kr_phi)
        # Gamma uses Gibbs (always accepted); returns count of changed indicators
        acc_gamma = self._step_gamma()
        ll_vec = self._compute_final_ll()  # shape: [T_unique]

        accepts = dict(beta=acc_beta, r=acc_r, xi=acc_xi,
                       eta=acc_eta, phi=acc_phi, gamma=acc_gamma)
        return kr_beta, kr_r, kr_xi, kr_eta, kr_phi, ll_vec, accepts

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_beta(self, current_kr):
        """
        RWMH update for beta_bar (mean fixed + random coefficients), shape [D].

        Refreshes the inner kernel log-prob to avoid stale cached values,
        then calls one_step which proposes a Normal perturbation and accepts/rejects.
        """
        fresh_inner = self.beta_kernel.inner_kernel.bootstrap_results(self.beta_bar.read_value())
        updated_kr = current_kr._replace(inner_results=fresh_inner)
        new_beta, next_kr = self.beta_kernel.one_step(self.beta_bar.read_value(), updated_kr)
        self.beta_bar.assign(new_beta)
        return next_kr, tf.cast(next_kr.inner_results.is_accepted, tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_r(self, current_kr):
        """
        RWMH update for r_vec (log standard deviations of random coefficients), shape [n_random].

        Parameterising as log(sigma) ensures sigma > 0 without constraints.
        """
        fresh_inner = self.r_kernel.inner_kernel.bootstrap_results(self.r_vec.read_value())
        updated_kr = current_kr._replace(inner_results=fresh_inner)
        new_r, next_kr = self.r_kernel.one_step(self.r_vec.read_value(), updated_kr)
        self.r_vec.assign(new_r)
        return next_kr, tf.cast(next_kr.inner_results.is_accepted, tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_xi(self, current_kr):
        """
        RWMH update for xi_bar (market-level mean quality shifter), shape [T_unique].

        Each market gets its own scalar draw; acceptance is tracked as a sum across markets.
        """
        fresh_inner = self.xi_kernel.inner_kernel.bootstrap_results(self.xi_bar.read_value())
        updated_kr = current_kr._replace(inner_results=fresh_inner)
        new_xi, next_kr = self.xi_kernel.one_step(self.xi_bar.read_value(), updated_kr)
        self.xi_bar.assign(new_xi)
        return next_kr, tf.cast(tf.reduce_sum(tf.cast(next_kr.inner_results.is_accepted, tf.int32)), tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_eta(self, current_kr):
        """
        RWMH update for eta (product-level quality deviations), shape [T_unique, J].

        Proposal scale is proportional to the spike/slab variance: wider proposals
        for active features (gamma=1, tau1) and narrower for inactive ones (gamma=0, tau0).
        """
        fresh_inner = self.eta_kernel.inner_kernel.bootstrap_results(self.eta.read_value())
        updated_kr = current_kr._replace(inner_results=fresh_inner)
        new_eta, next_kr = self.eta_kernel.one_step(self.eta.read_value(), updated_kr)
        self.eta.assign(new_eta)
        return next_kr, tf.cast(tf.reduce_sum(tf.cast(next_kr.inner_results.is_accepted, tf.int32)), tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_phi(self, current_kr):
        """
        RWMH update for phi (per-market Slab inclusion probability), shape [T_unique].

        Operates in unconstrained logit space (z_phi = logit(phi)) for the random walk,
        then maps back to (0,1) via sigmoid to update self.phi.
        """
        fresh_inner = self.phi_kernel.inner_kernel.bootstrap_results(self.z_phi.read_value())
        updated_kr = current_kr._replace(inner_results=fresh_inner)
        new_z_phi, next_kr = self.phi_kernel.one_step(self.z_phi.read_value(), updated_kr)
        self.z_phi.assign(new_z_phi)
        # Sync the constrained phi variable with the new logit value
        self.phi.assign(tf.math.sigmoid(new_z_phi))
        return next_kr, tf.cast(next_kr.inner_results.is_accepted, tf.int32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _step_gamma(self):
        """
        Gibbs update for gamma (spike-and-slab binary indicators), shape [T_unique, J].

        Counts the number of indicator flips as a proxy for chain activity.
        Gibbs sampling always accepts, so no acceptance ratio is needed.
        """
        gamma_kr = self.gamma_kernel.bootstrap_results(self.gamma.read_value())
        new_gamma, _ = self.gamma_kernel.one_step(self.gamma.read_value(), gamma_kr)
        # Count number of indicators that changed (0->1 or 1->0) this step
        acc_gamma = tf.reduce_sum(tf.cast(tf.not_equal(new_gamma, self.gamma.read_value()), tf.int32))
        self.gamma.assign(new_gamma)
        return acc_gamma

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _compute_final_ll(self):
        """
        Compute the market-level log-likelihood after all parameter updates in one sweep.

        Returns
        -------
        ll_vec : tf.Tensor, shape [T_unique], dtype=float64
            Log-likelihood contribution from each unique market.
        """
        xi_final = self.xi_bar.read_value()[:, None] + self.eta.read_value()
        _, _, ll_vec = self.compute_log_likelihood_unique(
            xi_final, self.beta_bar.read_value(), self.r_vec.read_value(), self.v_draws
        )
        return ll_vec

    # ─────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────

    def _initialize_from_dataset(self, dataset) -> None:
        """
        Pre-process the raw ChoiceDataset into a unique-market aggregated representation.

        Attributes:
        - self.unique_counts: shape [T_unique, J+1], choice frequency per market
        - self.unique_items_features: shape [T_unique, J+1, D], feature matrix per market
        - self._market_id_map: lookup tensor for raw market ID -> contiguous index
        - self._T_unique: number of unique markets
        """
        assert dataset.items_features_by_choice is not None, "items_features_by_choice is required"
        assert dataset.shared_features_by_choice is not None, "shared_features_by_choice is required"
        assert dataset.available_items_by_choice is not None, "available_items_by_choice is required"

        shared_np = dataset.shared_features_by_choice
        if isinstance(shared_np, (list, tuple)):
            shared_np = shared_np[0]
        shared_np = np.asarray(shared_np)

        # Extract market IDs from shared features (first column or flat array)
        if shared_np.ndim == 1:
            market_ids_full = shared_np.astype(int)
        else:
            market_ids_full = shared_np[:, 0].astype(int)
        market_ids_full = market_ids_full.flatten()

        choices_full = np.asarray(dataset.choices).astype(int)

        # Identify unique markets and record the first occurrence index for feature extraction
        unique_market_ids, first_indices = np.unique(market_ids_full, return_index=True)
        self._T_unique = len(unique_market_ids)  # int: number of unique markets

        # Build a fast lookup array: raw_market_id -> contiguous index
        max_mid = int(np.max(unique_market_ids))
        lookup = -np.ones((max_mid + 1,), dtype=np.int32)
        lookup[unique_market_ids] = np.arange(self._T_unique, dtype=np.int32)
        self._market_id_map = tf.constant(lookup, dtype=tf.int32)

        # Aggregate choice counts: counts[t, j] = number of times product j was chosen in market t
        mid_contig = lookup[market_ids_full]
        pairs = np.stack([mid_contig, choices_full], axis=1)
        counts = np.zeros((self._T_unique, self.J + 1), dtype=np.int32)
        np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)
        self.unique_counts = tf.constant(counts, dtype=tf.float64)  # [T_unique, J+1]

        # Store a single feature matrix per unique market (using first occurrence)
        items_full = dataset.items_features_by_choice
        if isinstance(items_full, (list, tuple)):
            items_full = items_full[0]

        if isinstance(items_full, tf.Tensor):
            unique_feats = tf.gather(items_full, first_indices)
        else:
            unique_feats = tf.constant(np.asarray(items_full)[first_indices], dtype=tf.float64)

        self.unique_items_features = tf.cast(unique_feats, tf.float64)  # [T_unique, J+1, D]

        # Extract availability mask
        avail_full = dataset.available_items_by_choice
        if isinstance(avail_full, (list, tuple)):
            avail_full = avail_full[0]

        if isinstance(avail_full, tf.Tensor):
            unique_avail = tf.gather(avail_full, first_indices)
        else:
            unique_avail = tf.constant(np.asarray(avail_full)[first_indices], dtype=tf.float64)

        self.unique_avail = tf.cast(unique_avail, tf.float64)
        # Drop the outside option (j=0) since eta only applies to inside goods
        self.avail_inside = self.unique_avail[:, 1:]
        
        self._is_initialized = True

    def _initialize_state(self) -> None:
        """
        Initialise all MCMC tf.Variables to their starting values.

        All continuous parameters start at zero / 0.5.
        Monte Carlo integration draws (v_draws) are fixed for the entire chain run.
        """
        tf.random.set_seed(self.seed)

        self.beta_bar = tf.Variable(tf.zeros((self.D,), dtype=tf.float64))                            # [D]
        self.r_vec = tf.Variable(tf.zeros((self.n_random,), dtype=tf.float64))                        # [n_random]
        self.xi_bar = tf.Variable(tf.zeros((self._T_unique,), dtype=tf.float64))                      # [T_unique]
        self.eta = tf.Variable(tf.zeros((self._T_unique, self.J), dtype=tf.float64))                  # [T_unique, J]
        self.phi = tf.Variable(tf.fill((self._T_unique,), tf.constant(0.5, tf.float64)))              # [T_unique]
        self.gamma = tf.Variable(tf.zeros((self._T_unique, self.J), dtype=tf.int32))                  # [T_unique, J]

        # z_phi is the logit-transformed phi; used as the unconstrained state for the RWMH kernel
        phi0 = tf.clip_by_value(self.phi.read_value(), 1e-6, 1.0 - 1e-6)
        self.z_phi = tf.Variable(tf.math.log(phi0) - tf.math.log(1.0 - phi0))                        # [T_unique]

        # Fixed Monte Carlo draws for integrating out random coefficients during likelihood evaluation
        # v_draws ~ N(0, I), shape [R0, n_random], held fixed throughout the chain
        self.v_draws = tf.random.normal((self.mcmc_params.R0, self.n_random), dtype=tf.float64)       # [R0, n_random]

    # ─────────────────────────────────────────────
    # Likelihood
    # ─────────────────────────────────────────────

    def compute_batch_utility_unique(
        self,
        unique_items: tf.Tensor,   # [T_unique, J+1, D]
        xi: tf.Tensor,             # [T_unique, J] inside-product quality
        beta_bar: tf.Tensor,       # [D]
        r_vec: tf.Tensor,          # [n_random]
        v_draws: tf.Tensor         # [R0, n_random]
    ) -> tf.Tensor:
        """
        Compute the mixed logit utility matrix for all unique markets × simulation draws.

        For each market t and draw r:
            U[t, r, j] = sum_{d fixed} beta_bar[d] * x[t,j,d]
                        + sum_{d random} (beta_bar[d] + exp(r[d]) * v[r,d]) * x[t,j,d]
                        + xi[t, j]   (with xi[t, 0] = 0 for outside option)

        Returns
        -------
        U : tf.Tensor, shape [T_unique, R0, J+1], dtype=float64
        """
        R = tf.shape(v_draws)[0]      # int: number of Monte Carlo draws
        T = tf.shape(unique_items)[0]  # int: T_unique
        J_plus = tf.shape(unique_items)[1]  # int: J+1

        # Fixed coefficient contribution (no simulation needed)
        fixed_util = tf.zeros((T, J_plus), dtype=tf.float64)
        for idx in self.fixed_indices.numpy():
            fixed_util += beta_bar[idx] * unique_items[:, :, idx]  # [T, J+1]

        # Random coefficient contribution (requires Monte Carlo integration)
        random_util = tf.zeros((T, R, J_plus), dtype=tf.float64)
        for i, idx in enumerate(self.random_indices.numpy()):
            sigma_d = tf.exp(r_vec[i])                             # scalar: std dev of coefficient d
            beta_draws = beta_bar[idx] + sigma_d * v_draws[:, i]  # [R]: individual-specific draws
            # Broadcast outer product: [T, J+1] * [R] -> [T, R, J+1]
            random_util += beta_draws[None, :, None] * unique_items[:, :, idx][:, None, :]

        # Quality shifters: pad xi with 0 for outside option column
        xi_padded = tf.concat([tf.zeros((T, 1), tf.float64), xi], axis=1)  # [T, J+1]
        u_xi = xi_padded[:, None, :]                                       # [T, 1, J+1] for broadcasting

        U = fixed_util[:, None, :] + random_util + u_xi

        # Mask unavailable items so they don't incorrectly receive likelihood mass
        if self.unique_avail is not None:
            U = tf.where(self.unique_avail[:, None, :] > 0.5, U, tf.constant(-1e9, tf.float64))
            
        return U  # [T, R, J+1]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compute_log_likelihood_unique(
        self,
        xi: tf.Tensor,        # [T_unique, J]
        beta_bar: tf.Tensor,  # [D]
        r_vec: tf.Tensor,     # [n_random]
        v_draws: tf.Tensor    # [R0, n_random]
    ):
        """
        Compute the simulated log-likelihood aggregated over unique markets.

        Steps:
        1. Compute raw utilities U [T, R, J+1].
        2. Apply softmax across products to get choice probabilities, then average over draws (R).
        3. Weight by empirical choice counts and sum to get the market-level log-likelihood.

        Returns
        -------
        sigma : tf.Tensor [T_unique, J+1], average predicted choice probabilities
        None  : placeholder (reserved for future diagnostics)
        ll_vec: tf.Tensor [T_unique], per-market log-likelihood contributions
        """
        U = self.compute_batch_utility_unique(
            self.unique_items_features, xi, beta_bar, r_vec, v_draws
        )  # [T, R, J+1]

        # Simulate choice probabilities by averaging softmax over Monte Carlo draws
        sigma = tf.reduce_mean(
            tf.exp(U - tf.reduce_logsumexp(U, axis=2, keepdims=True)), axis=1
        )  # [T, J+1]

        # Weighted log-likelihood: counts * log(predicted shares)
        ll_vec = tf.reduce_sum(
            self.unique_counts * tf.math.log(sigma + 1e-300), axis=1
        )  # [T]

        return sigma, None, ll_vec

    # ─────────────────────────────────────────────
    # Kernel Building
    # ─────────────────────────────────────────────

    def _build_kernels(self) -> None:
        """
        Construct all MCMC transition kernels.

        For continuous parameters (beta_bar, r_vec, xi_bar, eta, phi):
            Random Walk Metropolis-Hastings (RWMH) wrapped in SimpleStepSizeAdaptation.
            The adaptation tunes step sizes during burn-in
            so that the acceptance rates converge to `target_accept` (default 0.35).

        For discrete parameters (gamma):
            Exact Gibbs sampling via GammaGibbsKernel.
        """
        # Initialise per-parameter step sizes from mcmc_params, falling back to defaults
        for name, attr, default in [
            ("beta", "step_beta", 0.5), ("r", "step_r", 0.3),
            ("xi", "step_xibar", 0.2), ("phi", "step_phi", 0.1), ("eta", "step_eta", 0.1),
        ]:
            val = float(getattr(self.mcmc_params, attr, default))
            # Store as tf.Variable so Robbins-Monro can update them in-place
            self._step_vars[name] = tf.Variable(tf.cast(val, tf.float64), trainable=False)

        # Helper closures: link TFP's SimpleStepSizeAdaptation to our tf.Variable step sizes
        def make_setter(step_var: tf.Variable):
            """Returns a setter function that assigns the adapted step size to step_var."""
            def setter(kr, new_step_size):
                step_var.assign(tf.cast(new_step_size, tf.float64))
                return kr
            return setter

        def make_getter(step_var: tf.Variable):
            """Returns a getter function that reads the current step size from step_var."""
            def getter(kr):
                return tf.cast(step_var.read_value(), tf.float64)
            return getter

        target_accept = float(getattr(self.mcmc_params, 'target_accept', 0.35))
        burnin = int(getattr(self.mcmc_params, 'burn', 0))

        # ── beta_bar ──
        def beta_target(beta_val: tf.Tensor) -> tf.Tensor:
            """
            Log-posterior for beta_bar [D]: log-likelihood + Gaussian prior N(0, V_beta).
            """
            xi = self.xi_bar.read_value()[:, None] + self.eta.read_value()
            _, _, ll = self.compute_log_likelihood_unique(xi, beta_val, self.r_vec.read_value(), self.v_draws)
            V_beta = tf.cast(self.mcmc_params.V_beta, tf.float64)
            return tf.reduce_sum(ll) - 0.5 * tf.reduce_sum(beta_val ** 2) / V_beta

        self.beta_kernel = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfmcmc.RandomWalkMetropolis(
                target_log_prob_fn=beta_target,
                new_state_fn=tfmcmc.random_walk_normal_fn(scale=self._step_vars["beta"])
            ),
            num_adaptation_steps=burnin, target_accept_prob=target_accept,
            step_size_setter_fn=make_setter(self._step_vars["beta"]),
            step_size_getter_fn=make_getter(self._step_vars["beta"])
        )

        # ── r_vec ──
        def r_target(r_val: tf.Tensor) -> tf.Tensor:
            """
            Log-posterior for r_vec [n_random] (log-stdevs): log-likelihood + Gaussian prior N(0, V_r).
            """
            xi = self.xi_bar.read_value()[:, None] + self.eta.read_value()
            _, _, ll = self.compute_log_likelihood_unique(xi, self.beta_bar.read_value(), r_val, self.v_draws)
            V_r = tf.cast(self.mcmc_params.V_r, tf.float64)
            return tf.reduce_sum(ll) - 0.5 * tf.reduce_sum(r_val ** 2) / V_r

        self.r_kernel = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfmcmc.RandomWalkMetropolis(
                target_log_prob_fn=r_target,
                new_state_fn=tfmcmc.random_walk_normal_fn(scale=self._step_vars["r"])
            ),
            num_adaptation_steps=burnin, target_accept_prob=target_accept,
            step_size_setter_fn=make_setter(self._step_vars["r"]),
            step_size_getter_fn=make_getter(self._step_vars["r"])
        )

        # ── xi_bar ──
        def xi_target(xi_val: tf.Tensor) -> tf.Tensor:
            """
            Log-posterior for xi_bar [T_unique]: log-likelihood (per market) + Gaussian prior N(0, V_xibar).
            Returns a vector [T_unique] so each market can be accepted/rejected independently.
            """
            xi_full = xi_val[:, None] + self.eta.read_value()  # [T_unique, J]
            _, _, ll = self.compute_log_likelihood_unique(xi_full, self.beta_bar.read_value(), self.r_vec.read_value(), self.v_draws)
            V_xibar = tf.cast(self.mcmc_params.V_xibar, tf.float64)
            return ll - 0.5 * (xi_val ** 2) / V_xibar  # [T_unique]

        self.xi_kernel = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfmcmc.RandomWalkMetropolis(
                target_log_prob_fn=xi_target,
                new_state_fn=tfmcmc.random_walk_normal_fn(scale=self._step_vars["xi"])
            ),
            num_adaptation_steps=burnin, target_accept_prob=target_accept,
            step_size_setter_fn=make_setter(self._step_vars["xi"]),
            step_size_getter_fn=make_getter(self._step_vars["xi"])
        )

        # ── eta ──
        tau0 = tf.cast(self.mcmc_params.tau0, tf.float64)  # Spike std dev
        tau1 = tf.cast(self.mcmc_params.tau1, tf.float64)  # Slab std dev

        def eta_target(eta_val: tf.Tensor) -> tf.Tensor:
            """
            Log-posterior for eta [T_unique, J]: log-likelihood + spike-and-slab Gaussian prior.
            The prior variance of eta[t,j] is tau1^2 if gamma[t,j]=1 (Slab) else tau0^2 (Spike).
            Returns [T_unique] market-level log-posteriors.
            """

            # Unavailable items are rigidly zeroed out. 
            # We do not mask gamma=0 here, so the tau0 prior can properly penalize and shrink them.
            eta_valid = eta_val * self.avail_inside

            xi_full = self.xi_bar.read_value()[:, None] + eta_valid  # [T_unique, J]
            _, _, ll = self.compute_log_likelihood_unique(xi_full, self.beta_bar.read_value(), self.r_vec.read_value(), self.v_draws)

            gamma_val = self.gamma.read_value()
            # Per-element variance: tau1^2 if active, tau0^2 if inactive
            var_mat = tf.where(gamma_val == 1, tau1 ** 2, tau0 ** 2)  # [T_unique, J]

            # Sum prior log-prob over J products for each market T (Prior log prob only applied to available items)
            prior_lp = tf.reduce_sum(-0.5 * (eta_valid ** 2) / var_mat * self.avail_inside, axis=1)  # [T_unique]
            return ll + prior_lp

        def eta_new_state(state_parts, seed):
            """
            Custom RWMH proposal for eta: scale the random walk by spike/slab size.
            Active features (gamma=1) get larger proposals; inactive (gamma=0) get smaller.
            """
            step_eta = tf.cast(self._step_vars["eta"], tf.float64)
            gamma_val = tf.cast(self.gamma.read_value(), tf.float64)
            # Scale proposal noise proportionally to tau0/tau1 based on current gamma
            sigma_prop = step_eta * tf.where(
                gamma_val == 0.,
                tf.constant(tau0, tf.float64),
                tf.constant(tau1, tf.float64)
            )  # [T_unique, J]

            # Structurally unavailable items have a jump size of exactly 0.0
            sigma_prop = sigma_prop * self.avail_inside
            
            return tf.nest.map_structure(
                lambda s: s + tf.random.stateless_normal(
                    tf.shape(s), stddev=sigma_prop, dtype=tf.float64, seed=seed
                ),
                state_parts
            )

        self.eta_kernel = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfmcmc.RandomWalkMetropolis(
                target_log_prob_fn=eta_target,
                new_state_fn=eta_new_state
            ),
            num_adaptation_steps=burnin, target_accept_prob=target_accept,
            step_size_setter_fn=make_setter(self._step_vars["eta"]),
            step_size_getter_fn=make_getter(self._step_vars["eta"])
        )

        # ── phi (in logit space) ──
        def phi_target(z_phi_val: tf.Tensor) -> tf.Tensor:
            """
            Log-posterior for z_phi = logit(phi) [T_unique].

            Maps back to phi \in (0,1) via sigmoid, then evaluates the conditionally
            conjugate Beta-Binomial posterior: Beta(a_phi + sum(gamma), b_phi + J - sum(gamma)).
            """
            phi_val = tf.math.sigmoid(z_phi_val)  # [T_unique], constrained to (0,1)
            a_phi = tf.cast(self.mcmc_params.a_phi, tf.float64)
            b_phi = tf.cast(self.mcmc_params.b_phi, tf.float64)

            # Sufficient statistic: how many features are active per market
            sum_gamma = tf.reduce_sum(tf.cast(self.gamma.read_value(), tf.float64), axis=1)  # [T_unique]

            # Calculate the true number of available inside items per market
            J_avail = tf.reduce_sum(self.avail_inside, axis=1)
            
            # Conditionally conjugate posterior: Beta(a + sum_gamma, b + J_avail - sum_gamma)
            beta_dist = tfd.Beta(a_phi + sum_gamma, b_phi + J_avail - sum_gamma)

            # Log-posterior = log Beta density + log|d phi / d z_phi| (Jacobian of sigmoid)
            lp = (
                beta_dist.log_prob(phi_val)
                + tf.math.log(phi_val + 1e-300)
                + tf.math.log(1.0 - phi_val + 1e-300)
            )  # [T_unique]
            return tf.reduce_sum(lp)  # scalar

        self.phi_kernel = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=tfmcmc.RandomWalkMetropolis(
                target_log_prob_fn=phi_target,
                new_state_fn=tfmcmc.random_walk_normal_fn(scale=self._step_vars["phi"])
            ),
            num_adaptation_steps=burnin, target_accept_prob=target_accept,
            step_size_setter_fn=make_setter(self._step_vars["phi"]),
            step_size_getter_fn=make_getter(self._step_vars["phi"])
        )

        # ── gamma (Gibbs, no RWMH) ──
        # Gamma is updated by the GammaGibbsKernel which samples directly from the
        # Bernoulli conditional distribution 
        self.gamma_kernel = GammaGibbsKernel(self.phi, self.eta, tau0, tau1, self.avail_inside)

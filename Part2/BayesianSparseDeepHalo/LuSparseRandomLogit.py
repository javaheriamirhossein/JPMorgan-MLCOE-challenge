"""
Bayesian Sparse Market Product Shock Model with ChoiceModel
Based on Lu & Shimizu 2025

This model is using random walk/tailored metropolis hasting update for beta and random walk metropolis hasting for other variables 
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from choice_learn.models.base_model import ChoiceModel


tf.keras.backend.set_floatx('float64')

# tqdm for progress bar
try:
    from tqdm import trange, tqdm
except ImportError:
    def trange(n, desc=None, leave=True):
        print(f"Starting {desc}...")
        for i in range(n):
            yield i
    tqdm = None


# =====================================
# LU's SPARSE BAYESIAN MODEL CLASS (with ChoiceLearn)
# =====================================

class BayesianSparseRandomLogit(ChoiceModel):
    """
    Bayesian Sparse Random Logit model via ChoiceModel.

    ChoiceModel API:
      - compute_batch_utility(shared_features_by_choice, items_features_by_choice,
                              available_items_by_choice, choices)
      - fit(choice_dataset) -> losses_history with losses_history["train_loss"]
      - trainable_weights property
    """

    def __init__(
        self,
        sim_params,
        mcmc_params,
        random_coef_mask=None,
        seed=123,
        beta_method='rwmh',
        verbose=False,
    ):
        super().__init__(optimizer="adam")
        self.sim_params = sim_params
        self.mcmc_params = mcmc_params
        self.seed = seed
        self.beta_method = beta_method   # 'rwmh' (step_beta) or 'tmh' (kappa_beta)
        self.verbose = verbose

        self.T = sim_params.T
        self.J = sim_params.J
        self.D = sim_params.D

        if random_coef_mask is None:
            random_coef_mask = mcmc_params.random_coef_mask
        self.random_coef_mask = tf.constant(random_coef_mask, dtype=tf.float64)

        self.n_random = int(np.sum(np.array(random_coef_mask)))
        self.random_indices = tf.where(self.random_coef_mask > 0.5)[:, 0]
        self.fixed_indices = tf.where(self.random_coef_mask < 0.5)[:, 0]

        # Will be created/filled during fit()
        self._is_initialized = False
        self.unique_counts = None              # (T_unique, J+1)
        self.unique_items_features = None      # (T_unique, J+1, D)
        self._market_id_map = None             # maps raw market id -> 0..T_unique-1
        self._T_unique = None

        # Current state (used by compute_batch_utility)
        self.beta_bar = None   # (D,)
        self.r_vec = None      # (n_random,)
        self.xi_bar = None     # (T_unique,)
        self.eta = None        # (T_unique, J)
        self.phi = None        # (T_unique,)
        self.gamma = None      # (T_unique, J) int32

        # For Monte Carlo integration of random coefficients
        self.v_draws = None    # (R0, n_random)



    @property
    def trainable_weights(self):
        # MCMC isn't "trainable" via gradients. Just to comply with ChoiceModel that expects a list-like.
        weights = []
        for w in [self.beta_bar, self.r_vec, self.xi_bar, self.eta, self.phi]:
            if w is not None:
                weights.append(w)
        return weights

    def compute_batch_utility(
        self,
        shared_features_by_choice,
        items_features_by_choice,
        available_items_by_choice,
        choices=None,
    ):
        """
        ChoiceModel-required utility computation.

        Returns:
          utilities: (batch_size, J+1)
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call fit(choice_dataset) first.")

        # items_features_by_choice: (N, J+1, D)
        X = tf.cast(items_features_by_choice, tf.float64)
        N = tf.shape(X)[0]
        J_plus = tf.shape(X)[1]

        # shared_features_by_choice is expected to contain market_id in first column
        s = tf.convert_to_tensor(shared_features_by_choice)
        if len(s.shape) == 1:
            market_ids = tf.cast(s, tf.int32)
        else:
            market_ids = tf.cast(s[:, 0], tf.int32)

        # Map raw market ids to contiguous 0..T_unique-1 
        if isinstance(self._market_id_map, tf.Tensor):
            mid = tf.gather(self._market_id_map, market_ids)
        else:
            # Fallback: assume market_ids already contiguous
            mid = market_ids

        # Fixed part: sum_{d fixed} beta_d * x_{n,j,d}
        fixed_util = tf.zeros((N, J_plus), dtype=tf.float64)
        for idx in self.fixed_indices.numpy():
            fixed_util += self.beta_bar[idx] * X[:, :, idx]

        # Random part: integrate out random coefficients by averaging over v_draws
        random_mean_util = tf.zeros((N, J_plus), dtype=tf.float64)
        for idx in self.random_indices.numpy():
            random_mean_util += self.beta_bar[idx] * X[:, :, idx]

        # xi = xi_bar[t] + eta[t,j] (outside option j=0 has 0)
        xi_bar_n = tf.gather(self.xi_bar, mid)                     # (N,)
        eta_n = tf.gather(self.eta, mid)                           # (N, J)
        xi_inside = xi_bar_n[:, None] + eta_n                      # (N, J)
        xi_padded = tf.concat([tf.zeros((N, 1), tf.float64), xi_inside], axis=1)  # (N, J+1)

        util = fixed_util + random_mean_util + xi_padded

        # Apply availability mask if provided: unavailable 
        if available_items_by_choice is not None:
            avail = tf.cast(available_items_by_choice, tf.float64)
            util = tf.where(avail > 0.5, util, tf.constant(-1e9, tf.float64))

        return util

    def fit(self, choice_dataset, store_samples=True):
        """
        Runs MCMC and returns losses_history dict with:
          losses_history["train_loss"] = list of negative log-likelihood per iteration
        """
        assert self.mcmc_params.burn < self.mcmc_params.G, (
            f"burn ({self.mcmc_params.burn}) must be < G ({self.mcmc_params.G}). "
            f"No post-burn samples would be stored."
        )
        self._initialize_from_dataset(choice_dataset)
        self._initialize_state()

        losses_history = {"train_loss": []}

        # storage
        samples = None
        if store_samples:
            samples = {k: [] for k in ["beta_bar", "r_vec", "xi_bar", "eta", "gamma", "phi"]}

        # initial likelihood
        xi = self.xi_bar[:, None] + self.eta
        _, _, ll_vec = self.compute_log_likelihood_unique(xi, self.beta_bar, self.r_vec, self.v_draws)
        current_nll = float(-tf.reduce_sum(ll_vec).numpy())
        losses_history["train_loss"].append(current_nll)

        # masks for beta update (update all by default)
        update_beta_mask = tf.ones((self.D,), dtype=tf.float64)

        iterator = trange(self.mcmc_params.G, desc="Main MCMC chain")

        self.acc_beta_ = self.acc_r_ = self.acc_xi_ = self.acc_eta_ = 0

        # MCMC loop
        for g in iterator:
            # A. beta_bar  (choose updater via self.beta_method)
            if self.beta_method == 'tmh':
                self.beta_bar, ll_vec, acc = tmh_update_beta_cl(
                    self, self.beta_bar, ll_vec, self.xi_bar, self.eta,
                    self.r_vec, self.v_draws, self.mcmc_params, update_beta_mask
                )
            else:  # 'rwmh' (default)
                self.beta_bar, ll_vec, acc = mh_update_beta_cl(
                    self, self.beta_bar, ll_vec, self.xi_bar, self.eta,
                    self.r_vec, self.v_draws, self.mcmc_params, update_beta_mask
                )
            self.acc_beta_ += acc

            # B. r_vec
            self.r_vec, ll_vec, acc = mh_update_r_cl(
                self, self.r_vec, ll_vec, self.xi_bar[:, None] + self.eta,
                self.beta_bar, self.v_draws, self.mcmc_params
            )

            self.acc_r_ += acc

            # C. xi_bar
            self.xi_bar, ll_vec, acc = mh_update_xi_cl(
                self, self.xi_bar, ll_vec, self.eta,
                self.beta_bar, self.r_vec, self.v_draws, self.mcmc_params
            )

            self.acc_xi_ += acc

            # D. eta
            self.eta, ll_vec, acc = mh_update_eta_cl(
                self, self.eta, ll_vec, self.gamma,
                self.xi_bar, self.beta_bar, self.r_vec, self.v_draws, self.mcmc_params
            )

            self.acc_eta_ += acc

            # E. gamma/phi
            self.gamma, self.phi = gibbs_update_gamma_phi_tf(self.eta, self.phi, self.mcmc_params)

            # Track loss each iteration (negative log-likelihood)
            nll = float(-tf.reduce_sum(ll_vec).numpy())
            losses_history["train_loss"].append(nll)

            # store draws after burn-in
            if store_samples and g >= self.mcmc_params.burn:
                samples["beta_bar"].append(self.beta_bar.numpy())
                samples["r_vec"].append(self.r_vec.numpy())
                samples["xi_bar"].append(self.xi_bar.numpy())
                samples["eta"].append(self.eta.numpy())
                samples["gamma"].append(self.gamma.numpy())
                samples["phi"].append(self.phi.numpy())

        
        if store_samples:
            self.samples_ = {k: np.asarray(v) for k, v in samples.items()}
            if self.verbose:
                print(f"Final beta: {self.beta_bar.numpy()}")
                print(f"Final sigma (std): {np.exp(self.r_vec.numpy())}")
                print(f"Acceptance rates:  beta={self.acc_beta_/self.mcmc_params.G:.3f}  "
                      f"r={self.acc_r_/self.mcmc_params.G:.3f}  "
                      f"xi={self.acc_xi_/(self.mcmc_params.G*self._T_unique):.3f}  "
                      f"eta={self.acc_eta_/(self.mcmc_params.G*self._T_unique*self.J):.3f}")
        else:
            self.samples_ = None


        return losses_history

    # -----------------------
    # Internal initialization
    # -----------------------

    def _initialize_from_dataset(self, dataset):
        """
        Builds the aggregated unique-market representation used by the likelihood.
        """

        # Check if items features and availability data are given
        assert dataset.items_features_by_choice is not None, "items_features_by_choice is required" 
        assert dataset.shared_features_by_choice is not None, "shared_features_by_choice is required"
        assert dataset.available_items_by_choice is not None, "available_items_by_choice is required"

        
        shared_np = dataset.shared_features_by_choice
        if isinstance(shared_np, (list, tuple)):
            shared_np = shared_np[0]
        shared_np = np.asarray(shared_np)

        if shared_np.ndim == 1:
            market_ids_full = shared_np.astype(int)
        else:
            market_ids_full = shared_np[:, 0].astype(int)
        market_ids_full = market_ids_full.flatten()

        choices_full = np.asarray(dataset.choices).astype(int)

        unique_market_ids, first_indices = np.unique(market_ids_full, return_index=True)
        self._T_unique = len(unique_market_ids)


        max_mid = int(np.max(unique_market_ids))
        lookup = -np.ones((max_mid + 1,), dtype=np.int32)
        lookup[unique_market_ids] = np.arange(self._T_unique, dtype=np.int32)
        self._market_id_map = tf.constant(lookup, dtype=tf.int32)

        # Aggregate counts per unique market
        mid_contig = lookup[market_ids_full]
        pairs = np.stack([mid_contig, choices_full], axis=1)
        counts = np.zeros((self._T_unique, self.J + 1), dtype=np.int32)
        np.add.at(counts, (pairs[:, 0], pairs[:, 1]), 1)
        self.unique_counts = tf.constant(counts, dtype=tf.float64)

        # Unique item features (take first occurrence of each market)
        items_full = dataset.items_features_by_choice
        if isinstance(items_full, (list, tuple)):
            items_full = items_full[0]

        if isinstance(items_full, tf.Tensor):
            unique_feats = tf.gather(items_full, first_indices)
        else:
            unique_feats = tf.constant(np.asarray(items_full)[first_indices], dtype=tf.float64)

        self.unique_items_features = tf.cast(unique_feats, tf.float64)  # (T_unique, J+1, D)

        self._is_initialized = True

    def _initialize_state(self):
        tf.random.set_seed(self.seed)
        self.beta_bar = tf.Variable(tf.zeros((self.D,), dtype=tf.float64))
        self.r_vec = tf.Variable(tf.zeros((self.n_random,), dtype=tf.float64))
        self.xi_bar = tf.Variable(tf.zeros((self._T_unique,), dtype=tf.float64))
        self.eta = tf.Variable(tf.zeros((self._T_unique, self.J), dtype=tf.float64))

        self.phi = tf.Variable(tf.fill((self._T_unique,), tf.constant(0.5, dtype=tf.float64)))
        self.gamma = tf.Variable(tf.ones((self._T_unique, self.J), dtype=tf.int32))

        self.v_draws = tf.random.normal((self.mcmc_params.R0, self.n_random), dtype=tf.float64)

    # -----------------------
    # Unique-market likelihood
    # -----------------------

    def compute_batch_utility_unique(self, unique_items, xi, beta_bar, r_vec, v_draws):
        """
        Utility for unique markets:
          unique_items: (T_unique, J+1, D)
          xi:          (T_unique, J)   (inside only)
        Returns:
          U: (T_unique, R, J+1)
        """
        R = tf.shape(v_draws)[0]
        T = tf.shape(unique_items)[0]
        J_plus = tf.shape(unique_items)[1]

        fixed_util = tf.zeros((T, J_plus), dtype=tf.float64)
        for idx in self.fixed_indices.numpy():
            fixed_util += beta_bar[idx] * unique_items[:, :, idx]

        random_util = tf.zeros((T, R, J_plus), dtype=tf.float64)
        for i, idx in enumerate(self.random_indices.numpy()):
            sigma_d = tf.exp(r_vec[i])
            beta_draws = beta_bar[idx] + sigma_d * v_draws[:, i]  # (R,)
            random_util += beta_draws[None, :, None] * unique_items[:, :, idx][:, None, :]

        xi_padded = tf.concat([tf.zeros((T, 1), tf.float64), xi], axis=1)  # (T, J+1)
        u_xi = xi_padded[:, None, :]  # (T, 1, J+1)

        return fixed_util[:, None, :] + random_util + u_xi

    # @tf.function
    @tf.function(reduce_retracing=True) 
    def compute_log_likelihood_unique(self, xi, beta_bar, r_vec, v_draws):
        """
        xi: (T_unique, J) inside shocks
        """
        U = self.compute_batch_utility_unique(self.unique_items_features, xi, beta_bar, r_vec, v_draws)
        denom = tf.reduce_logsumexp(U, axis=2, keepdims=True)
        probs = tf.exp(U - denom)                       # (T_unique, R, J+1)
        sigma = tf.reduce_mean(probs, axis=1)           # (T_unique, J+1)
        ll_vec = tf.reduce_sum(self.unique_counts * tf.math.log(sigma + 1e-300), axis=1)  # (T_unique,)
        return sigma, probs, ll_vec



    @tf.function
    def compute_grad_hess_beta(self, xi, beta_bar, r_vec, v_draws):
        """
        Computes Gradient and Hessian of LL w.r.t beta_bar.
        Only updates components not masked as fixed externally.
        """
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(beta_bar)
            with tf.GradientTape() as tape1:
                tape1.watch(beta_bar)
                _, _, ll_vec = self.compute_log_likelihood_unique(
                    xi, beta_bar, r_vec, v_draws
                )
                ll_total = tf.reduce_sum(ll_vec)

            grad = tape1.gradient(ll_total, beta_bar)

        hess = tape2.jacobian(grad, beta_bar)
        del tape2

        return grad, hess

# =====================================
# MCMC UPDATE STEPS
# =====================================
def mh_update_beta_cl(model, beta_cur, ll_vec_cur, xi_bar, eta, r_vec, v_draws, mcmc, update_mask):
    """
    Random Walk MH update for beta_bar.
    Uses mcmc.step_beta as the isotropic proposal std.
    update_mask: (D,) float tensor - 1.0 for dims to update, 0.0 for fixed.
    """
    xi = xi_bar[:, None] + eta
    D = model.D

    # Symmetric RW proposal — only perturb unmasked dims
    noise     = tf.random.normal((D,), stddev=tf.cast(mcmc.step_beta, tf.float64), dtype=tf.float64)
    beta_prop = beta_cur + noise * update_mask

    # Likelihoods
    _, _, ll_vec_prop = model.compute_log_likelihood_unique(xi, beta_prop, r_vec, v_draws)

    ll_sum_cur  = tf.reduce_sum(ll_vec_cur)
    ll_sum_prop = tf.reduce_sum(ll_vec_prop)

    # Gaussian prior N(0, V_beta) on updated dims only
    Vb = tf.cast(mcmc.V_beta, tf.float64)
    log_prior_cur  = -0.5 * tf.reduce_sum((beta_cur  * update_mask) ** 2) / Vb
    log_prior_prop = -0.5 * tf.reduce_sum((beta_prop * update_mask) ** 2) / Vb

    # Symmetric proposal -> no q_diff correction
    log_alpha = (ll_sum_prop + log_prior_prop) - (ll_sum_cur + log_prior_cur)

    if tf.math.log(tf.random.uniform([], dtype=tf.float64)) < log_alpha:
        return beta_prop, ll_vec_prop, 1
    return beta_cur, ll_vec_cur, 0


def tmh_update_beta_cl(model, beta_cur, ll_vec_cur, xi_bar, eta, r_vec, v_draws, mcmc, update_mask):
    """
    TMH Update for Beta using Choice-Learn Model.

    Args:
        update_mask: (D,) boolean array - True for coefficients to update, False for fixed
    """
    xi = xi_bar[:, None] + eta
    D = model.D

    # Create TF variable for optimization
    beta_hat = tf.Variable(beta_cur, dtype=tf.float64)

    # Newton-Raphson to find mode (only for updateable parameters)
    for _ in range(5):
        g, H = model.compute_grad_hess_beta(xi, beta_hat, r_vec, v_draws)

        # Apply mask: zero out gradients and hessian for fixed parameters
        g_masked = g * update_mask
        H_masked = H * update_mask[:, None] * update_mask[None, :]

        # Prior contribution
        g_prior = -beta_hat / mcmc.V_beta * update_mask
        H_prior = -tf.linalg.diag(update_mask / mcmc.V_beta)

        g_total = g_masked + g_prior
        H_total = H_masked + H_prior

        try:
            H_reg = H_total - 1e-6 * tf.eye(D, dtype=tf.float64)
            step = tf.linalg.solve(H_reg, g_total[:, None])

            # Only update components where update_mask is True
            delta = tf.reshape(step, [-1]) * update_mask
            beta_hat.assign(beta_hat - delta)
        except:
            break

    # Proposal covariance
    negH = -H_total + 1e-3 * tf.eye(D, dtype=tf.float64)

    try:
        chol = tf.linalg.cholesky(negH)
        inv_negH = tf.linalg.cholesky_solve(chol, tf.eye(D, dtype=tf.float64))
    except:
        inv_negH = 0.01 * tf.eye(D, dtype=tf.float64)

    cov = (mcmc.kappa_beta ** 2) * inv_negH

    # Sample proposal
    beta_prop = tfp.distributions.MultivariateNormalFullCovariance(
        loc=beta_hat, covariance_matrix=cov
    ).sample()

    # Enforce fixed parameters
    beta_prop = tf.where(update_mask > 0.5, beta_prop, beta_cur)

    # Evaluate likelihoods
    _, _, ll_vec_prop = model.compute_log_likelihood_unique(xi, beta_prop, r_vec, v_draws)

    ll_sum_cur = tf.reduce_sum(ll_vec_cur)
    ll_sum_prop = tf.reduce_sum(ll_vec_prop)

    log_prior_cur = -0.5 * tf.reduce_sum((beta_cur * update_mask)**2) / mcmc.V_beta
    log_prior_prop = -0.5 * tf.reduce_sum((beta_prop * update_mask)**2) / mcmc.V_beta

    q_dist = tfp.distributions.MultivariateNormalFullCovariance(
        loc=beta_hat, covariance_matrix=cov
    )
    log_q_diff = q_dist.log_prob(beta_cur) - q_dist.log_prob(beta_prop)

    log_alpha = (ll_sum_prop + log_prior_prop - ll_sum_cur - log_prior_cur) + log_q_diff

    if tf.math.log(tf.random.uniform([], dtype=tf.float64)) < log_alpha:
        return beta_prop, ll_vec_prop, 1

    return beta_cur, ll_vec_cur, 0
    

def mh_update_xi_cl(model, xi_bar_cur, ll_vec_cur, eta, beta_bar, r_vec, v_draws, mcmc):
    """Update xi_bar using Model Likelihood"""
    T = tf.shape(xi_bar_cur)[0]
    prop_step = tf.random.normal((T,), stddev=mcmc.step_xibar, dtype=tf.float64)
    xi_bar_prop = xi_bar_cur + prop_step

    xi_prop = xi_bar_prop[:, None] + eta
    _, _, ll_vec_prop = model.compute_log_likelihood_unique(xi_prop, beta_bar, r_vec, v_draws)
    
    lp_cur = ll_vec_cur - 0.5 * (xi_bar_cur**2) / mcmc.V_xibar
    lp_prop = ll_vec_prop - 0.5 * (xi_bar_prop**2) / mcmc.V_xibar

    accept_mask = tf.math.log(tf.random.uniform((T,), dtype=tf.float64)) < (lp_prop - lp_cur)

    new_xi = tf.where(accept_mask, xi_bar_prop, xi_bar_cur)
    new_ll = tf.where(accept_mask, ll_vec_prop, ll_vec_cur)
    acc = tf.reduce_sum(tf.cast(accept_mask, tf.int32))

    return new_xi, new_ll, acc

def mh_update_r_cl(model, r_vec_cur, ll_vec_cur, xi, beta_bar, v_draws, mcmc):
    """
    Random Walk MH update for r_vec (log sigma for random coefficients).
    r_vec has dimension (n_random,)
    """
    n_random = tf.shape(r_vec_cur)[0]

    # Propose new r_vec
    r_vec_prop = r_vec_cur + tf.random.normal(
        tf.shape(r_vec_cur), stddev=mcmc.step_r, dtype=tf.float64
    )

    # Compute new Likelihood
    _, _, ll_vec_prop = model.compute_log_likelihood_unique(xi, beta_bar, r_vec_prop, v_draws)

    # Priors (Normal on each r)
    lp_cur = tf.reduce_sum(ll_vec_cur) - 0.5 * tf.reduce_sum(r_vec_cur**2) / mcmc.V_r
    lp_prop = tf.reduce_sum(ll_vec_prop) - 0.5 * tf.reduce_sum(r_vec_prop**2) / mcmc.V_r

    # Accept/Reject
    if tf.math.log(tf.random.uniform([], dtype=tf.float64)) < (lp_prop - lp_cur):
        return r_vec_prop, ll_vec_prop, 1

    return r_vec_cur, ll_vec_cur, 0

def mh_update_eta_cl(model, eta_cur, ll_vec_cur, gamma, xi_bar, beta_bar, r_vec, v_draws, mcmc):
    """
    Coordinate-wise update for eta using the Model.
    Iterates over J products, updating eta[:, j] for all markets T in parallel.
    """
    T = model.T
    J = model.J
    curr_eta = eta_cur
    curr_ll = ll_vec_cur
    acc_total = 0

    step_eta = tf.constant(mcmc.step_eta, dtype=tf.float64)
    scale_small = tf.constant(0.01, dtype=tf.float64)
    scale_large = tf.constant(1.0, dtype=tf.float64)
    tau0 = tf.constant(mcmc.tau0, dtype=tf.float64)
    tau1 = tf.constant(mcmc.tau1, dtype=tf.float64)

    for j in range(J):
        gamma_col = gamma[:, j]

        # Propose new eta column
        sigma_factors = tf.where(gamma_col == 0, scale_small, scale_large)
        sigma_prop = step_eta * sigma_factors
        noise = tf.random.normal((T,), dtype=tf.float64) * sigma_prop
        prop_col = curr_eta[:, j] + noise

        # Create candidate eta tensor
        indices = tf.stack([tf.range(T), tf.fill([T], j)], axis=1)
        prop_eta = tf.tensor_scatter_nd_update(curr_eta, indices, prop_col)

        # Compute Likelihood
        xi_prop = xi_bar[:, None] + prop_eta
        _, _, ll_vec_new = model.compute_log_likelihood_unique(xi_prop, beta_bar, r_vec, v_draws)

        # Priors
        var_vec = tf.where(gamma_col == 1, tau1**2, tau0**2)
        prior_cur = -0.5 * (curr_eta[:, j]**2) / var_vec
        prior_prop = -0.5 * (prop_col**2) / var_vec

        # Accept/Reject
        lp_diff = (ll_vec_new + prior_prop) - (curr_ll + prior_cur)
        accept_mask = tf.math.log(tf.random.uniform((T,), dtype=tf.float64)) < lp_diff

        curr_eta = tf.where(accept_mask[:, None], prop_eta, curr_eta)
        curr_ll = tf.where(accept_mask, ll_vec_new, curr_ll)
        acc_total += tf.reduce_sum(tf.cast(accept_mask, tf.int32))

    return curr_eta, curr_ll, acc_total

def gibbs_update_gamma_phi_tf(eta, phi, mcmc):
    """Gibbs update for Gamma and Phi"""
    phi = tf.cast(phi, tf.float64)
    eta = tf.cast(eta, tf.float64)

    tau0 = tf.constant(mcmc.tau0, dtype=tf.float64)
    tau1 = tf.constant(mcmc.tau1, dtype=tf.float64)
    a_phi = tf.constant(mcmc.a_phi, dtype=tf.float64)
    b_phi = tf.constant(mcmc.b_phi, dtype=tf.float64)

    logN0 = -0.5 * (eta**2)/(tau0**2) - tf.math.log(tau0)
    logN1 = -0.5 * (eta**2)/(tau1**2) - tf.math.log(tau1)

    safe_phi = tf.clip_by_value(phi, 1e-12, 1.0 - 1e-12)
    logit = (tf.math.log(safe_phi)[:, None] + logN1) -  (tf.math.log(1.0 - safe_phi)[:, None] + logN0)
    prob1 = tf.math.sigmoid(logit)

    gamma = tf.random.uniform(tf.shape(prob1), dtype=tf.float64) < prob1
    gamma = tf.cast(gamma, tf.int32)

    J = tf.cast(tf.shape(eta)[1], tf.float64)
    sum_gamma = tf.cast(tf.reduce_sum(gamma, axis=1), tf.float64)
    new_phi = tfp.distributions.Beta(a_phi + sum_gamma, b_phi + (J - sum_gamma)).sample()

    return gamma, new_phi




def adapt_step_size(val, acc_count, total_count, calib):
    """Adjusts a single step size based on acceptance rate."""
    rate = acc_count / max(total_count, 1)
    new_val = val

    if rate > calib.accept_target_high:
        new_val *= calib.upscale_ratio
    elif rate < calib.accept_target_low:
        new_val *= calib.downscale_ratio

    return float(np.clip(new_val, calib.min_step, calib.max_step))



def calibrate_stepsizes_cl(choice_dataset, sim_params, mcmc_params, calib_params,
                           fixed_beta_mask=None, seed=123,
                           beta_method="rwmh", verbose=False):
    """
    Calibrates MCMC step sizes using BayesianSparseRandomLogit.

    beta_method:
        - "tmh"  : uses tmh_update_beta_cl, adapts mcmc_params.kappa_beta
        - "rwmh" : uses mh_update_beta_cl,  adapts mcmc_params.step_beta

    Returns:
        mcmc_params (mutated in-place)
    """
    assert calib_params.burn_in < calib_params.calib_iters, (
        f"burn ({calib_params.burn_in}) must be < G ({calib_params.calib_iters}). "
        f"No adaptation!"
    )

    if verbose:
        print(">>> Starting Pilot Calibration...")

    # Build model 
    model = BayesianSparseRandomLogit(
        sim_params=sim_params,
        mcmc_params=mcmc_params,
        random_coef_mask=mcmc_params.random_coef_mask,
        seed=seed,
        beta_method=beta_method,
        verbose=verbose,
    )

    # Initialize internal unique-market representation from dataset + initial state
    model._initialize_from_dataset(choice_dataset)
    model._initialize_state()

    T_unique = model._T_unique
    J = model.J
    D = model.D

    # Determine update mask (1=update, 0=fixed)
    if fixed_beta_mask is None:
        update_beta_mask = tf.ones((D,), dtype=tf.float64)
    else:
        update_beta_mask = tf.constant(1.0 - np.asarray(fixed_beta_mask), dtype=tf.float64)

    # Local references
    beta_bar = model.beta_bar
    r_vec = model.r_vec
    xi_bar = model.xi_bar
    eta = model.eta
    phi = model.phi
    gamma = model.gamma
    v_draws = model.v_draws

    # Initial likelihood
    _, _, ll_vec = model.compute_log_likelihood_unique(
        xi_bar[:, None] + eta, beta_bar, r_vec, v_draws
    )

    # Counters
    acc_b = acc_r = acc_xi = acc_eta = 0
    win_b = win_r = win_xi = win_eta = 0

    if verbose:
        iterator = trange(calib_params.calib_iters, desc="Calibrating")
    else:
        iterator = range(calib_params.calib_iters)

    for g in iterator:
        # A. Beta update (method switch)
        if beta_method == "tmh":
            beta_bar, ll_vec, ok = tmh_update_beta_cl(
                model, beta_bar, ll_vec, xi_bar, eta, r_vec, v_draws, mcmc_params, update_beta_mask
            )
        else:  # "rwmh"
            beta_bar, ll_vec, ok = mh_update_beta_cl(
                model, beta_bar, ll_vec, xi_bar, eta, r_vec, v_draws, mcmc_params, update_beta_mask
            )
        acc_b += ok
        win_b += 1

        # B. r update
        r_vec, ll_vec, ok = mh_update_r_cl(
            model, r_vec, ll_vec, xi_bar[:, None] + eta, beta_bar, v_draws, mcmc_params
        )
        acc_r += ok
        win_r += 1

        # C. xi_bar update
        xi_bar, ll_vec, a_xi = mh_update_xi_cl(
            model, xi_bar, ll_vec, eta, beta_bar, r_vec, v_draws, mcmc_params
        )
        acc_xi += a_xi
        win_xi += T_unique

        # D. eta update
        eta, ll_vec, a_eta = mh_update_eta_cl(
            model, eta, ll_vec, gamma, xi_bar, beta_bar, r_vec, v_draws, mcmc_params
        )
        acc_eta += a_eta
        win_eta += (T_unique * J)

        # E. gamma/phi update
        gamma, phi = gibbs_update_gamma_phi_tf(eta, phi, mcmc_params)

        # Write back into the model state
        model.beta_bar.assign(beta_bar)
        model.r_vec.assign(r_vec)
        model.xi_bar.assign(xi_bar)
        model.eta.assign(eta)
        model.gamma.assign(gamma)
        model.phi.assign(phi)


        
        # Adaptation step
        if (g + 1) > calib_params.burn_in and (g + 1) % calib_params.adapt_every == 0:
            # beta step-size adaptation depends on beta_method
            if beta_method == "tmh":
                mcmc_params.kappa_beta = adapt_step_size(
                    mcmc_params.kappa_beta, acc_b, win_b, calib_params
                )
            else:  # "rwmh"
                mcmc_params.step_beta = adapt_step_size(
                    mcmc_params.step_beta, acc_b, win_b, calib_params
                )

            mcmc_params.step_r = adapt_step_size(
                mcmc_params.step_r, acc_r, win_r, calib_params
            )
            mcmc_params.step_xibar = adapt_step_size(
                mcmc_params.step_xibar, acc_xi, win_xi, calib_params
            )
            mcmc_params.step_eta = adapt_step_size(
                mcmc_params.step_eta, acc_eta, win_eta, calib_params
            )

            # Reset counters
            acc_b = acc_r = acc_xi = acc_eta = 0
            win_b = win_r = win_xi = win_eta = 0

    if verbose:
        if beta_method == "tmh":
            beta_msg = f"kappa_beta={mcmc_params.kappa_beta:.3f}"
        else:
            beta_msg = f"step_beta={mcmc_params.step_beta:.3f}"

        print(
            "Calibration Done. Tuned: "
            f"{beta_msg}, "
            f"step_r={mcmc_params.step_r:.3f}, "
            f"step_xibar={mcmc_params.step_xibar:.3f}, "
            f"step_eta={mcmc_params.step_eta:.3f}"
        )

    return mcmc_params


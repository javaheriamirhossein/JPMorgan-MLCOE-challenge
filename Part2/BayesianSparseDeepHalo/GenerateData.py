import numpy as np
import tensorflow as tf
from choice_learn.data import ChoiceDataset
from dataclasses import dataclass, field
from typing import Optional, Any

from BayesianSparseDeepHalo.DeepHalo import DeepHaloEncoder
from BayesianSparseDeepHalo.DeepHalo_MCEM_Core import compute_probs_and_ll_batch_masked

# =====================================
# (DGP) DATA GENERATION
# =====================================

@dataclass
class SimParams:
    """Data Generation Parameters"""
    T: int = 100                          # Number of markets
    J: int = 15                           # Number of inside products per market
    Nt: int = 1000                        # Consumers per market
    D: int = 2                            # Number of product features
    beta_mean: Optional[np.ndarray] = None   # (D,) mean taste coefficients
    sigma_beta: Optional[np.ndarray] = None  # (D,) std of random coefficients; 0 = fixed
    xi_bar: float = -1.0                 # Mean market-level demand shock
    seed: int = 123                      # RNG seed for reproducibility

    def __post_init__(self) -> None:
        # Default: price coefficient negative, characteristic coefficient positive
        if self.beta_mean is None:
            self.beta_mean = np.array([-1.0, 0.5])
        # Default: price is random, second characteristic is fixed
        if self.sigma_beta is None:
            self.sigma_beta = np.array([1.5, 0.0])


@dataclass
class MCMCParams:
    """Hyperparameters for the Bayesian Algorithm"""
    R0: int = 200        # Number of fixed simulation draws for share integral
    G: int = 10000       # Total MCMC iterations
    burn: int = 5000     # Burn-in samples to discard

    # (D,) binary mask: 1 = random coefficient, 0 = fixed coefficient
    random_coef_mask: Optional[np.ndarray] = None

    # --- Spike-and-slab prior hyperparameters ---
    tau0: float = 1e-3   # Spike variance (near-zero component)
    tau1: float = 1.0    # Slab variance (diffuse component)
    a_phi: float = 1.0   # Beta prior shape a for inclusion probability phi
    b_phi: float = 1.0   # Beta prior shape b for inclusion probability phi

    # --- Gaussian prior variances ---
    V_beta: float = 10.0    # Prior variance for taste coefficients beta
    V_xibar: float = 10.0   # Prior variance for market intercepts xi_bar
    V_r: float = 0.5        # Prior variance for random-coefficient std r

    # --- MH proposal step sizes (tuned adaptively toward target_accept) ---
    step_beta: float = 0.5
    step_r: float = 0.5
    step_xibar: float = 0.5
    step_eta: float = 0.5
    step_phi: float = 0.5

    target_accept: float = 0.35  # Target Metropolis-Hastings acceptance rate

    D: int = 2  # Feature dimension (must match SimParams.D)

    def __post_init__(self) -> None:
        # Default: only the first coefficient is treated as random
        if self.random_coef_mask is None:
            self.random_coef_mask = np.zeros(self.D)
            self.random_coef_mask[0] = 1.0


def generate_data_tf(
    dgp_type: int,   # DGP scenario: 1=sparse-exog, 2=sparse-endog, 3=normal-exog, 4=normal-endog
    sim: SimParams,  # Simulation configuration
) -> tuple[dict, ChoiceDataset, dict]:
    """
    Generates synthetic discrete-choice data using TensorFlow.

    Returns:
        data_mcmc:   Dict with raw numpy arrays for the MCMC runner.
        dataset_cl:  Choice-Learn ChoiceDataset with Market IDs in shared_features.
        true_params: Dict of true parameter values used in the DGP.
    """
    tf.random.set_seed(sim.seed)
    T, J, Nt, D = sim.T, sim.J, sim.Nt, sim.D

    # Raw product characteristics: uniform on [1, 2], shape (T, J, D)
    X_raw = tf.random.uniform((T, J, D), minval=1.0, maxval=2.0, dtype=tf.float64)
    # Cost shock used to create endogenous price variation, shape (T, J)
    u_cost = tf.random.normal((T, J), mean=0.0, stddev=0.7, dtype=tf.float64)

    # --- DGP-specific demand shock eta_star ---
    if dgp_type in (1, 2):
        # Sparse alternating pattern: 40% of products get ±1 shocks, rest zero
        n_active = int(0.4 * J)
        pattern = tf.where(tf.math.mod(tf.range(n_active), 2) == 1, 1.0, -1.0)
        pattern = tf.cast(pattern, tf.float64)
        pad_zeros = tf.zeros(J - n_active, dtype=tf.float64)
        eta_base = tf.concat([pattern, pad_zeros], axis=0)      # (J,)
        eta_star = tf.tile(eta_base[None, :], [T, 1])           # (T, J) identical across markets
    elif dgp_type in (3, 4):
        # Normally distributed shocks, small std ~ 1/3
        eta_star = tf.random.normal((T, J), mean=0.0, stddev=1.0/3.0, dtype=tf.float64)  # (T, J)
    else:
        raise ValueError("dgp_type must be 1..4")

    # Full demand shock: xi_jt = xi_bar + eta_jt, shape (T, J)
    xi_star = sim.xi_bar + eta_star

    # --- Endogeneity: alpha shifts price when eta_star is large ---
    alpha = tf.zeros((T, J), dtype=tf.float64)     # (T, J), zero by default (exogenous DGPs)
    if dgp_type == 2:
        # Sparse endogeneity: price shifted ±0.3 for high/low eta products
        alpha = tf.where(eta_star >  0.9,  0.3, alpha)
        alpha = tf.where(eta_star < -0.9, -0.3, alpha)
    elif dgp_type == 4:
        # Normal endogeneity: price shifted ±0.3 for |eta| >= 1/3
        alpha = tf.where(eta_star >= (1.0/3.0),  0.3, alpha)
        alpha = tf.where(eta_star <= -(1.0/3.0), -0.3, alpha)

    # Endogenous price: p_jt = alpha_jt + 0.3 * x_jt1 + u_cost_jt
    X_adjusted = X_raw.numpy()
    X_adjusted[:, :, 0] = (alpha + 0.3 * X_raw[:, :, 0] + u_cost).numpy()   # (T, J) first feature = price
    X_adjusted = tf.constant(X_adjusted, dtype=tf.float64)    # (T, J, D)

    # --- Simulate consumer choices ---
    eps = tf.random.normal((T, Nt, D), dtype=tf.float64)   # (T, Nt, D) standard normal draws
    # Individual-specific coefficients: beta_i ~ N(beta_mean, sigma_beta)
    beta_i = sim.beta_mean[None, None, :] + sim.sigma_beta[None, None, :] * eps   # (T, Nt, D)

    # Indirect utility for inside goods: U_jt = sum_d beta_id * x_jd + xi_jt
    U_inside = tf.einsum('tnd,tjd->tnj', beta_i, X_adjusted) + xi_star[:, None, :]  # (T, Nt, J)

    # Prepend outside option with normalized utility = 0
    zeros = tf.zeros((T, Nt, 1), dtype=tf.float64)
    U_all = tf.concat([zeros, U_inside], axis=2)   # (T, Nt, J+1)

    # Numerically stable softmax via log-sum-exp
    denom = tf.reduce_logsumexp(U_all, axis=2, keepdims=True)
    probs = tf.exp(U_all - denom)                  # (T, Nt, J+1) choice probabilities

    # Average over consumers to get market shares, then convert to counts
    shares = tf.reduce_mean(probs, axis=1)         # (T, J+1)
    q = tf.cast(tf.round(Nt * shares), tf.int32)   # (T, J+1) rounded counts

    # Fix rounding error: add/subtract difference from the outside option
    current_sum = tf.reduce_sum(q, axis=1)         # (T,)
    diff = Nt - current_sum                        # (T,) rounding residual
    q_outside = q[:, 0] + diff                     # (T,) corrected outside count
    q_rest = q[:, 1:]                              # (T, J) inside counts unchanged
    q_final = tf.concat([q_outside[:, None], q_rest], axis=1)   # (T, J+1)

    # --- Pack data for MCMC ---
    X_mcmc: np.ndarray = X_adjusted.numpy()        # (T, J, D)
    q_mcmc: np.ndarray = q_final.numpy()           # (T, J+1)

    data_mcmc: dict = {
        "X":  X_mcmc,
        "q":  q_mcmc,
        "Nt": Nt
    }

    # --- Pack data for Choice-Learn ---
    choices_list: list = []
    market_ids_list: list = []
    q_np = q_mcmc

    # Expand each count into individual observations (one row per consumer choice)
    for t in range(T):
        for j in range(J + 1):
            count = q_np[t, j]
            if count > 0:
                choices_list.append(np.full(count, j))        # choice index
                market_ids_list.append(np.full(count, t))     # market identifier

    choices_flat: np.ndarray = np.concatenate(choices_list)       # (N_obs,)
    market_ids_flat: np.ndarray = np.concatenate(market_ids_list) # (N_obs,)

    # Prepend a row of zeros for the outside option, shape (T, J+1, D)
    items_feat_np: np.ndarray = X_mcmc               # (T, J, D)
    zeros_outside = np.zeros((T, 1, D))
    items_feat_all: np.ndarray = np.concatenate([zeros_outside, items_feat_np], axis=1)  # (T, J+1, D)

    # Index item features by each consumer's market, shape (N_obs, J+1, D)
    items_feat_expanded: np.ndarray = items_feat_all[market_ids_flat]
    # All items available to all consumers (no availability restrictions)
    available_items: np.ndarray = np.ones((len(choices_flat), J + 1))

    # Market ID as the only shared (consumer-level) feature, shape (N_obs, 1)
    shared_feat: np.ndarray = market_ids_flat.reshape(-1, 1).astype(np.float32)

    feature_names: list[str] = [f"x{i}" for i in range(D)]

    dataset_cl = ChoiceDataset(
        choices=choices_flat,
        shared_features_by_choice=shared_feat,
        shared_features_by_choice_names=["market_id"],
        items_features_by_choice=items_feat_expanded,
        available_items_by_choice=available_items,
        items_features_by_choice_names=feature_names
    )

    true_params: dict = {
        "eta_star":   eta_star.numpy(),   # (T, J) product-level demand shocks
        "xi_star":    xi_star.numpy(),    # (T, J) full demand shocks = xi_bar + eta
        "X":          X_adjusted.numpy(), # (T, J, D) endogenous product characteristics
        "shares":     shares.numpy(),     # (T, J+1) true market shares incl. outside
        "beta_mean":  sim.beta_mean,      # (D,) mean taste parameters
        "sigma_beta": sim.sigma_beta,     # (D,) random coefficient std devs
        "u_cost":     u_cost.numpy(),     # (T, J) cost shocks
    }

    return data_mcmc, dataset_cl, true_params


def generate_teacher_dgp(
    sim: SimParams,
    encoder_kwargs: Optional[dict] = None,   # Kwargs forwarded to DeepHaloEncoder
    beta_star: Optional[np.ndarray] = None,  # (D,) true taste coefficients; sampled if None
    r_star: Optional[np.ndarray] = None,     # (D,) true random-coef stds; sampled if None
) -> tuple[dict, dict, Any]:
    """
    Generates data from the teacher (DeepHalo) DGP for hybrid sparse context-dependent demand estimation.

    Returns:
        data:    Dict with X, q, Nt, avail for the MCMC runner.
        true:    Dict of true latent objects (Z, xi, beta, etc.).
        teacher: The DeepHaloEncoder instance (weights frozen).
    """
    rng = np.random.default_rng(sim.seed)
    tf.random.set_seed(sim.seed)
    T, M, J, Nt, D = sim.T, sim.M, sim.J, sim.Nt, sim.D

    # Default encoder architecture; force out_dim = D
    if encoder_kwargs is None:
        encoder_kwargs = dict(H=7, depth=3, embed=16, dropout=0.0, block_type="qua", out_dim=D)
    else:
        encoder_kwargs = {**encoder_kwargs, "out_dim": D}

    # Universal product characteristics, shape (M, D); same across markets
    X_univ: np.ndarray = rng.normal(size=(M, D)).astype(np.float64)

    # Availability mask: each market t gets J products drawn without replacement from M
    avail: np.ndarray = np.zeros((T, M), dtype=np.float64)   # (T, M)
    for t in range(T):
        idx = rng.choice(M, size=J, replace=False)
        avail[t, idx] = 1.0

    # Tile universal characteristics and zero-out unavailable products
    X_full: np.ndarray = np.tile(X_univ[None, :, :], (T, 1, 1))   # (T, M, D)
    X_full *= avail[:, :, None]                                     # (T, M, D) masked

    # Sample true linear parameters if not provided
    beta_star = (rng.normal(size=(D,)).astype(np.float64)
                 if beta_star is None else np.array(beta_star, dtype=np.float64).reshape(-1))
    # r_star: small negative mean encourages realistic price-like coefficients
    r_star = (rng.normal(loc=-0.2, scale=0.3, size=(D,)).astype(np.float64)
              if r_star is None else np.array(r_star, dtype=np.float64).reshape(-1))

    # Build teacher encoder and run a warm-up forward pass to initialize weights
    teacher = DeepHaloEncoder(**encoder_kwargs)
    teacher(tf.constant(X_full[:2], tf.float64), tf.constant(avail[:2], tf.float64), training=False)

    # Generate context-aware embeddings Z for all markets, shape (T, M, D)
    Z: np.ndarray = teacher(
        tf.constant(X_full, tf.float64),
        tf.constant(avail, tf.float64),
        training=False
    ).numpy()

    # Market-level intercepts: xi_bar_t ~ N(0, sigma_xibar), shape (T,)
    xi_bar_star: np.ndarray = rng.normal(0.0, sim.sigma_xibar, size=(T,)).astype(np.float64)
    # Product-level shocks: sparse normal with sparsity probability sim.pi
    eta_star: np.ndarray = np.zeros((T, M), dtype=np.float64)
    non_zero = (rng.uniform(size=(T, M)) < (1 - sim.pi)) & (avail > 0.5)  # non-zero mask
    eta_star[non_zero] = rng.normal(0.0, sim.sigma_eta,
                                    size=int(np.sum(non_zero))).astype(np.float64)
    xi_star: np.ndarray = xi_bar_star[:, None] + eta_star   # (T, M) full demand shocks

    # Simulate market shares using the teacher embedding and true parameters
    R0_sim = 500
    v_draws = tf.random.normal((R0_sim, D), dtype=tf.float64)   # (R0_sim, D) integration draws
    q_dummy = tf.zeros((T, M + 1), dtype=tf.float64)            # placeholder quantity (unused)
    sigma_t, _, _ = compute_probs_and_ll_batch_masked(
        tf.constant(Z, tf.float64),
        q_dummy,
        tf.constant(avail, tf.float64),
        tf.constant(xi_star, tf.float64),
        tf.constant(beta_star, tf.float64),
        tf.constant(r_star, tf.float64),
        v_draws,
    )

    # Convert shares to integer counts; fix rounding to ensure sum = Nt
    shares: np.ndarray = sigma_t.numpy()                              # (T, M+1)
    q: np.ndarray = np.round(Nt * shares).astype(np.int32)           # (T, M+1)
    q[:, 0] += Nt - q.sum(axis=1)                                    # adjust outside option

    teacher.trainable = False   # Freeze teacher weights for distillation

    data: dict = {"X": X_full, "q": q, "Nt": Nt, "avail": avail}
    true: dict = {
        "X_univ":     X_univ,       # (M, D) universal product chars
        "X_full":     X_full,       # (T, M, D) market-level product chars (masked)
        "avail":      avail,        # (T, M) availability mask
        "Z_teacher":  Z,            # (T, M, D) teacher embeddings
        "beta_star":  beta_star,    # (D,) true taste coefficients
        "r_star":     r_star,       # (D,) true random-coef stds
        "xi_bar_star": xi_bar_star, # (T,) true market intercepts
        "eta_star":   eta_star,     # (T, M) true product shocks
        "xi_star":    xi_star,      # (T, M) full demand shocks
    }

    return data, true, teacher

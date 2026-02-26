import numpy as np
import tensorflow as tf
from choice_learn.data import ChoiceDataset
from dataclasses import dataclass, asdict

# =====================================
# (DGP) DATA GENERATION 
# =====================================

@dataclass
class SimParams:
    """Data Generation Parameters"""
    T: int = 100  # Number of Markets
    J: int = 15   # Number of Products per Market
    Nt: int = 1000  # Consumers per Market
    D: int = 2    # Number of features
    beta_mean: np.ndarray = None  # Mean of beta (D,)
    sigma_beta: np.ndarray = None  # Std of beta (D,)
    xi_bar: float = -1.0
    seed: int = 123

    def __post_init__(self):
        if self.beta_mean is None:
            self.beta_mean = np.array([-1.0, 0.5])
        if self.sigma_beta is None:
            self.sigma_beta = np.array([1.5, 0.0])  # Price random, w fixed (0 means fixed)
            
            
def generate_data_tf(dgp_type: int, sim: SimParams):
    """
    Generates synthetic data using TensorFlow.

    Returns:
        data_mcmc: Dict with raw numpy arrays for the MCMC runner.
        dataset_cl: Choice-Learn ChoiceDataset with Market IDs in shared_features.
        true_params: Dict of true parameter values.
    """
    tf.random.set_seed(sim.seed)
    T, J, Nt, D = sim.T, sim.J, sim.Nt, sim.D

    # Characteristics - generalized to D features
    X_raw = tf.random.uniform((T, J, D), minval=1.0, maxval=2.0, dtype=tf.float64)
    u_cost = tf.random.normal((T, J), mean=0.0, stddev=0.7, dtype=tf.float64)

    # DGP-Specific Sparse Shocks (Eta)
    if dgp_type in (1, 2):
        n_active = int(0.4 * J)
        pattern = tf.where(tf.math.mod(tf.range(n_active), 2) == 1, 1.0, -1.0)
        pattern = tf.cast(pattern, tf.float64)
        pad_zeros = tf.zeros(J - n_active, dtype=tf.float64)
        eta_base = tf.concat([pattern, pad_zeros], axis=0)
        eta_star = tf.tile(eta_base[None, :], [T, 1])
    elif dgp_type in (3, 4):
        eta_star = tf.random.normal((T, J), mean=0.0, stddev=1.0/3.0, dtype=tf.float64)
    else:
        raise ValueError("dgp_type must be 1..4")

    xi_star = sim.xi_bar + eta_star

    # Endogenous component for first feature (e.g., price)
    alpha = tf.zeros((T, J), dtype=tf.float64)
    if dgp_type == 2:
        alpha = tf.where(eta_star > 0.9, 0.3, alpha)
        alpha = tf.where(eta_star < -0.9, -0.3, alpha)
    elif dgp_type == 4:
        alpha = tf.where(eta_star >= (1.0/3.0), 0.3, alpha)
        alpha = tf.where(eta_star <= -(1.0/3.0), -0.3, alpha)

    # Adjust first feature with endogeneity
    X_adjusted = X_raw.numpy()
    X_adjusted[:, :, 0] = (alpha + 0.3 * X_raw[:, :, 0] + u_cost).numpy()
    X_adjusted = tf.constant(X_adjusted, dtype=tf.float64)

    # Consumer Choices (Simulation)
    eps = tf.random.normal((T, Nt, D), dtype=tf.float64)
    beta_i = sim.beta_mean[None, None, :] + sim.sigma_beta[None, None, :] * eps

    # Utilities: (T, Nt, J) = sum_d beta_i[:,:,d] * X[:,:,d]
    U_inside = tf.einsum('tnd,tjd->tnj', beta_i, X_adjusted) + xi_star[:, None, :]

    # Add Outside Option (0 utility)
    zeros = tf.zeros((T, Nt, 1), dtype=tf.float64)
    U_all = tf.concat([zeros, U_inside], axis=2)  # (T, Nt, J+1)

    denom = tf.reduce_logsumexp(U_all, axis=2, keepdims=True)
    probs = tf.exp(U_all - denom)

    # Market Shares -> Counts
    shares = tf.reduce_mean(probs, axis=1)  # (T, J+1)
    q = tf.cast(tf.round(Nt * shares), tf.int32)

    # Fix rounding differences
    current_sum = tf.reduce_sum(q, axis=1)
    diff = Nt - current_sum
    q_outside = q[:, 0] + diff
    q_rest = q[:, 1:]
    q_final = tf.concat([q_outside[:, None], q_rest], axis=1)

    # Pack Data for MCMC
    X_mcmc = X_adjusted.numpy()  # (T, J, D)
    q_mcmc = q_final.numpy()

    data_mcmc = {
        "X": X_mcmc,
        "q": q_mcmc,
        "Nt": Nt
    }

    # Pack Data for Choice-Learn
    choices_list = []
    market_ids_list = []
    q_np = q_mcmc

    for t in range(T):
        for j in range(J + 1):
            count = q_np[t, j]
            if count > 0:
                choices_list.append(np.full(count, j))
                market_ids_list.append(np.full(count, t))

    choices_flat = np.concatenate(choices_list)
    market_ids_flat = np.concatenate(market_ids_list)

    # Features with Outside Option (zeros) prepended
    items_feat_np = X_mcmc  # (T, J, D)
    zeros_outside = np.zeros((T, 1, D))
    items_feat_all = np.concatenate([zeros_outside, items_feat_np], axis=1)

    items_feat_expanded = items_feat_all[market_ids_flat]
    available_items = np.ones((len(choices_flat), J+1))

    shared_feat = market_ids_flat.reshape(-1, 1).astype(np.float32)

    feature_names = [f"x{i}" for i in range(D)]

    dataset_cl = ChoiceDataset(
        choices=choices_flat,
        shared_features_by_choice=shared_feat,
        shared_features_by_choice_names=["market_id"],
        items_features_by_choice=items_feat_expanded,
        available_items_by_choice=available_items,
        items_features_by_choice_names=feature_names
    )

    true_params = {
        "eta_star": eta_star.numpy(),
        "xi_star": xi_star.numpy(),
        "X": X_adjusted.numpy(),
        "shares": shares.numpy(),
        "beta_mean": sim.beta_mean,
        "sigma_beta": sim.sigma_beta,
        "u_cost": u_cost.numpy(),
    }

    return data_mcmc, dataset_cl, true_params

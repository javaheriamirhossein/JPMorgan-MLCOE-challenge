# Deephalo MCMC core

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from choice_learn.data import ChoiceDataset

from .LuSparseRandomLogit import calibrate_stepsizes_cl, BayesianSparseRandomLogit


# ============================================================
# SparseDeepHaloMCEM
# ============================================================
class SparseDeepHaloMCEM:
    def __init__(self, X_np, q_np, avail_np, encoder,
                 sim_params,
                 mcmc_params,
                 calib_params,
                 em_params,
                 seed: int = 2026,
                 beta_method="rwmh",
                 verbose=False):
        self.X_np = np.asarray(X_np)
        self.q_np = np.asarray(q_np)
        self.avail_np = np.asarray(avail_np)
        self.encoder = encoder

        self.sim_params = sim_params
        self.mcmc_params = mcmc_params
        self.calib_params = calib_params
        self.em_params = em_params
        self.seed = int(seed)

        self.beta_method = beta_method
        self.verbose = verbose

        self.X = tf.constant(self.X_np, tf.float64)
        self.avail = tf.constant(self.avail_np, tf.float64)
        self.T, self.M, _ = self.X_np.shape

        dummy_Z = self.encoder(self.X[:2], self.avail[:2], training=False)
        self.Dz = int(dummy_Z.shape[-1])

        self.opt = tf.keras.optimizers.Adam(learning_rate=float(em_params.nn_lr))
        self.posterior_means = None

    def _Z_np(self, training=False):
        Z = self.encoder(self.X, self.avail, training=training)
        return Z.numpy()

    @tf.function(reduce_retracing=True)
    def _mstep_loss(self, beta_mean, r_mean, xi_mean, eta_mean, v_draws):
        Z = self.encoder(self.X, self.avail, training=True)
        q = tf.cast(self.q_np, tf.float64)
        beta = tf.cast(beta_mean, tf.float64)
        r = tf.cast(r_mean, tf.float64)
        xi = tf.cast(xi_mean, tf.float64)
        eta = tf.cast(eta_mean, tf.float64)

        xi_inside = xi[:, None] + eta
        _, _, ll_vec = compute_probs_and_ll_batch_masked(Z, q, self.avail, xi_inside, beta, r, v_draws)
        nll = -tf.reduce_sum(ll_vec) / (tf.reduce_sum(q) + 1e-12)

        l2 = tf.add_n([tf.reduce_sum(w ** 2) for w in self.encoder.trainable_weights])
        return nll + tf.cast(self.em_params.nn_l2, tf.float64) * l2

    def _mstep(self, post):
        tf.random.set_seed(self.seed + 777)
        v_draws = tf.random.normal((self.mcmc_params.R0, self.Dz), dtype=tf.float64)
        last = None
        for _ in range(int(self.em_params.nn_steps)):
            with tf.GradientTape() as tape:
                loss = self._mstep_loss(post["beta"], post["r"], post["xi_bar"], post["eta"], v_draws)
            grads = tape.gradient(loss, self.encoder.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.encoder.trainable_weights))
            last = float(loss.numpy())
        return last

    def _estep(self):
        Z_np = self._Z_np(training=False)
        ds = build_choice_dataset_from_market_counts(Z_np, self.q_np, self.avail_np)

        self.mcmc_params.G = int(self.em_params.mcmc_per_outer)
        self.mcmc_params.burn = int(self.em_params.burn)

        if (self.posterior_means is None) or self.em_params.recalibrate_each_outer:
            self.mcmc_params = calibrate_mcmc_stepsizes(
                ds, self.sim_params, self.mcmc_params, self.calib_params,
                seed=self.seed + 456,
                beta_method=self.beta_method,
                verbose=self.verbose,
            )

        model = BayesianSparseRandomLogit(
            sim_params=self.sim_params,
            mcmc_params=self.mcmc_params,
            random_coef_mask=self.mcmc_params.random_coef_mask,
            seed=self.seed + 123,
            beta_method=self.beta_method,
            verbose=self.verbose,
        )

        _ = model.fit(ds, store_samples=True)
        draws = model.samples_

        thin = int(self.em_params.thin)
        if thin > 1:
            draws = {k: v[::thin] for k, v in draws.items()}

        post = {
            "beta": draws["beta_bar"].mean(axis=0),
            "r": draws["r_vec"].mean(axis=0),
            "xi_bar": draws["xi_bar"].mean(axis=0),
            "eta": draws["eta"].mean(axis=0),
            "phi": draws["phi"].mean(axis=0),
            "gamma": draws["gamma"].mean(axis=0),
        }
        self.posterior_means = post
        return post, model

    def run(self, tqdm_outer=None):
        # tqdm_outer: pass tqdm from caller to avoid importing it in this module
        outer_iter = range(int(self.em_params.outer_iters)) if tqdm_outer is None \
                     else tqdm_outer(range(int(self.em_params.outer_iters)), desc="Sparse DeepHalo MCEM outer")

        outer_losses = []
        for _ in outer_iter:
            post, _ = self._estep()
            loss = self._mstep(post)
            outer_losses.append(loss)

        return {
            "outer_losses": outer_losses,
            "beta_last":    self.posterior_means["beta"],
            "r_last":       self.posterior_means["r"],
            "xi_bar_last":  self.posterior_means["xi_bar"],
            "eta_last":     self.posterior_means["eta"],
            "phi_last":     self.posterior_means["phi"],
            # step sizes — name depends on beta_method
            "step_beta":    float(self.mcmc_params.kappa_beta)
                            if self.beta_method == "tmh"
                            else float(self.mcmc_params.step_beta),
            "step_r":       float(self.mcmc_params.step_r),
            "step_xibar":   float(self.mcmc_params.step_xibar),
            "step_eta":     float(self.mcmc_params.step_eta),
        }



# ============================================================
# Masked likelihood helper
# ============================================================
@tf.function
def compute_probs_and_ll_batch_masked(Z, q, avail, xi, beta_bar, r_vec, v_draws):
    T = tf.shape(Z)[0]
    M = tf.shape(Z)[1]
    R = tf.shape(v_draws)[0]

    sigma = tf.exp(r_vec)
    resid = v_draws * sigma[None, :]
    delta = tf.linalg.matvec(Z, beta_bar) + xi
    mu = tf.einsum("rd,tmd->trm", resid, Z)
    U_inside = delta[:, None, :] + mu

    avail_f = tf.cast(avail, tf.float64)
    U_inside_masked = U_inside + (-1e9) * (1.0 - avail_f)[:, None, :]

    zeros_outside = tf.zeros((T, R, 1), dtype=tf.float64)
    U_all = tf.concat([zeros_outside, U_inside_masked], axis=2)
    denom = tf.reduce_logsumexp(U_all, axis=2, keepdims=True)
    probs = tf.exp(U_all - denom)
    sigma_t = tf.reduce_mean(probs, axis=1)

    ll_vec = tf.reduce_sum(tf.cast(q, tf.float64) * tf.math.log(sigma_t + 1e-300), axis=1)
    return sigma_t, probs, ll_vec


# ============================================================
# ChoiceDataset builder
# ============================================================
def build_choice_dataset_from_market_counts(Z_np, q_np, avail_np, market_ids=None):
    T, M, Dz = Z_np.shape
    if market_ids is None:
        market_ids = np.arange(T, dtype=np.int32)

    Z_all = np.concatenate([np.zeros((T, 1, Dz)), Z_np], axis=1)
    avail_all = np.concatenate([np.ones((T, 1)), avail_np], axis=1).astype(np.float64)

    choices_list, market_list = [], []
    for t in range(T):
        for j in range(M + 1):
            c = int(q_np[t, j])
            if c > 0:
                choices_list.append(np.full(c, j, dtype=np.int32))
                market_list.append(np.full(c, market_ids[t], dtype=np.int32))

    choices = np.concatenate(choices_list, axis=0)
    market_flat = np.concatenate(market_list, axis=0)

    items_feat = Z_all[market_flat]
    avail_feat = avail_all[market_flat]
    shared_feat = market_flat.reshape(-1, 1).astype(np.float32)

    names = [f"z{i}" for i in range(Dz)]
    ds = ChoiceDataset(
        choices=choices,
        shared_features_by_choice=shared_feat,
        shared_features_by_choice_names=["market_id"],
        items_features_by_choice=items_feat,
        items_features_by_choice_names=names,
        available_items_by_choice=avail_feat,
    )
    return ds


# ============================================================
# calibrate step sizes helper
# ============================================================
def calibrate_mcmc_stepsizes(ds, sim_params, mcmc_params, calib_params, seed,
                            beta_method="rwmh", verbose=False):
    return calibrate_stepsizes_cl(
        choice_dataset=ds,
        sim_params=sim_params,
        mcmc_params=mcmc_params,
        calib_params=calib_params,
        fixed_beta_mask=None,
        seed=seed,
        beta_method=beta_method,
        verbose=verbose,
    )



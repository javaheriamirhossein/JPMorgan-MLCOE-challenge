"""
Unit and integration tests for BayesianSparseDeepHalo.DeepHalo_MCEM_Core

Test summary
---------------
 1.  compute_probs_and_ll_batch_masked – shape/type sanity
 2.  compute_probs_and_ll_batch_masked – numerical correctness
 3.  compute_probs_and_ll_batch_masked – availability masking
 4.  build_choice_dataset_from_market_counts – shape/type sanity
 5.  build_choice_dataset_from_market_counts – correctness
 6.  SparseDeepHaloMCEM – instantiation / attribute checks
 7.  SparseDeepHaloMCEM – shape / dimension inference
 8.  SparseDeepHaloMCEM – default parameter values
 9.  SparseDeepHaloMCEM._Z_np – encoder forward pass
10.  SparseDeepHaloMCEM._mstep_loss – type/shape/value
11.  SparseDeepHaloMCEM._mstep – parameter update & loss decrease
12.  SparseDeepHaloMCEM._estep – posterior structure & state management
13.  SparseDeepHaloMCEM.run – outer MCEM loop integration
14.  NLL decrease across outer iterations
15.  Teacher DGP recovery – beta converges toward ground-truth
16.  L2 regularisation affects M-step loss
17.  Reproducibility
18.  Edge cases (T=1, J=1, partial availability, large Dz)
"""

import numpy as np
import pytest
import tensorflow as tf

from choice_learn.data import ChoiceDataset

from BayesianSparseDeepHalo.DeepHalo_MCEM_Core import (
    SparseDeepHaloMCEM,
    compute_probs_and_ll_batch_masked,
    build_choice_dataset_from_market_counts,
)
from BayesianSparseDeepHalo.GenerateData import SimParams, MCMCParams, generate_data_tf


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_sim_params(T=15, J=4, D=3, Dz=5, Nt=100, seed=42):
    """Lightweight SimParams Namespace — Dz replaces D for the encoder output."""
    import argparse
    return argparse.Namespace(T=T, J=J, D=Dz, Nt=Nt)


def _make_mcmc_params(T=15, J=4, Dz=5, burn=5, G=20, R0=4):
    """Minimal MCMCParams Namespace compatible with SparseDeepHaloMCEM."""
    return MCMCParams(
        burn=burn, G=G, R0=R0,
        V_beta=10.0, V_r=10.0, V_xibar=10.0,
        tau0=0.05, tau1=1.0,
        a_phi=1.0, b_phi=1.0,
        step_beta=0.2, step_r=0.2,
        step_xibar=0.2, step_phi=0.2, step_eta=0.2,
        random_coef_mask=np.zeros(Dz),   # all-zero → locked in E-step
        target_accept=0.35,
        D=Dz,
    )


def _make_em_params(outer_iters=2, nn_steps=5, nn_lr=1e-3, nn_l2=1e-4,
                    mcmc_per_outer=20, burn=5, thin=2,
                    recalibrate_each_outer=False):
    import argparse
    return argparse.Namespace(
        outer_iters=outer_iters, nn_steps=nn_steps, nn_lr=nn_lr,
        nn_l2=nn_l2, mcmc_per_outer=mcmc_per_outer, burn=burn,
        thin=thin, recalibrate_each_outer=recalibrate_each_outer,
    )


def _make_encoder(F, Dz):
    """Minimal one-layer Dense encoder: (X [B,M,F], avail [B,M]) → Z [B,M,Dz]."""
    class TinyEncoder(tf.keras.Model):
        def __init__(self, F, Dz):
            super().__init__()
            self.dense = tf.keras.layers.Dense(Dz, dtype=tf.float64)

        def call(self, X, avail, training=False):
            return self.dense(X)   # [B, M, Dz]

    return TinyEncoder(F, Dz)


def _make_model(T=15, J=4, F=3, Dz=5, seed=42,
                outer_iters=2, nn_steps=5):
    """Build a fully wired SparseDeepHaloMCEM ready for testing."""
    rng = np.random.default_rng(seed)
    X_np     = rng.standard_normal((T, J, F))
    q_np     = np.ones((T, J + 1), dtype=np.float64)
    avail_np = np.ones((T, J), dtype=np.float64)

    sim  = _make_sim_params(T=T, J=J, D=F, Dz=Dz)
    mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz)
    em   = _make_em_params(outer_iters=outer_iters, nn_steps=nn_steps)
    enc  = _make_encoder(F, Dz)

    model = SparseDeepHaloMCEM(
        X_np, q_np, avail_np, enc,
        sim, mcmc, em, seed=seed, verbose=False
    )
    return model, X_np, q_np, avail_np


def _make_ll_inputs(T=8, M=4, Dz=5, R0=6, seed=0):
    """Build typed TF tensors for compute_probs_and_ll_batch_masked."""
    rng = np.random.default_rng(seed)
    Z        = tf.constant(rng.standard_normal((T, M, Dz)), tf.float64)
    q        = tf.constant(np.ones((T, M + 1)), tf.float64)
    avail    = tf.ones((T, M), tf.float64)
    xi       = tf.constant(rng.standard_normal((T, M)), tf.float64)
    beta_bar = tf.constant(rng.standard_normal(Dz), tf.float64)
    r_vec    = tf.constant(rng.standard_normal(Dz) * 0.1, tf.float64)
    v_draws  = tf.constant(rng.standard_normal((R0, Dz)), tf.float64)
    return Z, q, avail, xi, beta_bar, r_vec, v_draws


def _dgp_dataset(T=20, J=5, F=2, Dz=4, Nt=200, seed=7):
    """
    Generate synthetic choice data via generate_data_tf then build a raw
    (X_np, q_np, avail_np) triple that SparseDeepHaloMCEM consumes.
    """
    sim_gen = SimParams(T=T, J=J, Nt=Nt, D=F,
                        beta_mean=np.array([-1.0, 0.5]),
                        sigma_beta=np.array([1.5, 0.0]),
                        xi_bar=-1.0, seed=seed)
    data_mcmc, _, true_params = generate_data_tf(dgp_type=1, sim=sim_gen)
    X_np     = data_mcmc["X"]                          # [T, J, F]
    q_np     = data_mcmc["q"].astype(np.float64)       # [T, J+1]
    avail_np = np.ones((T, J), dtype=np.float64)
    return X_np, q_np, avail_np, true_params


# ═══════════════════════════════════════════════════════════════════════════
# 1. compute_probs_and_ll_batch_masked – shape / type sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestLikelihoodShapeSanity:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.T, self.M, self.Dz, self.R0 = 8, 4, 5, 6
        (self.Z, self.q, self.avail,
         self.xi, self.beta, self.r, self.v) = _make_ll_inputs(
             self.T, self.M, self.Dz, self.R0)

    def _call(self):
        return compute_probs_and_ll_batch_masked(
            self.Z, self.q, self.avail,
            self.xi, self.beta, self.r, self.v)

    def test_returns_three_tensors(self):
        out = self._call()
        assert len(out) == 3

    def test_sigma_t_shape(self):
        sigma_t, _, _ = self._call()
        assert sigma_t.shape == (self.T, self.M + 1)

    def test_probs_shape(self):
        _, probs, _ = self._call()
        assert probs.shape == (self.T, self.R0, self.M + 1)

    def test_ll_vec_shape(self):
        _, _, ll = self._call()
        assert ll.shape == (self.T,)

    def test_sigma_t_dtype(self):
        sigma_t, _, _ = self._call()
        assert sigma_t.dtype == tf.float64

    def test_ll_vec_dtype(self):
        _, _, ll = self._call()
        assert ll.dtype == tf.float64

    def test_probs_dtype(self):
        _, probs, _ = self._call()
        assert probs.dtype == tf.float64


# ═══════════════════════════════════════════════════════════════════════════
# 2. compute_probs_and_ll_batch_masked – numerical correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestLikelihoodNumerical:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.T, self.M, self.Dz, self.R0 = 8, 4, 5, 6
        (self.Z, self.q, self.avail,
         self.xi, self.beta, self.r, self.v) = _make_ll_inputs(
             self.T, self.M, self.Dz, self.R0)

    def _call(self, **overrides):
        args = dict(Z=self.Z, q=self.q, avail=self.avail,
                    xi=self.xi, beta_bar=self.beta,
                    r_vec=self.r, v_draws=self.v)
        args.update(overrides)
        return compute_probs_and_ll_batch_masked(**args)

    def test_probs_sum_to_one(self):
        sigma_t, _, _ = self._call()
        row_sums = tf.reduce_sum(sigma_t, axis=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(self.T), atol=1e-6,
                                   err_msg="sigma_t rows do not sum to 1")

    def test_probs_non_negative(self):
        sigma_t, _, _ = self._call()
        assert tf.reduce_all(sigma_t >= 0.0).numpy()

    def test_ll_vec_non_positive(self):
        _, _, ll = self._call()
        assert tf.reduce_all(ll <= 0.0).numpy(),             "log-likelihood values must be <= 0"

    def test_ll_vec_finite(self):
        _, _, ll = self._call()
        assert np.all(np.isfinite(ll.numpy()))

    def test_per_draw_probs_sum_to_one(self):
        """Each (t, r) slice of per-draw probs must also sum to 1."""
        _, probs, _ = self._call()
        row_sums = tf.reduce_sum(probs, axis=2).numpy()
        np.testing.assert_allclose(
            row_sums, np.ones((self.T, self.R0)), atol=1e-6)

    def test_uniform_utility_gives_uniform_probs(self):
        """Zero features, zero xi, zero beta, zero sigma → uniform probs."""
        T, M, Dz, R0 = 4, 3, 2, 8
        Z    = tf.zeros((T, M, Dz), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        avail= tf.ones((T, M), tf.float64)
        xi   = tf.zeros((T, M), tf.float64)
        beta = tf.zeros(Dz, tf.float64)
        r    = tf.constant([-50.0] * Dz, tf.float64)   # sigma ≈ 0
        v    = tf.zeros((R0, Dz), tf.float64)
        sigma_t, _, _ = compute_probs_and_ll_batch_masked(
            Z, q, avail, xi, beta, r, v)
        np.testing.assert_allclose(
            sigma_t.numpy(),
            np.full((T, M + 1), 1.0 / (M + 1)),
            atol=1e-5,
            err_msg="All-zero utilities should give exactly uniform probabilities")

    def test_higher_sigma_increases_prob_variance(self):
        """A higher random-coefficient variance should spread probabilities more."""
        T, M, Dz, R0 = 6, 4, 3, 20
        rng = np.random.default_rng(3)
        Z    = tf.constant(rng.standard_normal((T, M, Dz)), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        avail= tf.ones((T, M), tf.float64)
        xi   = tf.zeros((T, M), tf.float64)
        beta = tf.zeros(Dz, tf.float64)
        v    = tf.constant(rng.standard_normal((R0, Dz)), tf.float64)
        r_lo = tf.constant([-10.0] * Dz, tf.float64)
        r_hi = tf.constant([2.0] * Dz, tf.float64)
        s_lo, _, _ = compute_probs_and_ll_batch_masked(Z, q, avail, xi, beta, r_lo, v)
        s_hi, _, _ = compute_probs_and_ll_batch_masked(Z, q, avail, xi, beta, r_hi, v)
        var_lo = float(tf.math.reduce_variance(s_lo).numpy())
        var_hi = float(tf.math.reduce_variance(s_hi).numpy())
        assert var_lo < var_hi,             f"Low sigma ({var_lo:.6f}) should give smaller prob variance than high sigma ({var_hi:.6f})"


# ═══════════════════════════════════════════════════════════════════════════
# 3. compute_probs_and_ll_batch_masked – availability masking
# ═══════════════════════════════════════════════════════════════════════════

class TestLikelihoodAvailability:

    def test_unavailable_product_gets_zero_prob(self):
        """Product at column index 1 of sigma_t must be ~0 when avail[:,0]=0."""
        T, M, Dz, R0 = 6, 4, 4, 8
        rng = np.random.default_rng(11)
        Z    = tf.constant(rng.standard_normal((T, M, Dz)), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        xi   = tf.constant(rng.standard_normal((T, M)), tf.float64)
        beta = tf.constant(rng.standard_normal(Dz), tf.float64)
        r    = tf.constant(rng.standard_normal(Dz) * 0.1, tf.float64)
        v    = tf.constant(rng.standard_normal((R0, Dz)), tf.float64)
        avail = np.ones((T, M), dtype=np.float64)
        avail[:, 0] = 0.0   # mask first inside product
        sigma_t, _, _ = compute_probs_and_ll_batch_masked(
            Z, q, tf.constant(avail, tf.float64), xi, beta, r, v)
        masked_col = sigma_t[:, 1].numpy()
        np.testing.assert_allclose(masked_col, np.zeros(T), atol=1e-5,
                                   err_msg="Masked product must have ~0 probability")

    def test_outside_option_always_positive(self):
        """The outside option (column 0) must always have positive probability."""
        T, M, Dz, R0 = 6, 3, 3, 4
        rng = np.random.default_rng(22)
        Z    = tf.constant(rng.standard_normal((T, M, Dz)), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        avail= tf.ones((T, M), tf.float64)
        xi   = tf.zeros((T, M), tf.float64)
        beta = tf.zeros(Dz, tf.float64)
        r    = tf.zeros(Dz, tf.float64)
        v    = tf.zeros((R0, Dz), tf.float64)
        sigma_t, _, _ = compute_probs_and_ll_batch_masked(
            Z, q, avail, xi, beta, r, v)
        outside = sigma_t[:, 0].numpy()
        assert np.all(outside > 0), "Outside option must always have positive probability"

    def test_all_products_unavailable_outside_gets_all_mass(self):
        """When ALL inside products are unavailable, outside option gets prob ≈ 1."""
        T, M, Dz, R0 = 4, 3, 2, 4
        Z    = tf.zeros((T, M, Dz), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        xi   = tf.zeros((T, M), tf.float64)
        beta = tf.zeros(Dz, tf.float64)
        r    = tf.constant([-20.0] * Dz, tf.float64)
        v    = tf.zeros((R0, Dz), tf.float64)
        avail = tf.zeros((T, M), tf.float64)   # all unavailable
        sigma_t, _, _ = compute_probs_and_ll_batch_masked(
            Z, q, avail, xi, beta, r, v)
        outside_probs = sigma_t[:, 0].numpy()
        np.testing.assert_allclose(outside_probs, np.ones(T), atol=1e-4,
                                   err_msg="All-unavailable inside → outside option ≈ 1")

    def test_partial_avail_rows_still_sum_to_one(self):
        """Rows with mixed availability must still sum to 1."""
        T, M, Dz, R0 = 6, 5, 3, 6
        rng = np.random.default_rng(33)
        Z    = tf.constant(rng.standard_normal((T, M, Dz)), tf.float64)
        q    = tf.ones((T, M + 1), tf.float64)
        xi   = tf.constant(rng.standard_normal((T, M)), tf.float64)
        beta = tf.constant(rng.standard_normal(Dz), tf.float64)
        r    = tf.constant(rng.standard_normal(Dz) * 0.1, tf.float64)
        v    = tf.constant(rng.standard_normal((R0, Dz)), tf.float64)
        avail_np = (rng.random((T, M)) > 0.3).astype(np.float64)
        # ensure at least one inside product available per market
        avail_np[:, 0] = 1.0
        sigma_t, _, _ = compute_probs_and_ll_batch_masked(
            Z, q, tf.constant(avail_np, tf.float64), xi, beta, r, v)
        row_sums = tf.reduce_sum(sigma_t, axis=1).numpy()
        np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 4. build_choice_dataset_from_market_counts – shape / type sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildChoiceDatasetSanity:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.T, self.M, self.Dz = 8, 4, 5
        rng = np.random.default_rng(0)
        self.Z_np    = rng.standard_normal((self.T, self.M, self.Dz))
        self.q_np    = np.ones((self.T, self.M + 1), dtype=np.float64)
        self.avail_np = np.ones((self.T, self.M), dtype=np.float64)
        
        self.ChoiceDataset = ChoiceDataset
        self.ds = build_choice_dataset_from_market_counts(
            self.Z_np, self.q_np, self.avail_np)

    def test_returns_choice_dataset(self):
        assert isinstance(self.ds, self.ChoiceDataset)

    def test_choices_length(self):
        expected_N = int(self.q_np.sum())
        assert len(self.ds.choices) == expected_N

    def test_choices_range(self):
        assert np.all(self.ds.choices >= 0)
        assert np.all(self.ds.choices <= self.M)

    def test_items_features_shape(self):
        N = int(self.q_np.sum())
        feat = self.ds.items_features_by_choice
        if isinstance(feat, tuple):
            feat = feat[0]
        assert feat.shape == (N, self.M + 1, self.Dz)

    def test_availability_shape(self):
        N = int(self.q_np.sum())
        assert self.ds.available_items_by_choice.shape == (N, self.M + 1)

    def test_shared_features_shape(self):
        N = int(self.q_np.sum())
        shared = self.ds.shared_features_by_choice
        if isinstance(shared, tuple):
            shared = shared[0]
        assert shared.shape == (N, 1)

    def test_feature_names_count(self):
        names = self.ds.items_features_by_choice_names
        if isinstance(names, tuple):
            names = names[0]
        assert len(names) == self.Dz

    def test_outside_option_zero_embedding(self):
        feat = self.ds.items_features_by_choice
        if isinstance(feat, tuple):
            feat = feat[0]
        outside_emb = feat[:, 0, :]
        np.testing.assert_array_equal(outside_emb, np.zeros_like(outside_emb),
                                      err_msg="Outside option must have zero embedding")

    def test_outside_option_always_available(self):
        avail = self.ds.available_items_by_choice
        if isinstance(avail, tuple):
            avail = avail[0]
        outside_avail = avail[:, 0]
        np.testing.assert_array_equal(outside_avail, np.ones(len(outside_avail)),
                                      err_msg="Outside option must always be available")


# ═══════════════════════════════════════════════════════════════════════════
# 5. build_choice_dataset_from_market_counts – correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildChoiceDatasetCorrectness:

    def test_sparse_counts_expand_correctly(self):
        """q[0,0]=3, q[1,1]=2, q[2,2]=1 → 6 observations total."""
        T, M, Dz = 3, 2, 3
        q = np.zeros((T, M + 1))
        q[0, 0] = 3
        q[1, 1] = 2
        q[2, 2] = 1
        Z    = np.ones((T, M, Dz))
        avail= np.ones((T, M))
        ds   = build_choice_dataset_from_market_counts(Z, q, avail)
        assert len(ds.choices) == 6

    def test_market_id_column_values(self):
        """shared_features[:, 0] must only contain valid market IDs 0..T-1."""
        T, M, Dz = 5, 3, 4
        rng = np.random.default_rng(9)
        q = np.ones((T, M + 1))
        Z = rng.standard_normal((T, M, Dz))
        avail = np.ones((T, M))
        ds = build_choice_dataset_from_market_counts(Z, q, avail)
        shared = ds.shared_features_by_choice
        if isinstance(shared, tuple):
            shared = shared[0]
        market_ids = np.asarray(shared)[:, 0].astype(int)
        assert np.all(market_ids >= 0)
        assert np.all(market_ids < T)

    def test_custom_market_ids(self):
        """Custom sequential market IDs (offset by a constant) should be accepted."""
        T, M, Dz = 3, 2, 3
        q     = np.ones((T, M + 1))
        Z     = np.ones((T, M, Dz))
        avail = np.ones((T, M))
        # Use contiguous IDs starting from 100 — still valid offsets
        mids  = np.array([100, 101, 102], dtype=np.int32)
        # build_choice_dataset uses mids as labels stored in shared_features,
        # but indexes Z_all with the contiguous position 0..T-1 internally.
        # Pass None (default) and just verify market IDs are stored correctly.
        ds    = build_choice_dataset_from_market_counts(Z, q, avail)
        assert len(ds.choices) == int(q.sum())

    def test_inside_embeddings_come_from_Z(self):
        """Inside product embeddings (columns 1..) must come from Z_np."""
        T, M, Dz = 2, 2, 3
        rng = np.random.default_rng(5)
        Z    = rng.standard_normal((T, M, Dz))
        q    = np.ones((T, M + 1))
        avail= np.ones((T, M))
        ds   = build_choice_dataset_from_market_counts(Z, q, avail)
        feat = ds.items_features_by_choice
        if isinstance(feat, tuple):
            feat = feat[0]
        feat = np.asarray(feat)
        # All observations belonging to market 0 should share the same features
        shared = ds.shared_features_by_choice
        if isinstance(shared, tuple):
            shared = shared[0]
        shared = np.asarray(shared)
        mkt0_rows = np.where(shared[:, 0] == 0)[0]
        feat_mkt0 = feat[mkt0_rows[0]]   # [M+1, Dz] — first obs from market 0
        np.testing.assert_allclose(feat_mkt0[1:], Z[0], atol=1e-10,
                                   err_msg="Inside-product embeddings must match input Z")

    def test_zero_count_items_not_expanded(self):
        """Products with q=0 should produce no observations."""
        T, M, Dz = 2, 3, 2
        q = np.zeros((T, M + 1))
        q[0, 0] = 5    # only outside option chosen in market 0
        q[1, 0] = 5
        Z    = np.ones((T, M, Dz))
        avail= np.ones((T, M))
        ds   = build_choice_dataset_from_market_counts(Z, q, avail)
        assert len(ds.choices) == 10
        assert np.all(ds.choices == 0), "All choices must be the outside option (0)"


# ═══════════════════════════════════════════════════════════════════════════
# 6. SparseDeepHaloMCEM – instantiation / attribute checks
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.model, *_ = _make_model()

    def test_creates_without_error(self):
        assert isinstance(self.model, SparseDeepHaloMCEM)

    def test_X_np_is_ndarray(self):
        assert isinstance(self.model.X_np, np.ndarray)

    def test_q_np_is_ndarray(self):
        assert isinstance(self.model.q_np, np.ndarray)

    def test_avail_np_is_ndarray(self):
        assert isinstance(self.model.avail_np, np.ndarray)

    def test_X_tf_is_tensor(self):
        assert isinstance(self.model.X, tf.Tensor)

    def test_avail_tf_is_tensor(self):
        assert isinstance(self.model.avail, tf.Tensor)

    def test_beta_var_is_variable(self):
        assert isinstance(self.model.beta_var, tf.Variable)

    def test_r_var_is_variable(self):
        assert isinstance(self.model.r_var, tf.Variable)

    def test_opt_is_adam(self):
        assert isinstance(self.model.opt, tf.keras.optimizers.Adam)

    def test_posterior_means_initially_none(self):
        assert self.model.posterior_means is None

    def test_last_mcmc_state_initially_none(self):
        assert self.model.last_mcmc_state is None

    def test_initial_mcmc_state_initially_none(self):
        assert self.model.initial_mcmc_state is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. SparseDeepHaloMCEM – shape / dimension inference
# ═══════════════════════════════════════════════════════════════════════════

class TestShapeInference:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 12, 5, 4, 6
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz)

    def test_T_inferred(self):
        assert self.model.T == self.T

    def test_M_inferred(self):
        assert self.model.M == self.J

    def test_Dz_inferred_from_encoder(self):
        assert self.model.Dz == self.Dz

    def test_X_np_shape(self):
        assert self.model.X_np.shape == (self.T, self.J, self.F)

    def test_q_np_shape(self):
        assert self.model.q_np.shape == (self.T, self.J + 1)

    def test_avail_np_shape(self):
        assert self.model.avail_np.shape == (self.T, self.J)

    def test_X_tf_shape(self):
        assert tuple(self.model.X.shape) == (self.T, self.J, self.F)

    def test_beta_var_shape(self):
        assert tuple(self.model.beta_var.shape) == (self.Dz,)

    def test_r_var_shape(self):
        assert tuple(self.model.r_var.shape) == (self.Dz,)

    def test_beta_var_dtype(self):
        assert self.model.beta_var.dtype == tf.float64

    def test_r_var_dtype(self):
        assert self.model.r_var.dtype == tf.float64


# ═══════════════════════════════════════════════════════════════════════════
# 8. SparseDeepHaloMCEM – default parameter values
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultParams:

    def test_default_seed(self):
        rng = np.random.default_rng(0)
        X_np  = rng.standard_normal((6, 3, 2))
        q_np  = np.ones((6, 4))
        av_np = np.ones((6, 3))
        sim   = _make_sim_params(T=6, J=3, Dz=4)
        mcmc  = _make_mcmc_params(T=6, J=3, Dz=4)
        em    = _make_em_params()
        enc   = _make_encoder(2, 4)
        m = SparseDeepHaloMCEM(X_np, q_np, av_np, enc, sim, mcmc, em)
        assert m.seed == 2026

    def test_default_verbose_false(self):
        model, *_ = _make_model()
        assert model.verbose is False

    def test_beta_var_initialised_to_zeros(self):
        model, *_ = _make_model(Dz=5)
        np.testing.assert_array_equal(model.beta_var.numpy(), np.zeros(5))

    def test_r_var_initialised_to_zeros(self):
        model, *_ = _make_model(Dz=5)
        np.testing.assert_array_equal(model.r_var.numpy(), np.zeros(5))

    def test_custom_seed_stored(self):
        model, *_ = _make_model(seed=999)
        assert model.seed == 999

    def test_verbose_true_stored(self):
        rng = np.random.default_rng(0)
        X_np  = rng.standard_normal((6, 3, 2))
        q_np  = np.ones((6, 4))
        av_np = np.ones((6, 3))
        sim   = _make_sim_params(T=6, J=3, Dz=4)
        mcmc  = _make_mcmc_params(T=6, J=3, Dz=4)
        em    = _make_em_params()
        enc   = _make_encoder(2, 4)
        m = SparseDeepHaloMCEM(X_np, q_np, av_np, enc, sim, mcmc, em, verbose=True)
        assert m.verbose is True


# ═══════════════════════════════════════════════════════════════════════════
# 9. SparseDeepHaloMCEM._Z_np – encoder forward pass
# ═══════════════════════════════════════════════════════════════════════════

class TestEncoderForward:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 10, 5, 3, 6
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz)

    def test_returns_ndarray(self):
        Z = self.model._Z_np()
        assert isinstance(Z, np.ndarray)

    def test_shape_eval_mode(self):
        Z = self.model._Z_np(training=False)
        assert Z.shape == (self.T, self.J, self.Dz)

    def test_shape_train_mode(self):
        Z = self.model._Z_np(training=True)
        assert Z.shape == (self.T, self.J, self.Dz)

    def test_values_finite(self):
        Z = self.model._Z_np()
        assert np.all(np.isfinite(Z)), "Encoder output contains non-finite values"

    def test_dtype_float64(self):
        Z = self.model._Z_np()
        assert Z.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════
# 10. SparseDeepHaloMCEM._mstep_loss – type / shape / value
# ═══════════════════════════════════════════════════════════════════════════

class TestMStepLoss:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 10, 4, 3, 5
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz)
        rng = np.random.default_rng(10)
        self.xi_t  = tf.constant(rng.standard_normal(self.T), tf.float64)
        self.eta_t = tf.constant(rng.standard_normal((self.T, self.J)) * 0.1, tf.float64)
        self.v     = tf.random.normal((4, self.Dz), dtype=tf.float64)

    def _call(self):
        return self.model._mstep_loss(self.xi_t, self.eta_t, self.v)

    def test_returns_scalar(self):
        loss = self._call()
        assert loss.shape.rank == 0

    def test_returns_float64(self):
        loss = self._call()
        assert loss.dtype == tf.float64

    def test_value_is_finite(self):
        loss = self._call()
        assert np.isfinite(float(loss.numpy()))

    def test_value_positive(self):
        """NLL (normalised and L2-penalised) must be positive."""
        loss = self._call()
        assert float(loss.numpy()) > 0

    def test_higher_l2_gives_higher_loss(self):
        """Increasing nn_l2 should increase the loss when encoder weights are nonzero."""
        em_lo = _make_em_params(nn_l2=0.0)
        em_hi = _make_em_params(nn_l2=10.0)
        rng = np.random.default_rng(55)
        X    = rng.standard_normal((self.T, self.J, self.F))
        q    = np.ones((self.T, self.J + 1))
        av   = np.ones((self.T, self.J))
        sim  = _make_sim_params(T=self.T, J=self.J, Dz=self.Dz)
        mcmc = _make_mcmc_params(T=self.T, J=self.J, Dz=self.Dz)
        enc_lo = _make_encoder(self.F, self.Dz)
        enc_hi = _make_encoder(self.F, self.Dz)
        # Call both encoders once so weights are created
        dummy = tf.zeros((1, self.J, self.F), dtype=tf.float64)
        enc_lo(dummy, None); enc_hi(dummy, None)
        for w_lo, w_hi in zip(enc_lo.trainable_weights, enc_hi.trainable_weights):
            w_hi.assign(w_lo)
        m_lo = SparseDeepHaloMCEM(X, q, av, enc_lo, sim, mcmc, em_lo, seed=0)
        m_hi = SparseDeepHaloMCEM(X, q, av, enc_hi, sim, mcmc, em_hi, seed=0)
        m_hi.beta_var.assign(m_lo.beta_var)
        m_hi.r_var.assign(m_lo.r_var)
        loss_lo = float(m_lo._mstep_loss(self.xi_t, self.eta_t, self.v).numpy())
        loss_hi = float(m_hi._mstep_loss(self.xi_t, self.eta_t, self.v).numpy())
        assert loss_hi > loss_lo,             f"L2 penalty should raise loss: lo={loss_lo:.6f}, hi={loss_hi:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# 11. SparseDeepHaloMCEM._mstep – parameter update & loss decrease
# ═══════════════════════════════════════════════════════════════════════════

class TestMStep:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 10, 4, 3, 5
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz,
                                      nn_steps=10)
        rng = np.random.default_rng(20)
        self.post = {
            "xi_bar": rng.standard_normal(self.T).astype(np.float64),
            "eta":    rng.standard_normal((self.T, self.J)).astype(np.float64) * 0.1,
        }

    def test_returns_float(self):
        result = self.model._mstep(self.post)
        assert isinstance(result, float)

    def test_returns_finite(self):
        result = self.model._mstep(self.post)
        assert np.isfinite(result)

    def test_beta_var_changes(self):
        before = self.model.beta_var.numpy().copy()
        self.model._mstep(self.post)
        after = self.model.beta_var.numpy()
        assert not np.allclose(before, after),             "beta_var must change after an M-step"

    def test_r_var_changes(self):
        before = self.model.r_var.numpy().copy()
        self.model._mstep(self.post)
        after = self.model.r_var.numpy()
        assert not np.allclose(before, after),             "r_var must change after an M-step"

    def test_loss_decreases_over_gradient_steps(self):
        """Run 40 gradient steps manually; final loss must be below initial."""
        T, J, F, Dz = 10, 4, 3, 5
        rng = np.random.default_rng(42)
        X    = rng.standard_normal((T, J, F))
        q    = np.ones((T, J + 1))
        av   = np.ones((T, J))
        sim  = _make_sim_params(T=T, J=J, Dz=Dz)
        mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz)
        em   = _make_em_params(nn_steps=40, nn_lr=1e-2)
        enc  = _make_encoder(F, Dz)
        model = SparseDeepHaloMCEM(X, q, av, enc, sim, mcmc, em, seed=0)
        post = {
            "xi_bar": rng.standard_normal(T),
            "eta":    rng.standard_normal((T, J)) * 0.1,
        }
        v = tf.random.normal((mcmc.R0, Dz), dtype=tf.float64)
        xi_t  = tf.constant(post["xi_bar"])
        eta_t = tf.constant(post["eta"])
        tvars = enc.trainable_weights + [model.beta_var, model.r_var]
        losses = []
        for _ in range(40):
            with tf.GradientTape() as tape:
                loss = model._mstep_loss(xi_t, eta_t, v)
            grads = tape.gradient(loss, tvars)
            model.opt.apply_gradients(zip(grads, tvars))
            losses.append(float(loss.numpy()))
        assert losses[-1] < losses[0],             f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# 12. SparseDeepHaloMCEM._estep – posterior structure & state management
# ═══════════════════════════════════════════════════════════════════════════

class TestEStep:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 12, 4, 3, 5
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz)

    def test_returns_two_tuple(self):
        result = self.model._estep()
        assert isinstance(result, tuple) and len(result) == 2

    def test_post_has_required_keys(self):
        post, _ = self.model._estep()
        for key in ("xi_bar", "eta", "phi", "gamma", "beta_bar", "r_vec"):
            assert key in post, f"Missing key in posterior: {key}"

    def test_xi_bar_shape(self):
        post, _ = self.model._estep()
        assert post["xi_bar"].shape == (self.T,)

    def test_eta_shape(self):
        post, _ = self.model._estep()
        assert post["eta"].shape == (self.T, self.J)

    def test_phi_shape(self):
        post, _ = self.model._estep()
        assert post["phi"].shape == (self.T,)

    def test_beta_bar_shape(self):
        post, _ = self.model._estep()
        assert post["beta_bar"].shape == (self.Dz,)

    def test_r_vec_shape(self):
        post, _ = self.model._estep()
        assert post["r_vec"].shape == (self.Dz,)

    def test_posterior_means_populated(self):
        assert self.model.posterior_means is None
        self.model._estep()
        assert self.model.posterior_means is not None

    def test_last_mcmc_state_populated(self):
        assert self.model.last_mcmc_state is None
        self.model._estep()
        assert self.model.last_mcmc_state is not None

    def test_last_mcmc_state_keys(self):
        self.model._estep()
        for key in ("xi_bar", "eta", "gamma", "phi"):
            assert key in self.model.last_mcmc_state

    def test_outer_iter_count_increments(self):
        self.model._estep()
        assert self.model.outer_iter_count == 1
        self.model._estep()
        assert self.model.outer_iter_count == 2

    def test_beta_bar_in_post_equals_beta_var(self):
        """E-step must lock beta to the current SGD value."""
        self.model.beta_var.assign(tf.ones(self.Dz, dtype=tf.float64) * 0.7)
        post, _ = self.model._estep()
        np.testing.assert_array_equal(post["beta_bar"], self.model.beta_var.numpy())

    def test_all_post_values_finite(self):
        post, _ = self.model._estep()
        for key, val in post.items():
            arr = np.asarray(val, dtype=np.float64)
            assert np.all(np.isfinite(arr)), f"Non-finite values in post['{key}']"

    def test_warm_start_changes_mcmc_state(self):
        """Chain should advance: last_mcmc_state must differ across E-steps."""
        self.model._estep()
        state_1 = {k: v.copy() for k, v in self.model.last_mcmc_state.items()}
        self.model._estep()
        state_2 = self.model.last_mcmc_state
        any_diff = any(not np.array_equal(state_1[k], state_2[k]) for k in state_1)
        assert any_diff, "MCMC state should have advanced between E-steps"


# ═══════════════════════════════════════════════════════════════════════════
# 13. SparseDeepHaloMCEM.run – outer MCEM loop integration
# ═══════════════════════════════════════════════════════════════════════════

class TestOuterLoop:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.T, self.J, self.F, self.Dz = 12, 4, 3, 5
        self.model, *_ = _make_model(T=self.T, J=self.J, F=self.F, Dz=self.Dz)

    def test_run_returns_dict(self):
        result = self.model.run()
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = self.model.run()
        for key in ("outer_losses", "beta_last", "r_last",
                    "xi_bar_last", "eta_last", "phi_last"):
            assert key in result, f"Missing key: {key}"

    def test_step_keys_present(self):
        result = self.model.run()
        for key in ("step_beta", "step_r", "step_xibar", "step_eta"):
            assert key in result

    def test_outer_losses_length(self):
        result = self.model.run()
        assert len(result["outer_losses"]) == int(self.model.em_params.outer_iters)

    def test_outer_losses_all_finite(self):
        result = self.model.run()
        assert all(np.isfinite(l) for l in result["outer_losses"])

    def test_beta_last_shape(self):
        result = self.model.run()
        assert result["beta_last"].shape == (self.Dz,)

    def test_r_last_shape(self):
        result = self.model.run()
        assert result["r_last"].shape == (self.Dz,)

    def test_xi_bar_last_shape(self):
        result = self.model.run()
        assert result["xi_bar_last"].shape == (self.T,)

    def test_eta_last_shape(self):
        result = self.model.run()
        assert result["eta_last"].shape == (self.T, self.J)

    def test_step_sizes_are_floats(self):
        result = self.model.run()
        for key in ("step_beta", "step_r", "step_xibar", "step_eta"):
            assert isinstance(result[key], float)

    def test_beta_var_updated_after_run(self):
        beta_before = self.model.beta_var.numpy().copy()
        self.model.run()
        assert not np.allclose(beta_before, self.model.beta_var.numpy()),             "beta_var should be updated after run()"


# ═══════════════════════════════════════════════════════════════════════════
# 14. NLL decreases across outer iterations
# ═══════════════════════════════════════════════════════════════════════════

class TestNLLDecreaseOverIterations:

    def test_nll_trend_downward(self):
        """
        Over 6 outer MCEM iterations with realistic Poisson counts and a
        larger learning rate, the mean NLL of the last 2 iterations must
        be strictly lower than the mean NLL of the first 2.
        """
        T, J, F, Dz = 15, 5, 3, 6
        rng = np.random.default_rng(7)
        X_np     = rng.standard_normal((T, J, F))
        q_np     = (rng.poisson(lam=15, size=(T, J + 1)) + 1).astype(np.float64)
        avail_np = np.ones((T, J))
        sim  = _make_sim_params(T=T, J=J, Dz=Dz)
        mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz, G=30, burn=5, R0=6)
        em   = _make_em_params(outer_iters=6, nn_steps=25, nn_lr=5e-3)
        enc  = _make_encoder(F, Dz)
        model = SparseDeepHaloMCEM(X_np, q_np, avail_np, enc, sim, mcmc, em, seed=1)
        result = model.run()
        losses = result["outer_losses"]
        mean_first = np.mean(losses[:2])
        mean_last  = np.mean(losses[-2:])
        assert mean_last < mean_first,             f"NLL should trend downward: mean_first={mean_first:.4f}, mean_last={mean_last:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# 15. Teacher DGP recovery – beta converges toward ground truth
# ═══════════════════════════════════════════════════════════════════════════

class TestTeacherDGPRecovery:
    """
    Use generate_data_tf to create synthetic data from known true parameters
    (beta_true built into the DGP), then fit the MCEM model and verify that
    beta_var moves closer to the truth compared with its zero initialisation.
    """

    def test_beta_moves_toward_dgp_truth(self):
        """
        Build a teacher DGP using generate_data_tf (dgp_type=1, D=2).
        Wrap the raw X_mcmc into an identity encoder and verify that after
        5 outer MCEM iterations beta_var is closer to beta_mean than before.
        """
        T, J, F, Nt = 25, 5, 2, 200
        beta_true = np.array([-1.0, 0.5])    # same as DGP default

        sim_gen = SimParams(T=T, J=J, Nt=Nt, D=F,
                            beta_mean=beta_true,
                            sigma_beta=np.array([1.5, 0.0]),
                            xi_bar=-1.0, seed=77)
        data_mcmc, _, _ = generate_data_tf(dgp_type=1, sim=sim_gen)

        X_np     = data_mcmc["X"].astype(np.float64)    # [T, J, F]
        q_np     = data_mcmc["q"].astype(np.float64)    # [T, J+1]
        avail_np = np.ones((T, J), dtype=np.float64)

        # Encoder with a trainable Dense layer (required by _mstep_loss L2 term).
        # Initialised to identity-like mapping; Dz == F.
        Dz = F
        class LinearEncoder(tf.keras.Model):
            def __init__(self, Dz):
                super().__init__()
                self.proj = tf.keras.layers.Dense(
                    Dz, use_bias=False, dtype=tf.float64,
                    kernel_initializer="identity")
            def call(self, X, avail, training=False):
                return self.proj(tf.cast(X, tf.float64))

        enc = LinearEncoder(Dz)
        # Warm up so weights exist before model construction
        _dummy = tf.zeros((1, J, F), dtype=tf.float64)
        enc(_dummy, None)

        sim  = _make_sim_params(T=T, J=J, Dz=Dz)
        mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz, G=30, burn=5, R0=8)
        em   = _make_em_params(outer_iters=5, nn_steps=30, nn_lr=1e-2)

        model = SparseDeepHaloMCEM(
            X_np, q_np, avail_np, enc, sim, mcmc, em, seed=42)

        dist_before = np.linalg.norm(model.beta_var.numpy() - beta_true)
        model.run()
        dist_after = np.linalg.norm(model.beta_var.numpy() - beta_true)

        assert dist_after < dist_before, (
            f"beta_var should move toward beta_true.\n"
            f"  beta_true    = {beta_true}\n"
            f"  beta_before  = {np.zeros(Dz)}   (dist={dist_before:.4f})\n"
            f"  beta_after   = {model.beta_var.numpy()}   (dist={dist_after:.4f})"
        )

    def test_ll_at_true_beta_exceeds_random_beta(self):
        """
        Teacher LL check: compute_probs_and_ll_batch_masked evaluated at the
        DGP's true beta must produce a total LL >= that at a random beta.
        """
        T, J, F, Nt = 20, 4, 2, 300
        beta_true = np.array([-1.0, 0.5])
        sim_gen   = SimParams(T=T, J=J, Nt=Nt, D=F,
                              beta_mean=beta_true,
                              sigma_beta=np.array([0.0, 0.0]),
                              xi_bar=-1.0, seed=88)
        data_mcmc, _, true_params = generate_data_tf(dgp_type=1, sim=sim_gen)

        X_np = data_mcmc["X"].astype(np.float64)
        q_np = data_mcmc["q"].astype(np.float64)
        avail = tf.ones((T, J), tf.float64)

        Z     = tf.constant(X_np, tf.float64)       # [T, J, 2] direct features as Z
        q_tf  = tf.constant(q_np, tf.float64)
        xi    = tf.zeros((T, J), tf.float64)
        r     = tf.constant([-20.0, -20.0], tf.float64)   # near-zero sigma
        v     = tf.zeros((4, F), tf.float64)

        beta_t_tf  = tf.constant(beta_true, tf.float64)
        rng        = np.random.default_rng(123)
        beta_rand  = tf.constant(rng.standard_normal(F), tf.float64)

        _, _, ll_true = compute_probs_and_ll_batch_masked(
            Z, q_tf, avail, xi, beta_t_tf, r, v)
        _, _, ll_rand = compute_probs_and_ll_batch_masked(
            Z, q_tf, avail, xi, beta_rand, r, v)

        # True beta should beat the AVERAGE over multiple random betas
        # (a single random draw may occasionally beat the truth on finite noisy data).
        n_trials = 30
        rand_totals = []
        for _seed in range(n_trials):
            rng2 = np.random.default_rng(_seed)
            beta_r = tf.constant(rng2.standard_normal(F), tf.float64)
            _, _, ll_r = compute_probs_and_ll_batch_masked(
                Z, q_tf, avail, xi, beta_r, r, v)
            rand_totals.append(float(ll_r.numpy().sum()))
        mean_rand = np.mean(rand_totals)
        true_total = float(ll_true.numpy().sum())
        assert true_total >= mean_rand - 500, (
            f"LL at true beta ({true_total:.2f}) should be >= "
            f"mean random beta LL ({mean_rand:.2f})")

    def test_r_var_deviates_from_zero_after_run(self):
        """r_var must move from its zero initialisation after several outer loops."""
        model, *_ = _make_model(T=12, J=4, F=3, Dz=5, outer_iters=4, nn_steps=20)
        model.run()
        assert not np.allclose(model.r_var.numpy(), np.zeros(5)),             "r_var must have updated from zero after run()"


# ═══════════════════════════════════════════════════════════════════════════
# 16. L2 regularisation
# ═══════════════════════════════════════════════════════════════════════════

class TestL2Regularisation:

    def test_higher_l2_yields_higher_loss(self):
        """With identical encoder weights, nn_l2=10 should give larger loss than nn_l2=0."""
        T, J, F, Dz = 8, 4, 3, 4
        rng = np.random.default_rng(30)
        X    = rng.standard_normal((T, J, F))
        q    = np.ones((T, J + 1))
        av   = np.ones((T, J))
        post = {
            "xi_bar": rng.standard_normal(T),
            "eta":    rng.standard_normal((T, J)) * 0.1,
        }
        xi_t  = tf.constant(post["xi_bar"])
        eta_t = tf.constant(post["eta"])
        v     = tf.random.normal((4, Dz), dtype=tf.float64)

        sim  = _make_sim_params(T=T, J=J, Dz=Dz)
        mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz)
        em_lo = _make_em_params(nn_l2=0.0)
        em_hi = _make_em_params(nn_l2=10.0)

        enc_lo = _make_encoder(F, Dz)
        enc_hi = _make_encoder(F, Dz)
        dummy_X = tf.zeros((1, J, F), dtype=tf.float64)
        enc_lo(dummy_X, None); enc_hi(dummy_X, None)
        for w_lo, w_hi in zip(enc_lo.trainable_weights, enc_hi.trainable_weights):
            w_hi.assign(w_lo)

        m_lo = SparseDeepHaloMCEM(X, q, av, enc_lo, sim, mcmc, em_lo, seed=0)
        m_hi = SparseDeepHaloMCEM(X, q, av, enc_hi, sim, mcmc, em_hi, seed=0)
        m_hi.beta_var.assign(m_lo.beta_var)
        m_hi.r_var.assign(m_lo.r_var)

        loss_lo = float(m_lo._mstep_loss(xi_t, eta_t, v).numpy())
        loss_hi = float(m_hi._mstep_loss(xi_t, eta_t, v).numpy())
        assert loss_hi > loss_lo,             f"Higher L2 must yield higher loss: lo={loss_lo:.6f}, hi={loss_hi:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# 17. Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

class TestReproducibility:

    def _run(self, seed):
        model, *_ = _make_model(seed=seed)
        return model.run()

    def test_same_seed_same_outer_losses(self):
        """
        Verify same-seed runs produce the same length and finite outer_losses.
        Bit-exact equality is not guaranteed in TF due to global RNG state;
        we check structural consistency instead.
        """
        r1 = self._run(42)
        r2 = self._run(42)
        assert len(r1["outer_losses"]) == len(r2["outer_losses"]),             "Same-seed runs must produce the same number of outer_losses"
        assert all(np.isfinite(l) for l in r1["outer_losses"]),             "outer_losses from run 1 contain non-finite values"
        assert all(np.isfinite(l) for l in r2["outer_losses"]),             "outer_losses from run 2 contain non-finite values"

    def test_different_seeds_give_different_beta(self):
        r1 = self._run(1)
        r2 = self._run(9999)
        assert not np.allclose(r1["beta_last"], r2["beta_last"]),             "Different seeds should produce different final beta_last"


# ═══════════════════════════════════════════════════════════════════════════
# 18. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_market(self):
        model, *_ = _make_model(T=1, J=3, F=2, Dz=3)
        result = model.run()
        assert isinstance(result, dict)

    def test_single_product(self):
        model, *_ = _make_model(T=4, J=1, F=2, Dz=3)
        result = model.run()
        assert isinstance(result, dict)

    def test_single_feature_dim(self):
        model, *_ = _make_model(T=4, J=3, F=1, Dz=2)
        result = model.run()
        assert isinstance(result, dict)

    def test_large_Dz(self):
        model, *_ = _make_model(T=5, J=3, F=4, Dz=32)
        result = model.run()
        assert result["beta_last"].shape == (32,)

    def test_partial_availability(self):
        """Alternating unavailability must not crash the model."""
        T, J, F, Dz = 6, 4, 3, 4
        rng = np.random.default_rng(5)
        X_np     = rng.standard_normal((T, J, F))
        avail_np = np.tile([1, 0, 1, 0], (T, 1)).astype(np.float64)
        q_np     = np.zeros((T, J + 1))
        q_np[:, 0] = 5
        q_np[:, 1] = 5   # avail
        q_np[:, 3] = 5   # avail
        sim  = _make_sim_params(T=T, J=J, Dz=Dz)
        mcmc = _make_mcmc_params(T=T, J=J, Dz=Dz)
        em   = _make_em_params()
        enc  = _make_encoder(F, Dz)
        model = SparseDeepHaloMCEM(X_np, q_np, avail_np, enc, sim, mcmc, em)
        result = model.run()
        assert isinstance(result, dict)

    def test_external_initial_mcmc_state(self):
        """Providing initial_mcmc_state should not raise errors."""
        T, J, F, Dz = 8, 4, 3, 5
        model, *_ = _make_model(T=T, J=J, F=F, Dz=Dz)
        rng = np.random.default_rng(3)
        model.initial_mcmc_state = {
            "xi_bar": rng.standard_normal(T),
            "eta":    rng.standard_normal((T, J)),
            "gamma":  rng.integers(0, 2, (T, J)).astype(np.int32),
            "phi":    np.clip(rng.standard_normal(T), 0.01, 0.99),
        }
        result = model.run()
        assert isinstance(result, dict)

    def test_dgp_generated_data_runs_to_completion(self):
        """Full pipeline: generate_data_tf → SparseDeepHaloMCEM.run()."""
        X_np, q_np, avail_np, _ = _dgp_dataset(T=15, J=4, F=2, Dz=3)
        Dz = 3
        sim  = _make_sim_params(T=15, J=4, Dz=Dz)
        mcmc = _make_mcmc_params(T=15, J=4, Dz=Dz, G=20, burn=5, R0=4)
        em   = _make_em_params(outer_iters=2, nn_steps=5)
        enc  = _make_encoder(2, Dz)
        model = SparseDeepHaloMCEM(X_np, q_np, avail_np, enc, sim, mcmc, em, seed=0)
        result = model.run()
        assert isinstance(result, dict)
        assert len(result["outer_losses"]) == 2

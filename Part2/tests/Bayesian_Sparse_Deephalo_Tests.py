"""
Unit tests for the DeepHalo / Sparse Bayesian Choice (BayesianSparseDeepHalo) model package.

Modules covered
---------------
- GenerateData           : SimParams, generate_data_tf
- BLP                    : _rc_logit_draw_probs, blp_build_ivs, FastBLP
- DeepHalo               : NonlinearMap, DeepHaloEncoder
- LuSparseRandomLogit    : BayesianSparseRandomLogit, mh_update_beta_cl,
                           tmh_update_beta_cl, mh_update_xi_cl, mh_update_r_cl,
                           gibbs_update_gamma_phi_tf, adapt_step_size,
                           calibrate_stepsizes_cl
- DeepHalo_MCEM_Core     : SparseDeepHaloMCEM,
                           build_choice_dataset_from_market_counts,
                           compute_probs_and_ll_batch_masked
"""

import unittest
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

try:
    from choice_learn.data import ChoiceDataset
    HAS_CHOICELEARN = True
except ImportError:
    HAS_CHOICELEARN = False

skip_no_cl = unittest.skipUnless(HAS_CHOICELEARN, "choice-learn not installed")

from BayesianSparseDeepHalo.GenerateData import SimParams, generate_data_tf
from BayesianSparseDeepHalo.BLP import _rc_logit_draw_probs, blp_build_ivs, FastBLP
from BayesianSparseDeepHalo.DeepHalo import NonlinearMap, DeepHaloEncoder
from BayesianSparseDeepHalo.LuSparseRandomLogit import (
    BayesianSparseRandomLogit,
    adapt_step_size,
    gibbs_update_gamma_phi_tf,
    mh_update_xi_cl,
    mh_update_beta_cl,
    mh_update_r_cl,
    tmh_update_beta_cl,
    calibrate_stepsizes_cl,
)
from BayesianSparseDeepHalo.DeepHalo_MCEM_Core import (
    SparseDeepHaloMCEM,
    build_choice_dataset_from_market_counts,
    compute_probs_and_ll_batch_masked,
)


# ---------------------------------------------------------------------------
# Shared param dataclasses
# ---------------------------------------------------------------------------
@dataclass
class _McmcParams:
    G: int = 5
    burn: int = 2
    R0: int = 10
    step_beta: float = 0.1
    step_r: float = 0.1
    step_xibar: float = 0.1
    step_eta: float = 0.1
    V_beta: float = 10.0
    V_r: float = 10.0
    V_xibar: float = 10.0
    tau0: float = 0.1
    tau1: float = 1.0
    a_phi: float = 1.0
    b_phi: float = 1.0
    kappa_beta: float = 0.1
    random_coef_mask: object = None

    def __post_init__(self):
        if self.random_coef_mask is None:
            self.random_coef_mask = np.array([True, False])


@dataclass
class _CalibParams:
    calib_iters: int = 5
    burn_in: int = 2
    adapt_every: int = 2
    accept_target_high: float = 0.45
    accept_target_low: float = 0.15
    upscale_ratio: float = 1.2
    downscale_ratio: float = 0.8
    min_step: float = 1e-4
    max_step: float = 5.0


@dataclass
class _EmParams:
    outer_iters: int = 2
    mcmc_per_outer: int = 5
    burn: int = 2
    thin: int = 1
    nn_lr: float = 1e-3
    nn_l2: float = 1e-4
    nn_steps: int = 3
    recalibrate_each_outer: bool = False


def _make_tiny_model(T=4, J=3, D=2, beta_method="rwmh"):
    sp = SimParams(T=T, J=J, Nt=20, D=D, seed=0,
                   beta_mean=np.zeros(D), sigma_beta=np.zeros(D))
    mp = _McmcParams()
    model = BayesianSparseRandomLogit(
        sim_params=sp, mcmc_params=mp, seed=0,
        beta_method=beta_method, verbose=False
    )
    return model, sp, mp


def _make_tiny_dataset(T=4, J=3, D=2):
    sp = SimParams(T=T, J=J, Nt=20, D=D, seed=0,
                   beta_mean=np.zeros(D), sigma_beta=np.zeros(D))
    data_mcmc, ds, _ = generate_data_tf(1, sp)
    return ds, sp, data_mcmc


# ===========================================================================
# 1. GenerateData
# ===========================================================================
class TestSimParams(unittest.TestCase):

    def test_defaults(self):
        sp = SimParams()
        self.assertEqual(sp.T, 100)
        self.assertEqual(sp.J, 15)
        self.assertEqual(sp.Nt, 1000)
        self.assertEqual(sp.D, 2)
        self.assertIsNotNone(sp.beta_mean)
        self.assertIsNotNone(sp.sigma_beta)

    def test_beta_mean_length_matches_D(self):
        sp = SimParams(D=2)
        self.assertEqual(len(sp.beta_mean), sp.D)

    def test_sigma_beta_length_matches_D(self):
        sp = SimParams(D=2)
        self.assertEqual(len(sp.sigma_beta), sp.D)

    def test_custom_override(self):
        sp = SimParams(T=10, J=5, Nt=50, D=2)
        self.assertEqual(sp.T, 10)
        self.assertEqual(sp.J, 5)
        self.assertEqual(sp.Nt, 50)


class TestGenerateDataTf(unittest.TestCase):

    def setUp(self):
        self.sp = SimParams(T=5, J=4, Nt=50, D=2, seed=0)

    def _check_shapes(self, data_mcmc, true_params, T, J):
        self.assertEqual(data_mcmc["X"].shape, (T, J, 2))
        self.assertEqual(data_mcmc["q"].shape, (T, J + 1))
        self.assertEqual(data_mcmc["Nt"], 50)
        self.assertEqual(true_params["eta_star"].shape, (T, J))
        self.assertEqual(true_params["xi_star"].shape, (T, J))

    def test_dgp1_shapes(self):
        dm, _, tp = generate_data_tf(1, self.sp)
        self._check_shapes(dm, tp, 5, 4)

    def test_dgp2_shapes(self):
        dm, _, tp = generate_data_tf(2, self.sp)
        self._check_shapes(dm, tp, 5, 4)

    def test_dgp3_shapes(self):
        dm, _, tp = generate_data_tf(3, self.sp)
        self._check_shapes(dm, tp, 5, 4)

    def test_dgp4_shapes(self):
        dm, _, tp = generate_data_tf(4, self.sp)
        self._check_shapes(dm, tp, 5, 4)

    def test_counts_sum_to_Nt(self):
        dm, _, _ = generate_data_tf(1, self.sp)
        np.testing.assert_array_equal(dm["q"].sum(axis=1), np.full(5, 50))

    def test_invalid_dgptype_raises(self):
        with self.assertRaises(ValueError):
            generate_data_tf(99, self.sp)

    @skip_no_cl
    def test_dataset_choices_in_range(self):
        _, ds, _ = generate_data_tf(1, self.sp)
        choices = np.asarray(ds.choices)
        self.assertTrue(np.all(choices >= 0))
        self.assertTrue(np.all(choices <= self.sp.J))

    def test_reproducibility(self):
        sp1 = SimParams(T=5, J=4, Nt=50, D=2, seed=42)
        sp2 = SimParams(T=5, J=4, Nt=50, D=2, seed=42)
        d1, _, _ = generate_data_tf(1, sp1)
        d2, _, _ = generate_data_tf(1, sp2)
        np.testing.assert_array_equal(d1["X"], d2["X"])

    def test_xi_star_equals_xibar_plus_eta(self):
        sp = SimParams(T=5, J=4, Nt=50, D=2, seed=1, xi_bar=-1.0)
        _, _, tp = generate_data_tf(1, sp)
        np.testing.assert_allclose(tp["xi_star"], sp.xi_bar + tp["eta_star"], atol=1e-12)


# ===========================================================================
# 2. BLP
# ===========================================================================
class TestRcLogitDrawProbs(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.J = 5
        self.Ndraw = 30
        self.delta = rng.normal(size=self.J)
        self.price = rng.uniform(0.5, 2.0, size=self.J)
        self.v_draws = rng.normal(size=self.Ndraw)

    def test_output_shape(self):
        probs = _rc_logit_draw_probs(self.delta, self.price, -1.0, 0.8, self.v_draws)
        self.assertEqual(probs.shape, (self.Ndraw, self.J + 1))

    def test_probabilities_sum_to_one(self):
        probs = _rc_logit_draw_probs(self.delta, self.price, -1.0, 0.8, self.v_draws)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(self.Ndraw), atol=1e-10)

    def test_probabilities_non_negative(self):
        probs = _rc_logit_draw_probs(self.delta, self.price, -1.0, 0.8, self.v_draws)
        self.assertTrue(np.all(probs >= 0))

    def test_sigma_zero_collapses_to_mnl(self):
        probs = _rc_logit_draw_probs(self.delta, self.price, -1.0, 0.0, self.v_draws)
        np.testing.assert_allclose(probs.std(axis=0), 0.0, atol=1e-12)


class TestBlpBuildIvs(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.J = 6
        self.w_t = rng.uniform(0.5, 2.0, size=self.J)
        self.u_cost_t = rng.normal(size=self.J)

    def test_ivtype1_shape(self):
        Z = blp_build_ivs(self.w_t, self.u_cost_t, iv_type=1)
        self.assertEqual(Z.shape, (self.J, 5))

    def test_ivtype2_shape(self):
        Z = blp_build_ivs(self.w_t, self.u_cost_t, iv_type=2)
        self.assertEqual(Z.shape, (self.J, 5))

    def test_ivtype_invalid_raises(self):
        with self.assertRaises(ValueError):
            blp_build_ivs(self.w_t, self.u_cost_t, iv_type=3)

    def test_first_column_is_ones(self):
        Z = blp_build_ivs(self.w_t, self.u_cost_t, iv_type=1)
        np.testing.assert_array_equal(Z[:, 0], np.ones(self.J))

    def test_second_column_is_w(self):
        Z = blp_build_ivs(self.w_t, self.u_cost_t, iv_type=1)
        np.testing.assert_array_equal(Z[:, 1], self.w_t)


class TestFastBLP(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        T, J, Nt = 4, 5, 200
        X = rng.uniform(-1, 1, size=(T, J, 2))
        X[:, :, 0] = rng.uniform(0.5, 2.5, size=(T, J))
        shares = np.abs(rng.dirichlet(np.ones(J + 1), size=T))
        q = np.round(shares * Nt).astype(int)
        diff = Nt - q.sum(axis=1)
        for t in range(T): q[t, 0] += diff[t]
        self.data = {"X": X, "q": q, "Nt": Nt}
        self.u_cost = rng.normal(size=(T, J))
        self.T, self.J = T, J

    def test_init_stores_T_and_J(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        self.assertEqual(blp.T, self.T)
        self.assertEqual(blp.J, self.J)

    def test_compute_g_xi_types(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        g, xi = blp.compute_g_xi([-1.0, 0.5, 1.0])
        self.assertIsInstance(g, np.ndarray)
        self.assertIsInstance(xi, np.ndarray)

    def test_xi_shape(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        _, xi = blp.compute_g_xi([-1.0, 0.5, 1.0])
        self.assertEqual(xi.shape, (self.T, self.J))

    def test_stage1_obj_returns_float(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        self.assertIsInstance(blp.stage1_obj([-1.0, 0.5, 1.0]), float)

    def test_stage1_penalises_positive_price_coeff(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        self.assertEqual(blp.stage1_obj([1.0, 0.5, 1.0]), 1e12)

    def test_stage1_penalises_negative_sigma(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        self.assertEqual(blp.stage1_obj([-1.0, 0.5, -0.5]), 1e12)

    def test_delta_last_updated_after_compute_g_xi(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        blp.compute_g_xi([-1.0, 0.5, 1.0])
        self.assertIsNotNone(blp.delta_last)

    def test_xi_last_updated_after_compute_g_xi(self):
        blp = FastBLP(self.data, self.u_cost, iv_type=1, N_draw=20)
        blp.compute_g_xi([-1.0, 0.5, 1.0])
        self.assertIsNotNone(blp.xi_last)


# ===========================================================================
# 3. DeepHalo
# ===========================================================================
class TestNonlinearMap(unittest.TestCase):

    def setUp(self):
        self.H, self.embed = 4, 8
        self.layer = NonlinearMap(H=self.H, embed=self.embed)

    def test_output_shape(self):
        x = tf.random.normal((5, self.H * self.embed), dtype=tf.float64)
        self.assertEqual(self.layer(x).shape, (5, self.H * self.embed))

    def test_training_flag_accepted(self):
        x = tf.random.normal((3, self.H * self.embed), dtype=tf.float64)
        self.assertEqual(self.layer(x, training=True).shape,
                         self.layer(x, training=False).shape)

    def test_layer_norm_applied(self):
        x = tf.random.normal((64, self.H * self.embed), dtype=tf.float64)
        out = self.layer(x).numpy()
        self.assertLess(np.abs(out.mean(axis=-1)).mean(), 0.5)


class TestDeepHaloEncoder(unittest.TestCase):

    def _make(self, block_type="qua", depth=2, H=4, out_dim=3):
        return DeepHaloEncoder(H=H, depth=depth, embed=8, dropout=0.0,
                               block_type=block_type, out_dim=out_dim)

    def _inputs(self, T=6, M=4, H=4):
        return (tf.random.normal((T, M, H), dtype=tf.float64),
                tf.ones((T, M), dtype=tf.float64))

    def test_output_shape_qua(self):
        enc = self._make(block_type="qua", out_dim=3)
        self.assertEqual(enc(*self._inputs()).shape, (6, 4, 3))

    def test_output_shape_lin(self):
        enc = self._make(block_type="lin", out_dim=2)
        self.assertEqual(enc(*self._inputs()).shape, (6, 4, 2))

    def test_invalid_block_type_raises(self):
        enc = self._make(block_type="bad")
        with self.assertRaises(ValueError):
            enc(*self._inputs())

    def test_partial_availability(self):
        enc = self._make()
        X = tf.random.normal((4, 5, 4), dtype=tf.float64)
        avail = tf.constant([[1,1,0,1,1],[1,0,0,1,1],[1,1,1,1,0],[1,1,1,1,1]],
                             dtype=tf.float64)
        self.assertEqual(enc(X, avail).shape[0], 4)

    def test_training_vs_inference(self):
        enc = self._make()
        X, a = self._inputs()
        self.assertEqual(enc(X, a, training=True).shape,
                         enc(X, a, training=False).shape)

    def test_trainable_weights_exist_after_build(self):
        enc = self._make()
        enc(*self._inputs())
        self.assertGreater(len(enc.trainable_weights), 0)

    def test_dtype_float64(self):
        enc = self._make()
        self.assertEqual(enc(*self._inputs()).dtype, tf.float64)


# ===========================================================================
# 4. LuSparseRandomLogit — basic model tests
# ===========================================================================
class TestBayesianSparseRandomLogitInit(unittest.TestCase):

    def test_T_J_D_stored(self):
        model, sp, _ = _make_tiny_model()
        self.assertEqual(model.T, sp.T)
        self.assertEqual(model.J, sp.J)
        self.assertEqual(model.D, sp.D)

    def test_not_initialized_before_fit(self):
        self.assertFalse(_make_tiny_model()[0]._is_initialized)

    def test_compute_batch_utility_raises_before_fit(self):
        model, _, _ = _make_tiny_model()
        with self.assertRaises(RuntimeError):
            model.compute_batch_utility(
                tf.zeros((2, 1)), tf.zeros((2, 4, 2)), tf.ones((2, 4)))

    def test_trainable_weights_empty_before_fit(self):
        self.assertEqual(len(_make_tiny_model()[0].trainable_weights), 0)

    def test_samples_attr_absent_before_fit(self):
        self.assertFalse(hasattr(_make_tiny_model()[0], "samples_"))


class TestBayesianSparseRandomLogitFit(unittest.TestCase):

    @classmethod
    @skip_no_cl
    def setUpClass(cls):
        cls.ds, cls.sp, _ = _make_tiny_dataset(T=4, J=3, D=2)

    @skip_no_cl
    def test_fit_returns_train_loss_key(self):
        model, _, _ = _make_tiny_model()
        self.assertIn("train_loss", model.fit(self.ds, store_samples=False))

    @skip_no_cl
    def test_fit_loss_list_length(self):
        model, _, mp = _make_tiny_model()
        self.assertEqual(len(model.fit(self.ds, store_samples=False)["train_loss"]), mp.G + 1)

    @skip_no_cl
    def test_is_initialized_after_fit(self):
        model, _, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=False)
        self.assertTrue(model._is_initialized)

    @skip_no_cl
    def test_samples_stored_with_correct_keys(self):
        model, _, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=True)
        for key in ("beta_bar", "r_vec", "xi_bar", "eta", "gamma", "phi"):
            self.assertIn(key, model.samples_)

    @skip_no_cl
    def test_samples_none_when_not_requested(self):
        model, _, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=False)
        self.assertIsNone(model.samples_)

    @skip_no_cl
    def test_compute_batch_utility_shape_after_fit(self):
        T, J, D = 4, 3, 2
        model, _, _ = _make_tiny_model(T=T, J=J, D=D)
        model.fit(self.ds, store_samples=False)
        util = model.compute_batch_utility(
            tf.constant([[t] for t in range(T)], dtype=tf.float32),
            tf.random.normal((T, J+1, D), dtype=tf.float64),
            tf.ones((T, J+1), dtype=tf.float64))
        self.assertEqual(util.shape, (T, J+1))

    @skip_no_cl
    def test_beta_bar_shape_after_fit(self):
        model, sp, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=False)
        self.assertEqual(model.beta_bar.shape, (sp.D,))

    @skip_no_cl
    def test_xi_bar_shape_after_fit(self):
        model, _, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=False)
        self.assertEqual(model.xi_bar.shape[0], model._T_unique)

    @skip_no_cl
    def test_eta_shape_after_fit(self):
        model, sp, _ = _make_tiny_model()
        model.fit(self.ds, store_samples=False)
        self.assertEqual(model.eta.shape, (model._T_unique, sp.J))


class TestComputeLogLikelihoodUnique(unittest.TestCase):

    @classmethod
    @skip_no_cl
    def setUpClass(cls):
        ds, _, _ = _make_tiny_dataset()
        cls.model, _, _ = _make_tiny_model()
        cls.model.fit(ds, store_samples=False)

    @skip_no_cl
    def test_llvec_shape(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll_vec = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        self.assertEqual(ll_vec.shape[0], self.model._T_unique)

    @skip_no_cl
    def test_llvec_finite(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll_vec = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        self.assertTrue(np.all(np.isfinite(ll_vec.numpy())))

    @skip_no_cl
    def test_sigma_sums_to_one(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        sigma, _, _ = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        np.testing.assert_allclose(
            tf.reduce_sum(sigma, axis=1).numpy(),
            np.ones(self.model._T_unique), atol=1e-5)


class TestAdaptStepSize(unittest.TestCase):

    def _c(self): return _CalibParams()

    def test_increases_on_high_acceptance(self):
        self.assertGreater(adapt_step_size(0.1, 90, 100, self._c()), 0.1)

    def test_decreases_on_low_acceptance(self):
        self.assertLess(adapt_step_size(0.1, 5, 100, self._c()), 0.1)

    def test_clipped_at_max_step(self):
        self.assertLessEqual(adapt_step_size(4.9, 100, 100, self._c()), self._c().max_step)

    def test_clipped_at_min_step(self):
        self.assertGreaterEqual(adapt_step_size(1e-4, 0, 100, self._c()), self._c().min_step)

    def test_no_change_in_target_range(self):
        self.assertAlmostEqual(adapt_step_size(0.5, 30, 100, self._c()), 0.5)


class TestGibbsUpdateGammaPhi(unittest.TestCase):

    def setUp(self):
        T, J = 8, 5
        self.eta = tf.constant(np.random.default_rng(0).normal(size=(T, J)), dtype=tf.float64)
        self.phi = tf.constant(np.full((T,), 0.5), dtype=tf.float64)
        self.mp  = _McmcParams()

    def test_gamma_shape(self):
        gamma, _ = gibbs_update_gamma_phi_tf(self.eta, self.phi, self.mp)
        self.assertEqual(gamma.shape, self.eta.shape)

    def test_gamma_binary(self):
        gamma, _ = gibbs_update_gamma_phi_tf(self.eta, self.phi, self.mp)
        self.assertTrue(set(np.unique(gamma.numpy())).issubset({0, 1}))

    def test_phi_shape(self):
        _, phi_new = gibbs_update_gamma_phi_tf(self.eta, self.phi, self.mp)
        self.assertEqual(phi_new.shape, self.phi.shape)

    def test_phi_in_unit_interval(self):
        _, phi_new = gibbs_update_gamma_phi_tf(self.eta, self.phi, self.mp)
        p = phi_new.numpy()
        self.assertTrue(np.all(p > 0) and np.all(p < 1))


# ===========================================================================
# 5. beta_method / step-size parameter consistency
# ===========================================================================
class TestBetaMethodRouting(unittest.TestCase):

    @classmethod
    @skip_no_cl
    def setUpClass(cls):
        cls.ds, cls.sp, _ = _make_tiny_dataset(T=4, J=3, D=2)

    def _fitted_state(self, beta_method="rwmh"):
        """Return a fitted model plus the inputs needed for a single updater call."""
        sp = SimParams(T=4, J=3, Nt=20, D=2, seed=0,
                       beta_mean=np.zeros(2), sigma_beta=np.zeros(2))
        mp = _McmcParams()
        model = BayesianSparseRandomLogit(sim_params=sp, mcmc_params=mp,
                                          seed=0, beta_method=beta_method,
                                          verbose=False)
        model.fit(self.ds, store_samples=False)
        xi = model.xi_bar[:, None] + model.eta
        _, _, ll_vec = model.compute_log_likelihood_unique(
            xi, model.beta_bar, model.r_vec, model.v_draws)
        update_mask = tf.ones((model.D,), dtype=tf.float64)
        return model, ll_vec, update_mask

    # --- A. beta_method stored on model ---

    def test_rwmh_stored_on_model(self):
        model, _, _ = _make_tiny_model(beta_method="rwmh")
        self.assertEqual(model.beta_method, "rwmh")

    def test_tmh_stored_on_model(self):
        model, _, _ = _make_tiny_model(beta_method="tmh")
        self.assertEqual(model.beta_method, "tmh")

    # --- B. RWMH: step_beta controls proposal noise ---

    @skip_no_cl
    def test_rwmh_zero_step_beta_leaves_beta_unchanged(self):
        """step_beta=0 → noise = 0*mask = 0 → proposal always equals current."""
        model, ll_vec, mask = self._fitted_state("rwmh")
        model.mcmc_params.step_beta = 0.0
        beta_before = model.beta_bar.numpy().copy()
        for _ in range(10):
            model.beta_bar, ll_vec, _ = mh_update_beta_cl(
                model, model.beta_bar, ll_vec,
                model.xi_bar, model.eta, model.r_vec,
                model.v_draws, model.mcmc_params, mask)
        np.testing.assert_array_equal(
            model.beta_bar.numpy(), beta_before,
            err_msg="step_beta=0 must leave beta completely unchanged")

    @skip_no_cl
    def test_rwmh_nonzero_step_beta_can_move_beta(self):
        """step_beta=1.0 → proposal has variance → accepted moves are possible."""
        model, ll_vec, mask = self._fitted_state("rwmh")
        model.mcmc_params.step_beta = 1.0
        beta_before = model.beta_bar.numpy().copy()
        tf.random.set_seed(7)
        moved = False
        for _ in range(50):
            beta_new, ll_vec, _ = mh_update_beta_cl(
                model, model.beta_bar, ll_vec,
                model.xi_bar, model.eta, model.r_vec,
                model.v_draws, model.mcmc_params, mask)
            if not np.allclose(beta_new.numpy(), beta_before):
                moved = True
                break
            model.beta_bar = beta_new
        self.assertTrue(moved,
            "step_beta=1.0 should produce at least one accepted move in 50 rwmh steps")

    @skip_no_cl
    def test_rwmh_immune_to_kappa_beta(self):
        """Same RNG seed + different kappa_beta → identical mh_update_beta_cl output."""
        model, ll_vec, mask = self._fitted_state("rwmh")
        tf.random.set_seed(42)
        model.mcmc_params.kappa_beta = 0.001
        b1, _, a1 = mh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)

        tf.random.set_seed(42)
        model.mcmc_params.kappa_beta = 999.0
        b2, _, a2 = mh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)

        np.testing.assert_array_equal(b1.numpy(), b2.numpy(),
            err_msg="mh_update_beta_cl must not depend on kappa_beta")
        self.assertEqual(a1, a2)

    # --- C. TMH: kappa_beta controls proposal covariance scale ---

    @skip_no_cl
    def test_tmh_zero_kappa_beta_leaves_beta_near_mode(self):
        """kappa_beta=0 → proposal covariance = 0 → proposal = mode ≈ current."""
        model, ll_vec, mask = self._fitted_state("tmh")
        model.mcmc_params.kappa_beta = 0.0
        beta_before = model.beta_bar.numpy().copy()
        for _ in range(10):
            model.beta_bar, ll_vec, _ = tmh_update_beta_cl(
                model, model.beta_bar, ll_vec,
                model.xi_bar, model.eta, model.r_vec,
                model.v_draws, model.mcmc_params, mask)
        np.testing.assert_allclose(
            model.beta_bar.numpy(), beta_before, atol=1e-4,
            err_msg="kappa_beta=0 should keep beta near its starting value")

    @skip_no_cl
    def test_tmh_nonzero_kappa_beta_can_move_beta(self):
        """kappa_beta=2.0 → proposal has variance → accepted moves are possible."""
        model, ll_vec, mask = self._fitted_state("tmh")
        model.mcmc_params.kappa_beta = 2.0
        beta_before = model.beta_bar.numpy().copy()
        tf.random.set_seed(7)
        moved = False
        for _ in range(50):
            beta_new, ll_vec, _ = tmh_update_beta_cl(
                model, model.beta_bar, ll_vec,
                model.xi_bar, model.eta, model.r_vec,
                model.v_draws, model.mcmc_params, mask)
            if not np.allclose(beta_new.numpy(), beta_before):
                moved = True
                break
            model.beta_bar = beta_new
        self.assertTrue(moved,
            "kappa_beta=2.0 should produce at least one accepted move in 50 tmh steps")

    @skip_no_cl
    def test_tmh_immune_to_step_beta(self):
        """Same RNG seed + different step_beta → identical tmh_update_beta_cl output."""
        model, ll_vec, mask = self._fitted_state("tmh")
        tf.random.set_seed(42)
        model.mcmc_params.step_beta = 1e-6
        b1, _, a1 = tmh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)

        tf.random.set_seed(42)
        model.mcmc_params.step_beta = 999.0
        b2, _, a2 = tmh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)

        np.testing.assert_array_equal(b1.numpy(), b2.numpy(),
            err_msg="tmh_update_beta_cl must not depend on step_beta")
        self.assertEqual(a1, a2)

    # --- D. calibrate_stepsizes_cl adapts the correct parameter only ---

    @skip_no_cl
    def test_calibration_rwmh_does_not_touch_kappa_beta(self):
        sp = SimParams(T=4, J=3, Nt=20, D=2, seed=0,
                       beta_mean=np.zeros(2), sigma_beta=np.zeros(2))
        mp = _McmcParams()
        original_kappa = mp.kappa_beta
        calibrate_stepsizes_cl(self.ds, sp, mp, _CalibParams(),
                                seed=0, beta_method="rwmh", verbose=False)
        self.assertEqual(mp.kappa_beta, original_kappa,
                         "rwmh calibration must not modify kappa_beta")

    @skip_no_cl
    def test_calibration_tmh_does_not_touch_step_beta(self):
        sp = SimParams(T=4, J=3, Nt=20, D=2, seed=0,
                       beta_mean=np.zeros(2), sigma_beta=np.zeros(2))
        mp = _McmcParams()
        original_step = mp.step_beta
        calibrate_stepsizes_cl(self.ds, sp, mp, _CalibParams(),
                                seed=0, beta_method="tmh", verbose=False)
        self.assertEqual(mp.step_beta, original_step,
                         "tmh calibration must not modify step_beta")

    # --- E. Return-value contracts ---

    @skip_no_cl
    def test_mh_update_beta_cl_returns_triple(self):
        model, ll_vec, mask = self._fitted_state("rwmh")
        result = mh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)
        self.assertEqual(len(result), 3)
        beta_new, _, acc = result
        self.assertEqual(beta_new.shape, model.beta_bar.shape)
        self.assertIn(int(acc), (0, 1))

    @skip_no_cl
    def test_tmh_update_beta_cl_returns_triple(self):
        model, ll_vec, mask = self._fitted_state("tmh")
        result = tmh_update_beta_cl(
            model, model.beta_bar, ll_vec,
            model.xi_bar, model.eta, model.r_vec,
            model.v_draws, model.mcmc_params, mask)
        self.assertEqual(len(result), 3)
        beta_new, _, acc = result
        self.assertEqual(beta_new.shape, model.beta_bar.shape)
        self.assertIn(int(acc), (0, 1))


# ===========================================================================
# 6. DeepHalo_MCEM_Core
# ===========================================================================
class TestSparseDeepHaloMCEMInit(unittest.TestCase):

    def _mcem(self, T=4, M=3, H=4, out_dim=2):
        rng = np.random.default_rng(0)
        X_np = rng.uniform(-1, 1, size=(T, M, H))
        q_np = np.ones((T, M), dtype=np.int32) * 5
        avail_np = np.ones((T, M), dtype=np.float64)
        enc = DeepHaloEncoder(H=H, depth=2, embed=8, dropout=0.0,
                              block_type="qua", out_dim=out_dim)
        enc(tf.random.normal((T, M, H), dtype=tf.float64),
            tf.ones((T, M), dtype=tf.float64))
        sp = SimParams(T=T, J=M, Nt=20, D=out_dim, seed=0,
                       beta_mean=np.zeros(out_dim), sigma_beta=np.zeros(out_dim))
        mp = _McmcParams(R0=8, G=5, burn=2,
                         random_coef_mask=np.array([True]+[False]*(out_dim-1)))
        return SparseDeepHaloMCEM(
            X_np=X_np, q_np=q_np, avail_np=avail_np, encoder=enc,
            sim_params=sp, mcmc_params=mp, calib_params=_CalibParams(),
            em_params=_EmParams(), seed=0, beta_method="rwmh", verbose=False)

    def test_T_M_stored(self):
        m = self._mcem()
        self.assertEqual(m.T, 4); self.assertEqual(m.M, 3)

    def test_Dz_correct(self):
        self.assertEqual(self._mcem(out_dim=3).Dz, 3)

    def test_posterior_means_none_initially(self):
        self.assertIsNone(self._mcem().posterior_means)

    def test_X_is_tf_tensor(self):
        self.assertIsInstance(self._mcem().X, tf.Tensor)

    def test_avail_is_tf_tensor(self):
        self.assertIsInstance(self._mcem().avail, tf.Tensor)


class TestBuildChoiceDatasetFromMarketCounts(unittest.TestCase):

    def _make(self, T=3, M=4, Dz=2, count=3, seed=0):
        Z_np     = np.random.default_rng(seed).normal(size=(T, M, Dz))
        q_np     = np.ones((T, M + 1), dtype=np.int32) * count
        avail_np = np.ones((T, M), dtype=np.float64)
        return Z_np, q_np, avail_np

    @skip_no_cl
    def test_returns_dataset(self):
        self.assertIsNotNone(
            build_choice_dataset_from_market_counts(*self._make(seed=0)))

    @skip_no_cl
    def test_choices_in_valid_range(self):
        T, M = 3, 4
        ds = build_choice_dataset_from_market_counts(*self._make(T=T, M=M, seed=1))
        choices = np.asarray(ds.choices)
        self.assertTrue(np.all(choices >= 0) and np.all(choices <= M))

    @skip_no_cl
    def test_custom_market_ids_permutation(self):
        T, M = 3, 4
        Z_np, q_np, avail_np = self._make(T=T, M=M, seed=2)
        ds = build_choice_dataset_from_market_counts(
            Z_np, q_np, avail_np,
            market_ids=np.array([2, 0, 1], dtype=np.int32))
        self.assertIsNotNone(ds)

    @skip_no_cl
    def test_total_choices_equals_sum_of_q(self):
        T, M = 3, 4
        Z_np, q_np, avail_np = self._make(T=T, M=M, count=5, seed=3)
        ds = build_choice_dataset_from_market_counts(Z_np, q_np, avail_np)
        self.assertEqual(len(ds.choices), int(q_np.sum()))


class TestComputeProbsAndLLBatchMasked(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        T, M, Dz, R = 5, 4, 3, 20
        self.Z       = tf.constant(rng.normal(size=(T, M, Dz)),  dtype=tf.float64)
        self.q       = tf.constant(np.ones((T, M+1), dtype=np.int32) * 3)
        self.avail   = tf.ones((T, M), dtype=tf.float64)
        self.xi      = tf.constant(rng.normal(size=(T, M)),       dtype=tf.float64)
        self.beta    = tf.constant(rng.normal(size=(Dz,)),        dtype=tf.float64)
        self.r_vec   = tf.constant(rng.normal(size=(Dz,)) * 0.1, dtype=tf.float64)
        self.v_draws = tf.constant(rng.normal(size=(R, Dz)),      dtype=tf.float64)
        self.T, self.M = T, M

    def _call(self):
        return compute_probs_and_ll_batch_masked(
            self.Z, self.q, self.avail, self.xi,
            self.beta, self.r_vec, self.v_draws)

    def test_sigma_shape(self):
        sigma, _, _ = self._call()
        self.assertEqual(sigma.shape, (self.T, self.M + 1))

    def test_sigma_sums_to_one(self):
        sigma, _, _ = self._call()
        np.testing.assert_allclose(
            tf.reduce_sum(sigma, axis=1).numpy(), np.ones(self.T), atol=1e-5)

    def test_llvec_shape(self):
        _, _, ll_vec = self._call()
        self.assertEqual(ll_vec.shape[0], self.T)

    def test_llvec_finite(self):
        _, _, ll_vec = self._call()
        self.assertTrue(np.all(np.isfinite(ll_vec.numpy())))

    def test_probabilities_non_negative(self):
        sigma, _, _ = self._call()
        self.assertTrue(np.all(sigma.numpy() >= 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

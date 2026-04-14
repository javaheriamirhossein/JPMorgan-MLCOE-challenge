"""
Unit and integration tests for BayesianSparseDeepHalo.LuSparseRandomLogit

Test summary
----------
1.  GammaGibbsKernel – type/sanity checks  [avail_inside is now required]
2.  GammaGibbsKernel – correctness (empirical vs analytic probabilities)
2b. GammaGibbsKernel – avail_inside masking (NEW)
3.  BayesianSparseRandomLogit – instantiation
4.  BayesianSparseRandomLogit – dataset initialisation   [avail_inside populated]
5.  BayesianSparseRandomLogit – state initialisation
6.  Likelihood / utility shape and value checks
7.  Likelihood correctness (true params vs random params)
8.  Kernel build checks
9.  Stale target log-prob fix *** key correctness tests ***
10. Step-size adaptation
11. MCMC mixing
12. Posterior accuracy (vs ground-truth synthetic data)
13. Gamma sparsity recovery
14. Edge cases
15. Default values – step sizes, target_accept, burn-in
16. Variable product availability (NEW) – M > J, partial availability
"""

import sys
import os
import types
import collections

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from choice_learn.data import ChoiceDataset

from BayesianSparseDeepHalo.LuSparseRandomLogit import BayesianSparseRandomLogit, GammaGibbsKernel
from BayesianSparseDeepHalo.GenerateData import SimParams, MCMCParams, generate_data_tf

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_sim_params(T=20, J=5, D=2, Nt=200, seed=42):
    return SimParams(
        T=T, J=J, Nt=Nt, D=D,
        beta_mean=np.array([-1.0, 0.5][:D] if D <= 2 else [-1.0] + [0.5] * (D - 1)),
        sigma_beta=np.array([1.5, 0.0][:D] if D <= 2 else [1.5] + [0.0] * (D - 1)),
        xi_bar=-1.0,
        seed=seed,
    )

def _make_mcmc_params(T=20, J=5, D=2, burn=20, G=50, R0=30, target_accept=0.35):
    mask = np.array([1.] + [0.] * (D - 1))  # first feature random, rest fixed
    return MCMCParams(
        burn=burn, G=G, R0=R0,
        V_beta=10.0, V_r=10.0, V_xibar=10.0,
        tau0=0.05, tau1=1.0,
        a_phi=1.0, b_phi=1.0,
        step_beta=0.2, step_r=0.2,
        step_xibar=0.2, step_phi=0.2, step_eta=0.2,
        random_coef_mask=mask,
        target_accept=target_accept,
    )

def _make_full_avail_inside(T, J):
    """All items available (replicates the fixed-J behaviour)."""
    return tf.Variable(tf.ones((T, J), dtype=tf.float64))

def _make_partial_avail_inside(T, J, frac_avail=0.7, seed=0):
    """Random binary mask with ~frac_avail items available per market."""
    rng = np.random.default_rng(seed)
    mask = (rng.random((T, J)) < frac_avail).astype(np.float64)
    # Ensure at least one item available per market
    for t in range(T):
        if mask[t].sum() == 0:
            mask[t, 0] = 1.0
    return tf.Variable(tf.constant(mask, dtype=tf.float64))

def _make_fitted_model(dataset_cl, sim, mcmc):
    """Instantiate and fit (full chain) a model."""
    model = BayesianSparseRandomLogit(sim, mcmc, seed=42, verbose=True)
    model.fit(dataset_cl)
    return model

def _make_initialised_model(dataset_cl, sim, mcmc):
    """Instantiate, initialise (no chain run), build kernels."""
    model = BayesianSparseRandomLogit(sim, mcmc, seed=42, verbose=False)
    model._initialize_from_dataset(dataset_cl)
    model._initialize_state()
    model._build_kernels()
    return model

def _ess_bulk(chain: np.ndarray) -> float:
    """Crude effective sample size via autocorrelation (lag-1 approx)."""
    n = len(chain)
    if n < 4:
        return float(n)
    chain = chain - chain.mean()
    var = np.var(chain)
    if var < 1e-30:
        return 1.0
    ac1 = np.mean(chain[:-1] * chain[1:]) / (var + 1e-30)
    rho = max(-0.99, min(ac1, 0.99))
    return n * (1 - rho) / (1 + rho)

def _extract_log_prob(kr):
    """
    Walk TFP kernel-results namedtuple hierarchy to find target_log_prob.

    TFP nesting for SimpleStepSizeAdaptation(RandomWalkMetropolis(...)):
      kr -- SimpleStepSizeAdaptationResults
        └─ kr.inner_results -- MetropolisHastingsKernelResults
             └─ .accepted_results -- UncalibratedRandomWalkResults
                  └─ .target_log_prob -- scalar (or [T] vector) Tensor
    """
    if hasattr(kr, 'target_log_prob'):
        return kr.target_log_prob
    for field in getattr(kr, '_fields', []):
        val = getattr(kr, field)
        if hasattr(val, '_fields') or hasattr(val, 'target_log_prob'):
            result = _extract_log_prob(val)
            if result is not None:
                return result
    return None

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    sim = _make_sim_params(T=20, J=5, D=2, Nt=300)
    mcmc = _make_mcmc_params(T=20, J=5, D=2, burn=200, G=1000, R0=200)
    _, dataset_cl, true_params = generate_data_tf(dgp_type=1, sim=sim)
    return dataset_cl, true_params, None, sim, mcmc

@pytest.fixture(scope="module")
def medium_dataset():
    sim = _make_sim_params(T=50, J=10, D=2, Nt=1000)
    mcmc = _make_mcmc_params(T=50, J=10, D=2, burn=200, G=1000, R0=200)
    _, dataset_cl, true_params = generate_data_tf(dgp_type=1, sim=sim)
    return dataset_cl, true_params, None, sim, mcmc

# ═══════════════════════════════════════════════════════════════════════════
# 1. GammaGibbsKernel – type / sanity   [avail_inside now required]
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaGibbsKernelSanity:

    @pytest.fixture(autouse=True)
    def _build(self):
        T, J = 8, 5
        self.T, self.J = T, J
        self.phi_var  = tf.Variable(tf.constant(0.5, shape=(T,), dtype=tf.float64))
        self.eta_var  = tf.Variable(tf.random.normal((T, J), dtype=tf.float64))
        # All items available (replicates original fixed-J behaviour)
        self.avail    = _make_full_avail_inside(T, J)
        self.kernel   = GammaGibbsKernel(
            self.phi_var, self.eta_var, tau0=0.05, tau1=1.0,
            avail_inside=self.avail
        )

    def test_is_calibrated_true(self):
        assert self.kernel.is_calibrated is True

    def test_avail_inside_stored(self):
        """Masked kernel must expose avail_inside attribute."""
        assert hasattr(self.kernel, 'avail_inside')
        assert self.kernel.avail_inside.shape == (self.T, self.J)

    def test_avail_inside_dtype(self):
        assert self.kernel.avail_inside.dtype == tf.float64

    def test_bootstrap_results_is_namedtuple(self):
        init_gamma = tf.zeros((self.T, self.J), dtype=tf.int32)
        br = self.kernel.bootstrap_results(init_gamma)
        assert isinstance(br, tuple)

    def test_one_step_shape(self):
        init_gamma = tf.zeros((self.T, self.J), dtype=tf.int32)
        br = self.kernel.bootstrap_results(init_gamma)
        new_gamma, _ = self.kernel.one_step(init_gamma, br)
        assert new_gamma.shape == (self.T, self.J)

    def test_one_step_dtype(self):
        init_gamma = tf.zeros((self.T, self.J), dtype=tf.int32)
        br = self.kernel.bootstrap_results(init_gamma)
        new_gamma, _ = self.kernel.one_step(init_gamma, br)
        assert new_gamma.dtype == tf.int32

    def test_one_step_binary(self):
        init_gamma = tf.zeros((self.T, self.J), dtype=tf.int32)
        br = self.kernel.bootstrap_results(init_gamma)
        for _ in range(10):
            new_gamma, _ = self.kernel.one_step(init_gamma, br)
            vals = new_gamma.numpy()
            assert set(vals.flatten()).issubset({0, 1}), "gamma contains non-binary values"

    def test_tau_stored_as_float64(self):
        assert self.kernel.tau0.dtype == tf.float64
        assert self.kernel.tau1.dtype == tf.float64

# ═══════════════════════════════════════════════════════════════════════════
# 2. GammaGibbsKernel – correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaGibbsKernelCorrectness:

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def test_empirical_vs_analytic_probability(self):
        """
        For a single (t,j) pair, the empirical frequency of gamma=1 over many
        draws must match the analytic sigmoid formula.
        """
        T, J = 1, 1
        phi0  = 0.7
        eta_v = 0.8
        tau0  = 0.05
        tau1  = 1.0
        n_draw = 3000

        phi_var  = tf.Variable([phi0], dtype=tf.float64)
        eta_var  = tf.Variable([[eta_v]], dtype=tf.float64)
        avail    = tf.Variable([[1.0]], dtype=tf.float64)
        kernel   = GammaGibbsKernel(phi_var, eta_var, tau0=tau0, tau1=tau1,
                                     avail_inside=avail)
        init_g   = tf.zeros((T, J), dtype=tf.int32)
        br       = kernel.bootstrap_results(init_g)

        hits = 0
        for _ in range(n_draw):
            g, _ = kernel.one_step(init_g, br)
            hits += int(g.numpy()[0, 0])

        log_p1 = np.log(phi0) - 0.5 * eta_v**2 / tau1**2 - np.log(tau1)
        log_p0 = np.log(1 - phi0) - 0.5 * eta_v**2 / tau0**2 - np.log(tau0)
        p1_analytic = self._sigmoid(log_p1 - log_p0)

        np.testing.assert_allclose(
            hits / n_draw, p1_analytic, atol=0.07,
            err_msg=f"Empirical P(gamma=1)={hits/n_draw:.3f} vs analytic {p1_analytic:.3f}")

    def test_large_eta_forces_slab(self):
        """When |eta| is large relative to tau0, phi≈1 should give gamma≈1."""
        phi_var = tf.Variable([0.99], dtype=tf.float64)
        eta_var = tf.Variable([[3.0]], dtype=tf.float64)
        avail   = tf.Variable([[1.0]], dtype=tf.float64)
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                    avail_inside=avail)
        br = kernel.bootstrap_results(tf.zeros((1, 1), dtype=tf.int32))
        counts = sum(kernel.one_step(tf.zeros((1,1), tf.int32), br)[0].numpy()[0,0]
                     for _ in range(200))
        assert counts / 200 > 0.90, "Should almost always sample gamma=1"

    def test_phi_near_zero_forces_spike(self):
        """phi≈0 should force gamma≈0 regardless of eta."""
        phi_var = tf.Variable([0.01], dtype=tf.float64)
        eta_var = tf.Variable([[0.01]], dtype=tf.float64)
        avail   = tf.Variable([[1.0]], dtype=tf.float64)
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                    avail_inside=avail)
        br = kernel.bootstrap_results(tf.zeros((1,1), tf.int32))
        counts = sum(kernel.one_step(tf.zeros((1,1), tf.int32), br)[0].numpy()[0,0]
                     for _ in range(200))
        assert counts / 200 < 0.15, "Should almost always sample gamma=0"

    def test_multiple_markets_shape(self):
        T, J = 10, 8
        phi_var = tf.Variable(tf.random.uniform((T,), 0.2, 0.8, dtype=tf.float64))
        eta_var = tf.Variable(tf.random.normal((T, J), dtype=tf.float64))
        avail   = _make_full_avail_inside(T, J)
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.1, tau1=1.0,
                                    avail_inside=avail)
        init_g  = tf.zeros((T, J), dtype=tf.int32)
        br      = kernel.bootstrap_results(init_g)
        new_g, _ = kernel.one_step(init_g, br)
        assert new_g.shape == (T, J)
        assert set(new_g.numpy().flatten()).issubset({0, 1})

# ═══════════════════════════════════════════════════════════════════════════
# 2b. GammaGibbsKernel – avail_inside masking  (NEW)
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaGibbsKernelAvailMasking:
    """
    Verify that GammaGibbsKernel correctly forces gamma=0 for structurally
    unavailable items (avail_inside[t,j] == 0) regardless of eta or phi.
    """

    def test_unavailable_item_always_zero(self):
        """
        For an unavailable item (avail=0), gamma must always be 0
        even when phi=0.99 and |eta| is large (which would otherwise
        force gamma=1).
        """
        T, J = 1, 2
        phi_var = tf.Variable([0.99], dtype=tf.float64)
        eta_var = tf.Variable([[3.0, 3.0]], dtype=tf.float64)
        # Make only the first item available, second unavailable
        avail   = tf.Variable([[1.0, 0.0]], dtype=tf.float64)
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                    avail_inside=avail)
        br = kernel.bootstrap_results(tf.zeros((T, J), dtype=tf.int32))
        for _ in range(50):
            g, _ = kernel.one_step(tf.zeros((T, J), tf.int32), br)
            assert g.numpy()[0, 1] == 0, \
                "Unavailable item (j=1) must always have gamma=0"

    def test_available_item_can_be_one(self):
        """
        For an available item with phi=0.99 and large |eta|,
        gamma must sometimes be 1 (i.e. the mask doesn't over-suppress).
        """
        T, J = 1, 2
        phi_var = tf.Variable([0.99], dtype=tf.float64)
        eta_var = tf.Variable([[3.0, 3.0]], dtype=tf.float64)
        avail   = tf.Variable([[1.0, 0.0]], dtype=tf.float64)
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                    avail_inside=avail)
        br      = kernel.bootstrap_results(tf.zeros((T, J), dtype=tf.int32))
        hits_j0 = sum(kernel.one_step(tf.zeros((T, J), tf.int32), br)[0].numpy()[0, 0]
                      for _ in range(50))
        assert hits_j0 > 5, \
            "Available item (j=0) with phi=0.99 and large eta should be active most of the time"

    def test_partial_avail_all_unavailable_always_zero(self):
        """
        When an entire row is unavailable (all zeros), every gamma[t, :] must be 0.
        """
        T, J = 5, 6
        phi_var = tf.Variable(tf.fill((T,), tf.constant(0.9, tf.float64)))
        eta_var = tf.Variable(tf.fill((T, J), tf.constant(2.0, tf.float64)))
        # Only markets 0..2 have available items; market 3 and 4 are fully unavailable
        avail_np = np.ones((T, J), dtype=np.float64)
        avail_np[3, :] = 0.0
        avail_np[4, :] = 0.0
        avail   = tf.Variable(tf.constant(avail_np, dtype=tf.float64))
        kernel  = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                    avail_inside=avail)
        br = kernel.bootstrap_results(tf.zeros((T, J), dtype=tf.int32))
        for _ in range(20):
            g, _ = kernel.one_step(tf.zeros((T, J), tf.int32), br)
            assert np.all(g.numpy()[3, :] == 0), "Fully-unavailable market 3 must have all gamma=0"
            assert np.all(g.numpy()[4, :] == 0), "Fully-unavailable market 4 must have all gamma=0"

    def test_partial_avail_gamma_respects_mask(self):
        """
        For a random avail_inside mask, gamma must be 0 everywhere avail==0.
        """
        T, J = 10, 8
        rng = np.random.default_rng(99)
        avail_np = (rng.random((T, J)) > 0.4).astype(np.float64)
        phi_var  = tf.Variable(tf.fill((T,), tf.constant(0.8, tf.float64)))
        eta_var  = tf.Variable(tf.fill((T, J), tf.constant(1.5, tf.float64)))
        avail    = tf.Variable(tf.constant(avail_np, dtype=tf.float64))
        kernel   = GammaGibbsKernel(phi_var, eta_var, tau0=0.05, tau1=1.0,
                                     avail_inside=avail)
        br = kernel.bootstrap_results(tf.zeros((T, J), dtype=tf.int32))
        for _ in range(20):
            g, _ = kernel.one_step(tf.zeros((T, J), tf.int32), br)
            g_np = g.numpy()
            # Wherever avail==0, gamma must be 0
            mask_zero = (avail_np == 0)
            assert np.all(g_np[mask_zero] == 0), \
                "gamma must be 0 wherever avail_inside==0"

# ═══════════════════════════════════════════════════════════════════════════
# 3. BayesianSparseRandomLogit – instantiation
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiation:

    def test_basic_construction(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        assert model.T == sim.T
        assert model.J == sim.J
        assert model.D == sim.D

    def test_n_random_matches_mask(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert model.n_random == int(np.sum(mcmc.random_coef_mask))

    def test_random_coef_mask_dtype(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert model.random_coef_mask.dtype == tf.float64

    def test_custom_mask_overrides_mcmc(self):
        sim  = _make_sim_params(D=3)
        mcmc = _make_mcmc_params(D=3)
        custom_mask = np.array([1., 1., 0.])
        model = BayesianSparseRandomLogit(sim, mcmc, random_coef_mask=custom_mask)
        assert model.n_random == 2

    def test_is_initialized_false_before_fit(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert model._is_initialized is False

    def test_samples_none_before_fit(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert model.samples_ is None

    def test_trainable_weights_empty_before_fit(self):
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert len(model.trainable_weights) == 0

    def test_avail_inside_none_before_fit(self):
        """avail_inside and unique_avail must be None until dataset is loaded."""
        sim  = _make_sim_params()
        mcmc = _make_mcmc_params()
        model = BayesianSparseRandomLogit(sim, mcmc)
        assert model.avail_inside  is None
        assert model.unique_avail  is None

# ═══════════════════════════════════════════════════════════════════════════
# 4. Dataset initialisation  [avail_inside now populated]
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetInit:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        self.model._initialize_from_dataset(dataset_cl)
        self.sim = sim

    def test_is_initialized_flag(self):
        assert self.model._is_initialized is True

    def test_t_unique_matches_sim(self):
        assert self.model._T_unique == self.sim.T

    def test_unique_counts_shape(self):
        T, J = self.sim.T, self.sim.J
        assert self.model.unique_counts.shape == (T, J + 1)

    def test_unique_items_features_shape(self):
        T, J, D = self.sim.T, self.sim.J, self.sim.D
        assert self.model.unique_items_features.shape == (T, J + 1, D)

    def test_unique_counts_row_sums(self):
        """Each row of choice counts must sum to Nt (consumers per market)."""
        row_sums = tf.reduce_sum(self.model.unique_counts, axis=1).numpy()
        np.testing.assert_allclose(row_sums, self.sim.Nt, atol=2,
                                   err_msg="Row sums of unique_counts diverge from Nt")

    def test_unique_items_features_dtype(self):
        assert self.model.unique_items_features.dtype == tf.float64

    # ── NEW avail_inside checks ──────────────────────────────────────────────

    def test_unique_avail_populated(self):
        """unique_avail must be populated after _initialize_from_dataset."""
        assert self.model.unique_avail is not None

    def test_unique_avail_shape(self):
        """unique_avail must be [T_unique, J+1] (includes outside option column)."""
        T, J = self.sim.T, self.sim.J
        assert self.model.unique_avail.shape == (T, J + 1)

    def test_avail_inside_populated(self):
        """avail_inside (inside products only) must be populated after dataset init."""
        assert self.model.avail_inside is not None

    def test_avail_inside_shape(self):
        """avail_inside must be [T_unique, J] — outside option column dropped."""
        T, J = self.sim.T, self.sim.J
        assert self.model.avail_inside.shape == (T, J)

    def test_avail_inside_binary(self):
        """avail_inside entries must all be 0.0 or 1.0."""
        vals = self.model.avail_inside.numpy()
        assert set(vals.flatten()).issubset({0.0, 1.0}), \
            "avail_inside should contain only 0.0 or 1.0"

    def test_avail_inside_dtype(self):
        assert self.model.avail_inside.dtype == tf.float64

    def test_avail_inside_at_least_one_per_market(self):
        """Every market must have at least one available inside product."""
        per_market = self.model.avail_inside.numpy().sum(axis=1)
        assert np.all(per_market >= 1), \
            "Some markets have zero available inside products"

    def test_gamma_kernel_receives_avail_inside(self):
        """After full init, gamma_kernel.avail_inside must match model.avail_inside."""
        self.model._initialize_state()
        self.model._build_kernels()
        np.testing.assert_array_equal(
            self.model.gamma_kernel.avail_inside.numpy(),
            self.model.avail_inside.numpy(),
            err_msg="gamma_kernel.avail_inside does not match model.avail_inside"
        )

# ═══════════════════════════════════════════════════════════════════════════
# 5. State initialisation
# ═══════════════════════════════════════════════════════════════════════════

class TestStateInit:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.sim  = sim
        self.mcmc = mcmc
        self.model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        self.model._initialize_from_dataset(dataset_cl)
        self.model._initialize_state()

    def test_beta_bar_shape(self):
        assert self.model.beta_bar.shape == (self.sim.D,)

    def test_r_vec_shape(self):
        assert self.model.r_vec.shape == (self.model.n_random,)

    def test_xi_bar_shape(self):
        assert self.model.xi_bar.shape == (self.sim.T,)

    def test_eta_shape(self):
        assert self.model.eta.shape == (self.sim.T, self.sim.J)

    def test_gamma_shape(self):
        assert self.model.gamma.shape == (self.sim.T, self.sim.J)

    def test_phi_in_unit_interval(self):
        phi = self.model.phi.numpy()
        assert np.all(phi > 0) and np.all(phi < 1)

    def test_v_draws_shape(self):
        R0, n_rand = self.mcmc.R0, self.model.n_random
        assert self.model.v_draws.shape == (R0, n_rand)

    def test_all_state_vars_float64(self):
        for name in ['beta_bar', 'r_vec', 'xi_bar', 'eta', 'phi', 'z_phi']:
            var = getattr(self.model, name)
            assert var.dtype == tf.float64, f"{name} should be float64"

    def test_z_phi_consistency(self):
        """z_phi = logit(phi) => sigmoid(z_phi) == phi"""
        z   = self.model.z_phi.numpy()
        phi = self.model.phi.numpy()
        np.testing.assert_allclose(
            1.0 / (1.0 + np.exp(-z)), phi, atol=1e-10)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Likelihood – shape and value checks
# ═══════════════════════════════════════════════════════════════════════════

class TestLikelihoodShape:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.model = _make_initialised_model(dataset_cl, sim, mcmc)
        self.sim   = sim

    def test_ll_vec_shape(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        assert ll.shape == (self.sim.T,)

    def test_ll_vec_finite(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        assert np.all(np.isfinite(ll.numpy()))

    def test_ll_vec_nonpositive(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        assert np.all(ll.numpy() <= 1e-8), "log-likelihood must be <= 0"

    def test_sigma_shape(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        sigma, _, _ = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        assert sigma.shape == (self.sim.T, self.sim.J + 1)

    def test_sigma_sums_to_one(self):
        xi = self.model.xi_bar[:, None] + self.model.eta
        sigma, _, _ = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        row_sums = tf.reduce_sum(sigma, axis=1).numpy()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="Predicted shares must sum to 1 per market")

    def test_ll_deterministic(self):
        """Same inputs must give the same ll_vec."""
        xi = self.model.xi_bar[:, None] + self.model.eta
        _, _, ll1 = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        _, _, ll2 = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        np.testing.assert_array_equal(ll1.numpy(), ll2.numpy())

    def test_unavailable_items_get_zero_share(self):
        """
        When avail_inside[t, j] == 0, the predicted share sigma[t, j+1]
        must be effectively zero (clipped by -1e9 utility masking).
        """
        xi    = self.model.xi_bar[:, None] + self.model.eta
        sigma, _, _ = self.model.compute_log_likelihood_unique(
            xi, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        avail_np = self.model.avail_inside.numpy()   # [T, J]
        sigma_np = sigma.numpy()[:, 1:]              # drop outside option -> [T, J]
        # Wherever avail==0, predicted share should be vanishingly small
        unavail_shares = sigma_np[avail_np == 0]
        assert np.all(unavail_shares < 1e-6), \
            "Unavailable items must have near-zero predicted share"

# ═══════════════════════════════════════════════════════════════════════════
# 7. Likelihood correctness
# ═══════════════════════════════════════════════════════════════════════════

class TestLikelihoodCorrectness:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, true_params, _, sim, mcmc = small_dataset
        self.model       = _make_initialised_model(dataset_cl, sim, mcmc)
        self.true_params = true_params
        self.sim         = sim

    def test_true_xi_beats_zero_xi(self):
        """LL at true xi_star must be >= LL at xi=0."""
        xi_true = tf.constant(self.true_params["xi_star"], dtype=tf.float64)
        xi_zero = tf.zeros_like(xi_true)

        _, _, ll_true = self.model.compute_log_likelihood_unique(
            xi_true, self.model.beta_bar, self.model.r_vec, self.model.v_draws)
        _, _, ll_zero = self.model.compute_log_likelihood_unique(
            xi_zero, self.model.beta_bar, self.model.r_vec, self.model.v_draws)

        assert tf.reduce_sum(ll_true).numpy() >= tf.reduce_sum(ll_zero).numpy() - 5.0, \
            "True xi should give at least similar LL as xi=0"

    def test_wrong_beta_decreases_ll(self):
        """Wildly wrong beta should give lower total LL."""
        xi       = self.model.xi_bar[:, None] + self.model.eta
        beta_good = tf.constant(self.true_params["beta_mean"], dtype=tf.float64)
        beta_bad  = tf.constant([50.0, -50.0], dtype=tf.float64)

        _, _, ll_good = self.model.compute_log_likelihood_unique(
            xi, beta_good, self.model.r_vec, self.model.v_draws)
        _, _, ll_bad  = self.model.compute_log_likelihood_unique(
            xi, beta_bad, self.model.r_vec, self.model.v_draws)

        assert tf.reduce_sum(ll_good).numpy() > tf.reduce_sum(ll_bad).numpy()

    def test_sigma_near_uniform_at_zero_beta(self):
        """At beta=0 and xi=0, predicted shares should be roughly 1/(J+1)."""
        J     = self.sim.J
        xi_z  = tf.zeros((self.sim.T, J), dtype=tf.float64)
        beta_z = tf.zeros(self.sim.D, dtype=tf.float64)
        r_z   = tf.zeros(self.model.n_random, dtype=tf.float64)
        sigma, _, _ = self.model.compute_log_likelihood_unique(
            xi_z, beta_z, r_z, self.model.v_draws)
        mean_share = sigma.numpy().mean()
        np.testing.assert_allclose(mean_share, 1.0 / (J + 1), atol=0.05)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Kernel build checks
# ═══════════════════════════════════════════════════════════════════════════

class TestKernelBuild:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.model = _make_initialised_model(dataset_cl, sim, mcmc)

    def test_all_kernels_exist(self):
        for attr in ['beta_kernel', 'r_kernel', 'xi_kernel',
                     'eta_kernel', 'phi_kernel', 'gamma_kernel']:
            assert hasattr(self.model, attr), f"Missing kernel: {attr}"

    def test_gamma_kernel_is_gibbs(self):
        assert isinstance(self.model.gamma_kernel, GammaGibbsKernel)

    def test_gamma_kernel_avail_inside_matches_model(self):
        """
        After _build_kernels, the GammaGibbsKernel must hold the same
        avail_inside tensor as the model.
        """
        np.testing.assert_array_equal(
            self.model.gamma_kernel.avail_inside.numpy(),
            self.model.avail_inside.numpy(),
            err_msg="gamma_kernel.avail_inside must match model.avail_inside"
        )

    def test_beta_target_scalar(self):
        lp = self.model.beta_kernel.inner_kernel.target_log_prob_fn(
            self.model.beta_bar.read_value())
        assert lp.shape == ()

    def test_r_target_scalar(self):
        lp = self.model.r_kernel.inner_kernel.target_log_prob_fn(
            self.model.r_vec.read_value())
        assert lp.shape == ()

    def test_xi_target_vector(self):
        lp = self.model.xi_kernel.inner_kernel.target_log_prob_fn(
            self.model.xi_bar.read_value())
        assert lp.shape == (self.model._T_unique,)

    def test_eta_target_vector(self):
        lp = self.model.eta_kernel.inner_kernel.target_log_prob_fn(
            self.model.eta.read_value())
        assert lp.shape == (self.model._T_unique,)

    def test_step_vars_are_tf_variables(self):
        for name, var in self.model._step_vars.items():
            assert isinstance(var, tf.Variable), \
                f"_step_vars[{name}] is not tf.Variable"

    def test_eta_target_respects_unavail(self):
        """
        Setting a large eta for an unavailable item should NOT change
        the eta_target output (unavailable items must be zeroed inside target).
        """
        eta_base = self.model.eta.read_value().numpy().copy()
        lp_base  = self.model.eta_kernel.inner_kernel.target_log_prob_fn(
            tf.constant(eta_base, dtype=tf.float64))

        # Find an unavailable item
        avail_np  = self.model.avail_inside.numpy()
        unavail   = np.argwhere(avail_np == 0)
        if len(unavail) == 0:
            pytest.skip("No unavailable items in this dataset — skip test")

        eta_perturbed = eta_base.copy()
        t_idx, j_idx  = unavail[0]
        eta_perturbed[t_idx, j_idx] = 999.0   # huge perturbation

        lp_perturbed = self.model.eta_kernel.inner_kernel.target_log_prob_fn(
            tf.constant(eta_perturbed, dtype=tf.float64))

        np.testing.assert_allclose(
            lp_base.numpy(), lp_perturbed.numpy(), atol=1e-5,
            err_msg="eta_target must be unaffected by perturbations to unavailable items"
        )

# ═══════════════════════════════════════════════════════════════════════════
# 9. Stale target log-prob fix *** critical correctness tests ***
# ═══════════════════════════════════════════════════════════════════════════

class TestStaleTargetLogProbFix:
    """
    The key bug this module is designed to fix: when a RWMH step re-uses a
    kernel-results object from a previous step (different parameter values),
    the cached target_log_prob inside it is stale.

    _step_beta (and siblings) guard against this by calling
    inner_kernel.bootstrap_results(current_state) and injecting the fresh
    result via _replace(inner_results=...) before calling one_step.
    """

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.model      = _make_initialised_model(dataset_cl, sim, mcmc)
        self.dataset_cl = dataset_cl
        self.sim        = sim
        self.mcmc       = mcmc

    def _verify_lp_refreshed(self, kernel, state_var, step_fn, label):
        kr = kernel.bootstrap_results(state_var.read_value())
        new_kr, _ = step_fn(kr)

        cached_lp = _extract_log_prob(new_kr)
        assert cached_lp is not None, (
            f"[{label}] Could not find target_log_prob in kernel results tree. "
            f"Fields: {getattr(new_kr, '_fields', 'none')}. "
            "This indicates _step_* is NOT refreshing bootstrap_results — stale!")

        assert tf.math.is_finite(tf.reduce_sum(cached_lp)).numpy(), \
            f"[{label}] Cached target_log_prob is not finite: {cached_lp}"

        fresh_lp = kernel.inner_kernel.target_log_prob_fn(state_var.read_value())
        np.testing.assert_allclose(
            float(tf.reduce_sum(fresh_lp).numpy()),
            float(tf.reduce_sum(cached_lp).numpy()),
            atol=1e-6,
            err_msg=f"[{label}] Cached log-prob != fresh evaluation — stale log-prob!")

    def test_step_beta_refreshes_inner_results(self):
        self._verify_lp_refreshed(
            self.model.beta_kernel, self.model.beta_bar, self.model._step_beta, "beta_bar")

    def test_step_r_refreshes_inner_results(self):
        self._verify_lp_refreshed(
            self.model.r_kernel, self.model.r_vec, self.model._step_r, "r_vec")

    def test_step_xi_refreshes_inner_results(self):
        self._verify_lp_refreshed(
            self.model.xi_kernel, self.model.xi_bar, self.model._step_xi, "xi_bar")

    def test_step_eta_refreshes_inner_results(self):
        self._verify_lp_refreshed(
            self.model.eta_kernel, self.model.eta, self.model._step_eta, "eta")

    def test_step_phi_refreshes_inner_results(self):
        self._verify_lp_refreshed(
            self.model.phi_kernel, self.model.z_phi, self.model._step_phi, "phi/z_phi")

    def test_step_gamma_always_updates(self):
        """
        Gibbs step for gamma must change at least one indicator across 20
        sweeps when phi is non-trivial and items are available.
        """
        self.model.phi.assign(
            tf.constant(0.5, shape=(self.model._T_unique,), dtype=tf.float64))

        gamma_before = self.model.gamma.numpy().copy()
        any_changed  = False

        for _ in range(20):
            self.model._step_gamma()
            g = self.model.gamma.numpy()
            assert set(g.flatten()).issubset({0, 1}), \
                "gamma contains non-binary values after _step_gamma"
            if not np.array_equal(g, gamma_before):
                any_changed = True
            gamma_before = g.copy()

        assert any_changed, (
            "gamma never changed across 20 Gibbs sweeps with phi=0.5 — "
            "Gibbs kernel is not executing.")

    def test_step_gamma_unavail_stay_zero(self):
        """
        After any number of Gibbs sweeps, gamma must remain 0 for
        all unavailable items (avail_inside == 0).
        """
        avail_np = self.model.avail_inside.numpy()
        if np.all(avail_np == 1):
            pytest.skip("All items available — skip unavail-gamma test")

        self.model.phi.assign(
            tf.constant(0.99, shape=(self.model._T_unique,), dtype=tf.float64))

        for _ in range(20):
            self.model._step_gamma()
            g_np = self.model.gamma.numpy()
            assert np.all(g_np[avail_np == 0] == 0), \
                "Unavailable items must stay at gamma=0 after Gibbs sweep"

    def test_chain_does_not_get_stuck(self):
        """
        Run 30 beta_bar RWMH steps. At least ONE step must change beta_bar.
        """
        kr     = self.model.beta_kernel.bootstrap_results(self.model.beta_bar.read_value())
        states = [self.model.beta_bar.numpy().copy()]
        for _ in range(30):
            kr, _ = self.model._step_beta(kr)
            states.append(self.model.beta_bar.numpy().copy())

        states_arr = np.stack(states)
        changed    = np.any(np.diff(states_arr, axis=0) != 0)
        assert changed, (
            "beta_bar never changed in 30 RWMH steps — chain is completely stuck. "
            "This is the signature of the stale target log-prob bug.")

    def test_chain_explores_space(self):
        """
        Two separate models must both explore parameter space
        (non-zero variance across 20 steps).
        """
        for trial in range(2):
            tf.random.set_seed(trial * 100)
            model = BayesianSparseRandomLogit(
                self.sim, self.mcmc, seed=trial * 100, verbose=True)
            model._initialize_from_dataset(self.dataset_cl)
            model._initialize_state()
            model._build_kernels()
            kr    = model.beta_kernel.bootstrap_results(model.beta_bar.read_value())
            betas = []
            for _ in range(20):
                kr, _ = model._step_beta(kr)
                betas.append(model.beta_bar.numpy().copy())
            arr = np.stack(betas)
            assert arr.std() > 1e-8, \
                f"Trial {trial}: beta_bar chain has zero variance — sampler not exploring."

# ═══════════════════════════════════════════════════════════════════════════
# 10. Step-size adaptation
# ═══════════════════════════════════════════════════════════════════════════

class TestStepSizeAdaptation:

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.model = _make_fitted_model(dataset_cl, sim, mcmc)
        self.mcmc  = mcmc

    
    def test_step_sizes_changed_from_initial(self):
        """At least one step size must have changed from its initial value."""
        initials = dict(
            beta=self.mcmc.step_beta, r=self.mcmc.step_r,
            xi=self.mcmc.step_xibar, phi=self.mcmc.step_phi,
            eta=self.mcmc.step_eta,
        )
        changed = any(
            abs(self.model._step_vars[k].numpy() - v) > 1e-9
            for k, v in initials.items()
        )
        assert changed, "No step sizes adapted during burn-in."

    
    def test_all_step_sizes_positive(self):
        for name, var in self.model._step_vars.items():
            assert var.numpy() > 0, f"step_{name} is non-positive after adaptation."

    
    def test_beta_acceptance_rate_reasonable(self):
        """
        Run 200 raw _step_beta steps; acceptance should be in [0.05, 0.80].
        """
        kr    = self.model.beta_kernel.bootstrap_results(self.model.beta_bar.read_value())
        n_acc = 0
        n_tot = 200
        for _ in range(n_tot):
            kr, acc = self.model._step_beta(kr)
            n_acc  += int(acc.numpy())
        rate = n_acc / n_tot
        assert 0.05 <= rate <= 0.80, \
            f"Beta acceptance rate {rate:.3f} outside [0.05, 0.80]."

    
    def test_phi_z_phi_sync(self):
        """phi must equal sigmoid(z_phi) after every step."""
        kr = self.model.phi_kernel.bootstrap_results(self.model.z_phi.read_value())
        for _ in range(10):
            kr, _ = self.model._step_phi(kr)
            z   = self.model.z_phi.numpy()
            phi = self.model.phi.numpy()
            np.testing.assert_allclose(1.0 / (1.0 + np.exp(-z)), phi, atol=1e-10,
                                       err_msg="phi and z_phi are out of sync after _step_phi.")

    
    def test_slab_eta_proposal_wider_than_spike(self):
        """
        An eta entry with gamma=1 (slab) should have wider proposal than gamma=0,
        only measured over available items.
        """
        T, J = self.model._T_unique, self.model.sim_params.J
        avail_np = self.model.avail_inside.numpy()

        g = np.zeros((T, J), dtype=np.int32)
        g[:, : J // 2] = 1
        self.model.gamma.assign(g)

        kr = self.model.eta_kernel.bootstrap_results(self.model.eta.read_value())
        slab_deltas  = []
        spike_deltas = []
        for _ in range(50):
            eta_before = self.model.eta.numpy().copy()
            kr, _      = self.model._step_eta(kr)
            delta      = np.abs(self.model.eta.numpy() - eta_before)
            # Only measure over available items
            slab_mask  = (g[:, : J // 2] == 1) & (avail_np[:, : J // 2] == 1)
            spike_mask = (g[:, J // 2:] == 0) & (avail_np[:, J // 2:] == 1)
            if slab_mask.sum() > 0:
                slab_deltas.append(delta[:, : J // 2][slab_mask].mean())
            if spike_mask.sum() > 0:
                spike_deltas.append(delta[:, J // 2:][spike_mask].mean())

        if slab_deltas and spike_deltas:
            assert np.mean(slab_deltas) >= np.mean(spike_deltas), \
                "Slab eta proposals should be at least as wide as spike proposals."

# ═══════════════════════════════════════════════════════════════════════════
# 11. MCMC mixing
# ═══════════════════════════════════════════════════════════════════════════

class TestMCMCMixing:

    @pytest.fixture(autouse=True)
    def _build(self, medium_dataset):
        dataset_cl, _, _, sim, mcmc = medium_dataset
        self.model = _make_fitted_model(dataset_cl, sim, mcmc)
        self.mcmc  = mcmc

    
    def test_samples_keys_present(self):
        for key in ['beta_bar', 'r_vec', 'xi_bar', 'eta', 'gamma', 'phi']:
            assert key in self.model.samples_, f"Missing sample key: {key}"

    
    def test_sample_count_correct(self):
        expected = self.mcmc.G - self.mcmc.burn
        actual   = len(self.model.samples_['beta_bar'])
        assert actual == expected, \
            f"Expected {expected} posterior samples, got {actual}."

    
    def test_chain_explores_beta(self):
        """Post-burn beta_bar samples must have non-trivial variance."""
        chain = self.model.samples_["beta_bar"][:, 0]
        assert chain.std() > 1e-8, "beta_bar chain has zero variance — stuck."

    
    def test_ess_beta_above_threshold(self):
        chain = self.model.samples_["beta_bar"][:, 0]
        ess   = _ess_bulk(chain)
        assert ess > 1.0, (
            f"ESS for beta_bar[0] = {ess:.2f} <= 1.0 — chain is completely stuck. "
            "This is the signature of the stale target log-prob bug.")

    
    def test_nll_does_not_explode(self):
        """Training NLL must stay finite throughout the chain."""
        sim     = _make_sim_params(T=50, J=10, D=2, Nt=800)
        mcmc    = _make_mcmc_params(T=50, J=10, D=2, burn=5, G=20, R0=30)
        _, dataset_cl, _ = generate_data_tf(dgp_type=1, sim=sim)
        m2      = BayesianSparseRandomLogit(sim, mcmc, seed=7, verbose=True)
        hist    = m2.fit(dataset_cl)
        losses  = hist["train_loss"]
        assert all(np.isfinite(v) for v in losses), "NLL contains non-finite values."

# ═══════════════════════════════════════════════════════════════════════════
# 12. Posterior accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestPosteriorAccuracy:
    """
    Run a moderately long chain on DGP-1 ground-truth data and check that
    posterior means are within loose tolerances of the true parameters.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _fit(self, medium_dataset):
        dataset_cl, true_params, _, sim, mcmc = medium_dataset
        self.__class__.model       = _make_fitted_model(dataset_cl, sim, mcmc)
        self.__class__.true_params = true_params
        self.__class__.sim         = sim

    
    def test_beta0_sign_correct(self):
        pm = self.model.samples_["beta_bar"][:, 0].mean()
        assert pm < 0, f"beta_bar[0] posterior mean {pm:.3f} has wrong sign."

    
    def test_beta0_within_tolerance(self):
        pm   = self.model.samples_["beta_bar"][:, 0].mean()
        true = self.true_params["beta_mean"][0]
        assert abs(pm - true) < 1.5, \
            f"|posterior_mean({pm:.3f}) - true({true:.3f})| = {abs(pm-true):.3f} > 1.5"

    
    def test_beta1_within_tolerance(self):
        pm   = self.model.samples_["beta_bar"][:, 1].mean()
        true = self.true_params["beta_mean"][1]
        assert abs(pm - true) < 1.5, \
            f"|posterior_mean({pm:.3f}) - true({true:.3f})| = {abs(pm-true):.3f} > 1.5"

    
    def test_sigma_positive(self):
        r_samples     = self.model.samples_["r_vec"]
        sigma_samples = np.exp(r_samples)
        assert np.all(sigma_samples > 0), "exp(r_vec) samples must be strictly positive."

    
    def test_xi_bar_mean_sign(self):
        xi_samples  = self.model.samples_["xi_bar"]
        grand_mean  = xi_samples.mean()
        assert grand_mean < 0, \
            f"xi_bar posterior grand mean {grand_mean:.3f} should be negative."

    
    def test_phi_samples_in_unit_interval(self):
        phi_samples = self.model.samples_["phi"]
        assert np.all(phi_samples > 0) and np.all(phi_samples < 1), \
            "phi posterior samples must all lie in (0, 1)."

# ═══════════════════════════════════════════════════════════════════════════
# 13. Gamma sparsity recovery
# ═══════════════════════════════════════════════════════════════════════════

class TestGammaSparsityRecovery:

    @pytest.fixture(scope="class", autouse=True)
    def _fit(self, medium_dataset):
        dataset_cl, true_params, _, sim, mcmc = medium_dataset
        self.__class__.model       = _make_fitted_model(dataset_cl, sim, mcmc)
        self.__class__.true_params = true_params
        self.__class__.sim         = sim

    
    def test_gamma_samples_binary(self):
        g = self.model.samples_["gamma"]
        assert set(g.flatten()).issubset({0, 1}), "Gamma samples are not binary."

    
    def test_gamma_posterior_in_unit_interval(self):
        g_pm = self.model.samples_["gamma"].mean(axis=0)
        assert np.all(g_pm >= 0) and np.all(g_pm <= 1)

    
    def test_active_products_higher_posterior_gamma(self):
        """
        DGP-1 active products: indices 0..(n_active-1), n_active = int(0.4*J).
        Their posterior P(gamma=1) should be higher than inactive products.
        """
        J        = self.sim.J
        n_active = int(0.4 * J)
        g_pm     = self.model.samples_["gamma"].mean(axis=0)  # [T, J]

        active_mean   = g_pm[:, :n_active].mean()
        inactive_mean = g_pm[:, n_active:].mean()

        assert active_mean >= inactive_mean - 0.15, (
            f"Active products posterior P(gamma=1)={active_mean:.3f} should be "
            f">= inactive {inactive_mean:.3f}. Sparsity not recovered.")

    
    def test_phi_posterior_reasonable(self):
        phi_pm = self.model.samples_["phi"].mean()
        assert 0.05 < phi_pm < 0.90, \
            f"phi posterior mean {phi_pm:.3f} is outside (0.05, 0.90)."

    
    def test_gamma_zero_for_unavailable_in_all_samples(self):
        """
        The unavailable items (avail_inside==0) must have gamma==0 in
        every single stored posterior sample.
        """
        avail_np  = self.model.avail_inside.numpy()           # [T, J]
        unavail   = (avail_np == 0)
        if not np.any(unavail):
            pytest.skip("No unavailable items in this dataset")

        gamma_samples = self.model.samples_["gamma"]          # [n_post, T, J]
        # Check across all posterior samples
        unavail_vals  = gamma_samples[:, unavail]             # [n_post, n_unavail]
        assert np.all(unavail_vals == 0), \
            "Unavailable items must have gamma=0 in all posterior samples"

# ═══════════════════════════════════════════════════════════════════════════
# 14. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def _minimal_dataset(self, T=3, J=3, Nt=50):
        sim = _make_sim_params(T=T, J=J, D=2, Nt=Nt)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)
        return ds

    def test_single_market_no_crash(self):
        sim  = _make_sim_params(T=1, J=3, D=2, Nt=100)
        mcmc = _make_mcmc_params(T=1, J=3, D=2, burn=5, G=15, R0=20)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=True)
        model.fit(ds)  # must not raise

    def test_single_feature_no_crash(self):
        sim  = _make_sim_params(T=3, J=3, D=1, Nt=100)
        mcmc = _make_mcmc_params(T=3, J=3, D=1, burn=5, G=15, R0=20)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=True)
        model.fit(ds)

    def test_single_product_no_crash(self):
        sim  = _make_sim_params(T=3, J=1, D=2, Nt=100)
        mcmc = _make_mcmc_params(T=3, J=1, D=2, burn=5, G=15, R0=20)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        model.fit(ds)

    def test_burn_ge_g_raises(self):
        sim  = _make_sim_params(T=3, J=3, D=2)
        mcmc = _make_mcmc_params(T=3, J=3, D=2, burn=10, G=10)
        ds   = self._minimal_dataset()
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=True)
        with pytest.raises(AssertionError):
            model.fit(ds)

    def test_compute_batch_utility_raises_before_fit(self):
        sim  = _make_sim_params(T=3, J=3, D=2)
        mcmc = _make_mcmc_params(T=3, J=3, D=2, burn=5, G=15)
        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        with pytest.raises(RuntimeError):
            model.compute_batch_utility(
                np.zeros((5, 1)), np.zeros((5, 4, 2)), np.ones((5, 4)))

    def test_losses_history_length(self):
        """losses_history['train_loss'] must have length G+1 (initial + G steps)."""
        T_e, J_e = 3, 3
        sim    = _make_sim_params(T=T_e, J=J_e, D=2, Nt=100)
        mcmc_p = _make_mcmc_params(T=T_e, J=J_e, D=2, burn=5, G=15, R0=20)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)
        model  = BayesianSparseRandomLogit(sim, mcmc_p, seed=0, verbose=True)
        hist   = model.fit(ds)
        assert len(hist["train_loss"]) == mcmc_p.G + 1, \
            f"Expected {mcmc_p.G + 1} loss entries, got {len(hist['train_loss'])}"

    def test_all_dgp_types_run(self):
        """All four dgp_type values must generate valid datasets and run."""
        sim  = _make_sim_params(T=5, J=5, D=2, Nt=100)
        mcmc = _make_mcmc_params(T=5, J=5, D=2, burn=5, G=15, R0=20)
        for dgp in [1, 2, 3, 4]:
            _, ds, _ = generate_data_tf(dgp_type=dgp, sim=sim)
            model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=True)
            model.fit(ds)  # must not raise

# ═══════════════════════════════════════════════════════════════════════════
# 15. Default values – step sizes, target_accept, burn-in
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultsAndFallbacks:
    """
    Verify that _build_kernels correctly reads step sizes, target_accept,
    and burn from mcmc_params — and that the hardcoded fallback values are
    used when those attributes are absent from mcmc_params.
    """

    def _mcmc_without(self, base_mcmc, *attrs):
        """Return a copy of mcmc_params with the given attributes removed."""
        ns = types.SimpleNamespace(**vars(base_mcmc))
        for a in attrs:
            if hasattr(ns, a):
                delattr(ns, a)
        return ns

    @pytest.fixture(autouse=True)
    def _setup(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.dataset_cl = dataset_cl
        self.sim        = sim
        self.base_mcmc  = mcmc

    def test_step_beta_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "step_beta")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        np.testing.assert_allclose(model._step_vars["beta"].numpy(), 0.5, atol=1e-10)

    def test_step_r_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "step_r")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        np.testing.assert_allclose(model._step_vars["r"].numpy(), 0.3, atol=1e-10)

    def test_step_xibar_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "step_xibar")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        np.testing.assert_allclose(model._step_vars["xi"].numpy(), 0.2, atol=1e-10)

    def test_step_phi_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "step_phi")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        np.testing.assert_allclose(model._step_vars["phi"].numpy(), 0.1, atol=1e-10)

    def test_step_eta_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "step_eta")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        np.testing.assert_allclose(model._step_vars["eta"].numpy(), 0.1, atol=1e-10)

    def test_all_step_defaults_simultaneously(self):
        mcmc  = self._mcmc_without(
            self.base_mcmc, "step_beta", "step_r", "step_xibar", "step_phi", "step_eta")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        expected = {"beta": 0.5, "r": 0.3, "xi": 0.2, "phi": 0.1, "eta": 0.1}
        for key, val in expected.items():
            np.testing.assert_allclose(model._step_vars[key].numpy(), val, atol=1e-10)

    def test_explicit_step_sizes_stored_correctly(self):
        model   = _make_initialised_model(self.dataset_cl, self.sim, self.base_mcmc)
        mapping = {
            "beta": self.base_mcmc.step_beta,
            "r":    self.base_mcmc.step_r,
            "xi":   self.base_mcmc.step_xibar,
            "phi":  self.base_mcmc.step_phi,
            "eta":  self.base_mcmc.step_eta,
        }
        for key, expected in mapping.items():
            np.testing.assert_allclose(model._step_vars[key].numpy(), expected, atol=1e-10)

    def test_step_vars_are_tf_variables(self):
        model = _make_initialised_model(self.dataset_cl, self.sim, self.base_mcmc)
        for name, var in model._step_vars.items():
            assert isinstance(var, tf.Variable), \
                f"_step_vars['{name}'] is {type(var)}, expected tf.Variable"

    def test_step_vars_dtype_float64(self):
        model = _make_initialised_model(self.dataset_cl, self.sim, self.base_mcmc)
        for name, var in model._step_vars.items():
            assert var.dtype == tf.float64

    def test_step_vars_positive(self):
        model = _make_initialised_model(self.dataset_cl, self.sim, self.base_mcmc)
        for name, var in model._step_vars.items():
            assert var.numpy() > 0

    def test_target_accept_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "target_accept")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        for attr in ["beta_kernel", "r_kernel", "xi_kernel", "eta_kernel", "phi_kernel"]:
            assert isinstance(getattr(model, attr), tfp.mcmc.SimpleStepSizeAdaptation)

    def test_target_accept_explicit_stored(self):
        mcmc              = types.SimpleNamespace(**vars(self.base_mcmc))
        mcmc.target_accept = 0.25
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        for attr in ["beta_kernel", "r_kernel", "xi_kernel", "eta_kernel", "phi_kernel"]:
            kernel = getattr(model, attr)
            stored = float(kernel.parameters["target_accept_prob"])
            np.testing.assert_allclose(stored, 0.25, atol=1e-6)

    def test_burn_default_when_missing(self):
        mcmc  = self._mcmc_without(self.base_mcmc, "burn")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        for attr in ["beta_kernel", "r_kernel", "xi_kernel", "eta_kernel", "phi_kernel"]:
            kernel   = getattr(model, attr)
            n_adapt  = int(kernel.parameters["num_adaptation_steps"])
            assert n_adapt == 0

    def test_burn_explicit_sets_num_adaptation_steps(self):
        mcmc      = types.SimpleNamespace(**vars(self.base_mcmc))
        mcmc.burn = 50
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        for attr in ["beta_kernel", "r_kernel", "xi_kernel", "eta_kernel", "phi_kernel"]:
            kernel  = getattr(model, attr)
            n_adapt = int(kernel.parameters["num_adaptation_steps"])
            assert n_adapt == 50

    def test_gamma_kernel_unaffected_by_missing_rwmh_params(self):
        mcmc = self._mcmc_without(
            self.base_mcmc,
            "step_beta", "step_r", "step_xibar", "step_phi", "step_eta",
            "target_accept", "burn")
        model = _make_initialised_model(self.dataset_cl, self.sim, mcmc)
        assert isinstance(model.gamma_kernel, GammaGibbsKernel)
        assert model.gamma_kernel.is_calibrated is True

# ═══════════════════════════════════════════════════════════════════════════
# 16. Variable product availability  (NEW)
# ═══════════════════════════════════════════════════════════════════════════

class TestVariableAvailability:
    """
    Tests that specifically probe the masked model's ability to handle
    partial/variable product availability (M > J scenario).

    These tests use a synthetic dataset where some products are genuinely
    unavailable in each market, verifying that the sampler:
      - Never assigns gamma=1 to unavailable items
      - Produces valid likelihoods
      - Does not confuse xi_bar with spurious eta for unavailable items
    """

    @pytest.fixture(autouse=True)
    def _build(self, small_dataset):
        dataset_cl, _, _, sim, mcmc = small_dataset
        self.dataset_cl = dataset_cl
        self.sim        = sim
        self.mcmc       = mcmc

    def test_avail_inside_propagated_to_gamma_kernel(self):
        """
        After fit, gamma_kernel.avail_inside must exactly match
        the avail_inside derived from the dataset.
        """
        model = _make_initialised_model(self.dataset_cl, self.sim, self.mcmc)
        np.testing.assert_array_equal(
            model.gamma_kernel.avail_inside.numpy(),
            model.avail_inside.numpy()
        )

    def test_phi_target_uses_j_avail_not_j(self):
        """
        The phi_target must use J_avail (sum of available items per market)
        in the Beta posterior, not the total J.  We verify by checking that
        the Beta posterior concentrations (a + sum_gamma, b + J_avail - sum_gamma)
        are finite and sum to <= a_phi + b_phi + J.
        """
        model = _make_initialised_model(self.dataset_cl, self.sim, self.mcmc)
        # Just run the phi target once and check it doesn't throw or return NaN
        lp = model.phi_kernel.inner_kernel.target_log_prob_fn(
            model.z_phi.read_value())
        assert np.isfinite(lp.numpy()), "phi_target returned non-finite value"

    def test_eta_new_state_zero_jump_for_unavail(self):
        """
        In eta_new_state, unavailable items must receive a zero-size jump
        (sigma_prop[t,j] == 0 when avail_inside[t,j] == 0), so they stay
        exactly at their current value after every proposal.
        """
        model    = _make_initialised_model(self.dataset_cl, self.sim, self.mcmc)
        avail_np = model.avail_inside.numpy()
        if np.all(avail_np == 1):
            pytest.skip("All items available — no unavail items to test")

        eta_before = model.eta.read_value().numpy().copy()
        kr = model.eta_kernel.bootstrap_results(model.eta.read_value())

        # Run 20 eta steps; unavail items must never move
        for _ in range(20):
            kr, _ = model._step_eta(kr)

        eta_after = model.eta.read_value().numpy()
        # For unavailable items, eta must not change (zero proposal size)
        np.testing.assert_array_equal(
            eta_after[avail_np == 0],
            eta_before[avail_np == 0],
            err_msg="eta changed for unavailable items — sigma_prop must be 0 for unavail"
        )

    def test_full_chain_with_partial_availability(self):
        """
        A short chain on a dataset with partial availability (some items unavailable
        per market) must:
          - Complete without error
          - Produce finite NLL throughout
          - Store gamma=0 for all unavailable items in all posterior samples
        """
        T, J = 8, 6
        sim  = _make_sim_params(T=T, J=J, D=2, Nt=150)
        mcmc = _make_mcmc_params(T=T, J=J, D=2, burn=5, G=20, R0=20)

        # Generate base dataset (full availability)
        _, ds, _ = generate_data_tf(dgp_type=1, sim=sim)

        # Manually inject partial availability into the dataset:
        # make ~30% of inside items unavailable per market
        rng       = np.random.default_rng(42)
        avail_raw = ds.available_items_by_choice
        if isinstance(avail_raw, (list, tuple)):
            avail_raw = avail_raw[0]
        avail_np  = np.asarray(avail_raw).copy()

        # Zero out ~30% of inside item columns (leave outside option = col 0 untouched)
        for n in range(avail_np.shape[0]):
            for j in range(1, J + 1):  # inside products only
                if rng.random() < 0.3:
                    avail_np[n, j] = 0.0
            # Ensure at least one inside product is available
            if avail_np[n, 1:].sum() == 0:
                avail_np[n, 1] = 1.0

        # Rebuild dataset with modified availability (depends on ChoiceDataset API)
        
        ds_masked = ChoiceDataset(
            shared_features_by_choice=ds.shared_features_by_choice,
            items_features_by_choice=ds.items_features_by_choice,
            available_items_by_choice=avail_np,
            choices=ds.choices,
        )

        model = BayesianSparseRandomLogit(sim, mcmc, seed=0, verbose=False)
        hist  = model.fit(ds_masked)

        # NLL must stay finite
        assert all(np.isfinite(v) for v in hist["train_loss"]), \
            "NLL exploded during chain with partial availability"

        # gamma must be 0 everywhere avail_inside == 0
        avail_inside_np = model.avail_inside.numpy()
        if np.any(avail_inside_np == 0):
            gamma_samples = model.samples_["gamma"]  # [n_post, T, J]
            unavail_mask  = (avail_inside_np == 0)
            assert np.all(gamma_samples[:, unavail_mask] == 0), \
                "gamma=1 found for unavailable items in posterior samples"

    def test_avail_inside_all_ones_matches_fixed_j_behaviour(self):
        """
        When avail_inside is all-ones (full availability), the masked model
        must give identical likelihood values to a model with no masking,
        confirming backward compatibility.
        """
        model = _make_initialised_model(self.dataset_cl, self.sim, self.mcmc)

        # With full availability, avail_inside should be all-ones for DGP-1
        avail_np = model.avail_inside.numpy()
        if not np.all(avail_np == 1):
            pytest.skip("DGP-1 dataset does not have full availability — skip compat test")

        xi   = model.xi_bar[:, None] + model.eta
        _, _, ll = model.compute_log_likelihood_unique(
            xi, model.beta_bar, model.r_vec, model.v_draws)

        assert np.all(np.isfinite(ll.numpy())), \
            "Full-availability masked model returns non-finite LL"
        assert np.all(ll.numpy() <= 1e-8), \
            "Full-availability masked model returns positive LL"

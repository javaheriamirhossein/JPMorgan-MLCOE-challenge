"""
Unit and integration tests for BayesianSparseDeepHalo.BLP

Test summary
----------
 1. Unit tests — _rc_logit_draw_probs              
 2. Unit tests — blp_build_ivs (all 3 iv_types)     
 3. Unit tests — blp_contraction_all_markets       
 4. Unit tests — FastBLP.__init__ (shapes, matrices) 
 5. Unit tests — FastBLP.compute_g_xi (2SLS math)    
 6. Unit tests — FastBLP.obj (penalty, positivity)  
 7. Sanity checks — moment condition E[Z·xi] ~= 0   
 8. Integration — blp_estimator_fast recovers params 
 9. Edge cases — sigma boundary, zero shares, T=1,J=1
10. Numerical — convergence, warm-start, conditioning
"""

import sys
import numpy as np
import pytest
from numpy.random import default_rng
from scipy.special import logsumexp

from BayesianSparseDeepHalo.BLP import (
    _rc_logit_draw_probs,
    blp_build_ivs,
    blp_contraction_all_markets,
    FastBLP,
    blp_estimator_fast,
)

from BayesianSparseDeepHalo.GenerateData import SimParams, generate_data_tf


# ===========================================================================
# SHARED DGP HELPER — wraps generate_data_tf
# ===========================================================================

def make_synthetic_data(T=20, J=5, Nt=5000, beta_p=-1.0, beta_w=0.5,
                        sigma_p=1.5, seed=42):
    """
    Generate a minimal DGP satisfying BLP moment conditions.
    """
    sim = SimParams(
        T=T,
        J=J,
        Nt=Nt,
        D=2,                                    # price + quality (2 features)
        beta_mean=np.array([beta_p, beta_w]),   # [D]: mean coefficients
        sigma_beta=np.array([sigma_p, 0.0]),    # [D]: only price is random
        xi_bar=-1.0,
        seed=seed,
    )

    # dgp_type=1: sparse exogenous shocks — E[Z·xi]=0 holds
    data_mcmc, _, true_params = generate_data_tf(dgp_type=1, sim=sim)

    # Unpack into the format the existing tests expect
    X      = data_mcmc["X"]          # [T, J, 2]: (price, w)
    q      = data_mcmc["q"]          # [T, J+1]: choice counts
    u_cost = true_params["u_cost"]   # [T, J]:   cost shifter for IVs

    data = {"X": X, "q": q, "Nt": Nt}

    # Reconstruct the true mean utility (delta) from the true parameters
    beta_mean  = true_params["beta_mean"]   # [D]
    xi_star    = true_params["xi_star"]     # [T, J]
    delta_true = (beta_mean[0] * X[:, :, 0]   # beta_p * price
                + beta_mean[1] * X[:, :, 1]   # beta_w * quality
                + xi_star)                     # [T, J]

    true = {
        "beta_p":  float(beta_mean[0]),
        "beta_w":  float(beta_mean[1]),
        "sigma_p": float(sim.sigma_beta[0]),
        "xi":      xi_star,       # [T, J]
        "delta":   delta_true,    # [T, J]
    }

    return data, u_cost, true


def _exact_mc_shares(delta, price, sigma_p, v):
    """
    Compute MC-shares using exactly the same formula and draws as the
    contraction mapping. 
    """
    T, J = delta.shape
    s = []
    for t in range(T):
        mu    = sigma_p * v[:, None] * price[t][None, :]
        U_in  = delta[t][None, :] + mu
        U_all = np.concatenate([np.zeros((len(v), 1)), U_in], axis=1)
        denom = logsumexp(U_all, axis=1, keepdims=True)
        s.append(np.exp(U_all - denom).mean(axis=0))
    return np.stack(s)  # (T, J+1)


# ===========================================================================
# MODULE-SCOPED FIXTURES
# ===========================================================================

@pytest.fixture(scope="module")
def small_data():
    """20 markets, 5 products — fast for unit tests"""
    return make_synthetic_data(T=20, J=5, Nt=5000)


@pytest.fixture(scope="module")
def large_data():
    """100 markets, 15 products — for integration tests."""
    return make_synthetic_data(T=100, J=15, Nt=10_000)


# ===========================================================================
# 1. UNIT TESTS — _rc_logit_draw_probs
# ===========================================================================

class TestRcLogitDrawProbs:

    def test_output_shape(self):
        """Output must be (R, J+1) including the outside option."""
        J, R = 5, 100
        delta   = np.zeros(J)
        price   = np.ones(J)
        v_draws = np.random.randn(R)
        probs   = _rc_logit_draw_probs(delta, price, 1.0, v_draws)
        assert probs.shape == (R, J + 1), \
            f"Expected ({R}, {J+1}), got {probs.shape}"

    def test_probabilities_sum_to_one(self):
        """Choice probs must sum to 1 across all products for each draw"""
        delta   = np.array([-1.0, 0.0, 1.0])
        price   = np.array([1.5, 1.0, 0.5])
        v_draws = np.linspace(-2, 2, 50)
        probs   = _rc_logit_draw_probs(delta, price, 0.5, v_draws)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12,
            err_msg="Row probabilities do not sum to 1.")

    def test_probabilities_non_negative(self):
        """All probabilities must be >= 0"""
        delta   = np.random.randn(8)
        price   = np.abs(np.random.randn(8)) + 0.1
        v_draws = np.random.randn(200)
        probs   = _rc_logit_draw_probs(delta, price, 1.0, v_draws)
        assert np.all(probs >= 0), "Negative probabilities encountered."

    def test_sigma_zero_collapses_to_mnl(self):
        """When sigma=0 the RC logit must collapse to the standard logit"""
        delta   = np.array([0.5, -0.5, 1.0])
        price   = np.array([1.0, 2.0, 1.5])
        v_draws = np.random.randn(500)
        probs   = _rc_logit_draw_probs(delta, price, 0.0, v_draws)
        u_all   = np.concatenate([[0.0], delta])
        exp_u   = np.exp(u_all - u_all.max())
        logit_p = exp_u / exp_u.sum()
        np.testing.assert_allclose(probs.mean(axis=0), logit_p, atol=1e-10,
            err_msg="sigma=0 RC logit does not match plain logit")

    def test_outside_option_is_column_zero(self):
        """With very negative inside utilities the outside option dominates"""
        delta   = np.full(5, -10.0)
        price   = np.ones(5)
        v_draws = np.zeros(10)
        probs   = _rc_logit_draw_probs(delta, price, 1.0, v_draws)
        assert probs[:, 0].mean() > 0.99, \
            "Outside option should dominate when delta << 0"

    def test_high_sigma_increases_variance_across_draws(self):
        """Higher sigma should increase cross-draw variance in predicted shares."""
        delta   = np.zeros(3)
        price   = np.array([1.0, 2.0, 3.0])
        v_draws = np.linspace(-3, 3, 300)
        var_low  = _rc_logit_draw_probs(delta, price, 0.1, v_draws)[:, 1].var()
        var_high = _rc_logit_draw_probs(delta, price, 2.0, v_draws)[:, 1].var()
        assert var_high > var_low, \
            "Higher sigma should produce more cross-draw variation."

    def test_large_negative_delta_drives_share_to_zero(self):
        """A product with delta -> -inf should have near-zero share."""
        delta   = np.array([-100.0, 0.0, 0.0])
        price   = np.ones(3)
        v_draws = np.random.randn(500)
        probs   = _rc_logit_draw_probs(delta, price, 0.1, v_draws)
        assert probs[:, 1].mean() < 1e-10, \
            "Product with delta=-100 should have essentially zero share."


# ===========================================================================
# 2. UNIT TESTS — blp_build_ivs
# ===========================================================================

class TestBlpBuildIvs:

    @pytest.fixture
    def market_inputs(self):
        rng = default_rng(0)
        J   = 10
        return rng.uniform(1, 2, J), rng.normal(0, 0.7, J)

    def test_iv_type1_shape(self, market_inputs):
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=1)
        assert Z.shape == (len(w), 5), f"Expected (J,5), got {Z.shape}"

    def test_iv_type2_shape(self, market_inputs):
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=2)
        assert Z.shape == (len(w), 5), f"Expected (J,5), got {Z.shape}"

    def test_iv_type3_shape(self, market_inputs):
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=3)
        assert Z.shape == (len(w), 4), f"Expected (J,4), got {Z.shape}"

    @pytest.mark.parametrize("iv_type", [1, 2, 3])
    def test_first_column_is_ones(self, market_inputs, iv_type):
        """First column (intercept) must be all ones for every iv_type"""
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=iv_type)
        np.testing.assert_array_equal(Z[:, 0], np.ones(len(w)),
            err_msg=f"iv_type={iv_type}: first column should be all ones.")

    def test_second_column_is_w(self, market_inputs):
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=1)
        np.testing.assert_array_equal(Z[:, 1], w,
              err_msg=f"iv_type=1: second column should be w.")

    def test_iv_type3_competitor_sum_correct(self, market_inputs):
        """iv_type=3 col 2 should be perfectly correlated with sum_{k!=j} w_k"""
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=3)
        expected_raw = np.sum(w) - w
        corr = np.corrcoef(Z[:, 2], expected_raw)[0, 1]
        assert corr > 0.999, \
            f"Col 2 should track sum_{{k!=j}} w_k (r={corr:.4f})."

    def test_iv_type3_diff_sq_non_negative(self, market_inputs):
        """Sum-of-squared distances must be non-negative before standardisation"""
        w, _ = market_inputs
        J    = len(w)
        raw  = np.array([np.sum((w[j] - w)**2) for j in range(J)])
        assert np.all(raw >= 0), "Sum of squared distances must be >= 0"

    def test_invalid_iv_type_raises(self, market_inputs):
        w, u = market_inputs
        with pytest.raises(ValueError, match="iv_type must be"):
            blp_build_ivs(w, u, iv_type=99)

    def test_iv_type2_standardised_col1_zero_mean(self, market_inputs):
        """Standardised w-tilde column (col 1) should have zero mean"""
        w, u = market_inputs
        Z    = blp_build_ivs(w, u, iv_type=2)
        assert abs(Z[:, 1].mean()) < 1e-10, \
            "Standardised w-tilde should have mean ~= 0"


# ===========================================================================
# 3. UNIT TESTS — blp_contraction_all_markets
# ===========================================================================

class TestContractionMapping:
    """
    The contraction mapping inverts EXACTLY within the MC approximation
    defined by the given v_draws.  Here we use shares with the same
    v_draws that are passed to the contraction, so that inversion is numerically
    exact (up to tol=1e-10).
    """

    def _make_exact_shares(self, T, J, sigma_p, N_draw=500, seed=7):
        rng        = default_rng(seed)
        delta_true = rng.uniform(-1.0, 1.0, size=(T, J))
        price      = rng.uniform(1.0,  2.0, size=(T, J))
        X          = np.stack([price, np.ones_like(price)], axis=-1)
        v          = rng.normal(0, 1, N_draw)
        s          = _exact_mc_shares(delta_true, price, sigma_p, v)
        return X, s, delta_true, v

    def test_output_shape(self):
        """Contraction mapping output must be (T, J)."""
        T, J = 4, 5
        X, s, _, v = self._make_exact_shares(T, J, 1.0)
        delta_hat  = blp_contraction_all_markets(s, X, 1.0, v, np.zeros((T, J)))
        assert delta_hat.shape == (T, J)

    def test_predicted_shares_match_observed(self):
        """
        After convergence, predicted MC-shares (same v) must equal observed
        shares to within 1e-6 (contraction tol=1e-10 used).
        """
        T, J, sigma_p = 5, 4, 1.2
        X, s, _, v    = self._make_exact_shares(T, J, sigma_p)
        delta_hat     = blp_contraction_all_markets(
            s, X, sigma_p, v, np.zeros((T, J)), tol=1e-10)
        price         = X[:, :, 0]
        s_hat         = _exact_mc_shares(delta_hat, price, sigma_p, v)
        np.testing.assert_allclose(s_hat[:, 1:], s[:, 1:], atol=1e-6,
            err_msg="Predicted shares (same v) don't match observed.")

    def test_warm_start_gives_same_result(self):
        """Starting from delta_true vs zeros should give identical converged delta."""
        T, J, sigma_p         = 3, 5, 1.0
        X, s, delta_true, v   = self._make_exact_shares(T, J, sigma_p)
        d_cold = blp_contraction_all_markets(
            s, X, sigma_p, v, np.zeros((T, J)), tol=1e-12)
        d_warm = blp_contraction_all_markets(
            s, X, sigma_p, v, delta_true.copy(), tol=1e-12)
        np.testing.assert_allclose(d_cold, d_warm, atol=1e-8,
            err_msg="Contraction result must not depend on warm-start value")

    def test_sigma_zero_recovers_logit_inversion(self):
        """At sigma=0 the contraction must recover log(s_j) - log(s_0) exactly"""
        T, J = 4, 6
        rng       = default_rng(13)
        delta_true = rng.uniform(-1, 1, (T, J))
        exp_d      = np.exp(delta_true)
        denom      = 1.0 + exp_d.sum(axis=1, keepdims=True)
        s_in       = exp_d / denom
        s_out      = 1.0 / denom
        s          = np.concatenate([s_out, s_in], axis=1)
        X          = np.stack([np.ones((T, J)), np.ones((T, J))], axis=-1)
        v          = np.zeros(10)
        delta_hat  = blp_contraction_all_markets(
            s, X, 0.0, v, np.zeros((T, J)), tol=1e-12)
        expected   = np.log(s_in) - np.log(s_out)
        np.testing.assert_allclose(delta_hat, expected, atol=1e-8,
            err_msg="sigma=0 contraction should recover logit inversion exactly.")

    def test_contraction_does_not_mutate_input_shares(self):
        """The function must not modify the passed s_obs array"""
        T, J    = 3, 4
        X, s, _, v = self._make_exact_shares(T, J, 1.0)
        s_copy  = s.copy()
        blp_contraction_all_markets(s, X, 1.0, v, np.zeros((T, J)))
        np.testing.assert_array_equal(s, s_copy,
            err_msg="blp_contraction_all_markets must not mutate input shares")


# ===========================================================================
# 4. UNIT TESTS — FastBLP.__init__
# ===========================================================================

class TestFastBLPInit:

    def test_x_mat_shape(self, small_data):
        data, u_cost, _ = small_data
        T, J, _         = data["X"].shape
        blp             = FastBLP(data, u_cost, iv_type=1)
        assert blp.X_mat.shape == (T * J, 3)

    def test_x_mat_first_column_is_ones(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        np.testing.assert_array_equal(blp.X_mat[:, 0], np.ones(blp.N),
            err_msg="First column of X_mat must be all ones.")

    @pytest.mark.parametrize("iv_type,K", [(1, 5), (2, 5), (3, 4)])
    def test_z_mat_shape(self, small_data, iv_type, K):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, iv_type=iv_type)
        assert blp.Z_mat.shape == (blp.N, K), \
            f"iv_type={iv_type}: expected ({blp.N},{K}), got {blp.Z_mat.shape}"

    def test_w1_is_symmetric(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        np.testing.assert_allclose(blp.W1, blp.W1.T, atol=1e-10,
            err_msg="W1 must be symmetric.")

    def test_w1_is_positive_semidefinite(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        eigvals         = np.linalg.eigvalsh(blp.W1)
        assert np.all(eigvals >= -1e-8), \
            f"W1 has negative eigenvalues: {eigvals.min():.2e}"

    def test_v_draws_shape(self, small_data):
        data, u_cost, _ = small_data
        N_draw          = 150
        blp             = FastBLP(data, u_cost, N_draw=N_draw)
        assert blp.v_draws.shape == (N_draw,)

    def test_same_seed_gives_identical_draws(self, small_data):
        data, u_cost, _ = small_data
        blp1 = FastBLP(data, u_cost, seed=99)
        blp2 = FastBLP(data, u_cost, seed=99)
        np.testing.assert_array_equal(blp1.v_draws, blp2.v_draws)

    def test_different_seeds_give_different_draws(self, small_data):
        data, u_cost, _ = small_data
        blp1 = FastBLP(data, u_cost, seed=1)
        blp2 = FastBLP(data, u_cost, seed=2)
        assert not np.allclose(blp1.v_draws, blp2.v_draws)

    def test_s_obs_sums_to_one_per_market(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        np.testing.assert_allclose(blp.s_obs.sum(axis=1), 1.0, atol=1e-6,
            err_msg="Market shares (incl. outside option) must sum to 1.")

    def test_delta_last_warm_start_shape(self, small_data):
        data, u_cost, _ = small_data
        T, J, _         = data["X"].shape
        blp             = FastBLP(data, u_cost)
        assert blp.delta_last.shape == (T, J)


# ===========================================================================
# 5. UNIT TESTS — FastBLP.compute_g_xi
# ===========================================================================

class TestComputeGXi:

    def test_output_shape(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, iv_type=1)
        g               = blp.compute_g_xi(1.5, blp.W1)
        assert g.shape == (blp.K,)

    def test_beta_last_shape(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        blp.compute_g_xi(1.5, blp.W1)
        assert blp.beta_last.shape == (3,)

    def test_xi_last_shape(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        blp.compute_g_xi(1.5, blp.W1)
        assert blp.xi_last.shape == (blp.N,)

    def test_moment_condition_small_at_true_sigma(self, small_data):
        """
        At sigma=1.5 with iv_type=1 the average moments should be small.
        Tolerance 0.08 accounts for finite-sample simulation error (~1/sqrt(2000))
        """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, iv_type=1, N_draw=2000)
        g               = blp.compute_g_xi(1.5, blp.W1)
        assert np.abs(g).max() < 0.08, \
            f"Max |moment| at true sigma: {np.abs(g).max():.4f} (expected < 0.08)"

    def test_xi_equals_delta_minus_x_beta(self, small_data):
        """Structural identity: xi = delta - X*beta must hold exactly"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        blp.compute_g_xi(1.5, blp.W1)
        xi_check = blp.delta_last.flatten() - blp.X_mat @ blp.beta_last
        np.testing.assert_allclose(blp.xi_last, xi_check, atol=1e-10,
            err_msg="xi != delta - X*beta: structural identity violated.")

    def test_moment_vector_equals_z_transpose_xi_over_n(self, small_data):
        """g must equal Z'xi / N by definition"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        g               = blp.compute_g_xi(1.5, blp.W1)
        g_manual        = (blp.Z_mat.T @ blp.xi_last) / blp.N
        np.testing.assert_allclose(g, g_manual, atol=1e-12,
            err_msg="Moment vector g != Z'xi/N.")

    def test_delta_last_updated_after_call(self, small_data):
        """compute_g_xi must update delta_last """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        d_bef           = blp.delta_last.copy()
        blp.compute_g_xi(1.5, blp.W1)
        assert not np.allclose(blp.delta_last, d_bef), \
            "delta_last should be updated by compute_g_xi"


# ===========================================================================
# 6. UNIT TESTS — FastBLP.obj
# ===========================================================================

class TestObjectiveFunction:

    def test_returns_python_float(self, small_data):
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        val             = blp.obj(np.array([1.5]), blp.W1)
        assert isinstance(val, float)

    def test_penalty_for_near_zero_sigma(self, small_data):
        """obj() must return 1e12 when sigma < 1e-6"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        assert blp.obj(np.array([1e-7]), blp.W1) == 1e12

    def test_non_negative_for_valid_sigma(self, small_data):
        """GMM criterion Q = N*g'Wg >= 0 always"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            val = blp.obj(np.array([sigma]), blp.W1)
            assert val >= 0, f"Negative objective at sigma={sigma}: {val}"

    def test_finite_for_valid_sigma_range(self, small_data):
        """obj() must be finite for sigma in (0, 3]"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        for sigma in [0.01, 0.5, 1.0, 1.5, 2.5, 3.0]:
            val = blp.obj(np.array([sigma]), blp.W1)
            assert np.isfinite(val), f"Non-finite objective at sigma={sigma}: {val}"

    def test_lower_near_true_sigma_than_far(self, small_data):
        """
        The objective at sigma=1.5 (true value) should be lower than at
        sigma=0.05 or sigma=2.9.  
        """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, N_draw=1000)
        v_true  = blp.obj(np.array([1.5]), blp.W1)
        v_low   = blp.obj(np.array([0.05]), blp.W1)
        v_high  = blp.obj(np.array([2.9]), blp.W1)
        assert v_true < v_low, \
            f"obj(1.5)={v_true:.2f} should be < obj(0.05)={v_low:.2f}"
        assert v_true < v_high, \
            f"obj(1.5)={v_true:.2f} should be < obj(2.9)={v_high:.2f}"

    def test_w2_objective_finite_and_non_negative(self, small_data):
        """After building W2 from Stage-1 residuals, obj() must be finite >= 0"""
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost)
        blp.compute_g_xi(1.5, blp.W1)
        xi1  = blp.xi_last
        Z_xi = blp.Z_mat * xi1[:, None]
        S    = (Z_xi.T @ Z_xi) / blp.N
        W2   = np.linalg.pinv(S, rcond=1e-8)
        val  = blp.obj(np.array([1.5]), W2)
        assert np.isfinite(val) and val >= 0


# ===========================================================================
# 7. SANITY CHECKS — Moment condition E[Z*xi] ~= 0
# ===========================================================================

class TestMomentConditionSanity:

    @pytest.mark.parametrize("iv_type", [1, 3])
    def test_moment_condition_near_zero_at_truth(self, small_data, iv_type):
        """
        At true sigma=1.5, max|g| should be small for strong IVs (1 and 3).
        """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, iv_type=iv_type, N_draw=2000)
        g               = blp.compute_g_xi(1.5, blp.W1)
        assert np.abs(g).max() < 0.1, \
            f"iv_type={iv_type}: max|g| = {np.abs(g).max():.4f} at true sigma"

    def test_sign_of_beta_p_at_true_sigma(self, small_data):
        """
        At the true sigma, the 2SLS estimate of beta_p must be negative
        (standard demand assumption: higher price -> lower utility)
        """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, iv_type=1, N_draw=2000)
        blp.compute_g_xi(1.5, blp.W1)
        assert blp.beta_last[1] < 0, \
            f"beta_p should be negative, got {blp.beta_last[1]:.4f}"

    def test_misspecified_sigma_inflates_moments(self, small_data):
        """
        Moments at a severely misspecified sigma (0.05) must exceed moments at
        the true sigma (1.5), confirming the identification condition
        """
        data, u_cost, _ = small_data
        blp             = FastBLP(data, u_cost, N_draw=1000)
        g_true  = blp.compute_g_xi(1.5, blp.W1)
        g_wrong = blp.compute_g_xi(0.05, blp.W1)
        assert np.abs(g_wrong).max() > np.abs(g_true).max(), \
            "Misspecified sigma should yield larger moments than true sigma."


# ===========================================================================
# 8. INTEGRATION TESTS — Full estimator recovers true parameters
# ===========================================================================

class TestFullEstimator:

    
    def test_iv1_recovers_sigma(self, large_data):
        """iv_type=1 should recover sigma ~= 1.5 within +/- 0.25."""
        data, u_cost, true = large_data
        out = blp_estimator_fast(data, u_cost, iv_type=1, N_draw=300, seed=42)
        assert abs(out["sigma_p"] - true["sigma_p"]) < 0.25, \
            f"sigma_hat={out['sigma_p']:.4f}, truth={true['sigma_p']}"

    
    def test_iv1_recovers_beta_p(self, large_data):
        """iv_type=1 should recover beta_p ~= -1.0 within +/- 0.25."""
        data, u_cost, true = large_data
        out = blp_estimator_fast(data, u_cost, iv_type=1, N_draw=300, seed=42)
        assert abs(out["beta_bar"][0] - true["beta_p"]) < 0.25, \
            f"beta_p_hat={out['beta_bar'][0]:.4f}, truth={true['beta_p']}"

    
    def test_iv1_recovers_beta_w(self, large_data):
        """iv_type=1 should recover beta_w ~= 0.5 within +/- 0.25."""
        data, u_cost, true = large_data
        out = blp_estimator_fast(data, u_cost, iv_type=1, N_draw=300, seed=42)
        assert abs(out["beta_bar"][1] - true["beta_w"]) < 0.25, \
            f"beta_w_hat={out['beta_bar'][1]:.4f}, truth={true['beta_w']}"


    def test_output_keys_present(self, small_data):
        data, u_cost, _ = small_data
        out = blp_estimator_fast(data, u_cost, iv_type=1, N_draw=100, seed=0)
        for key in ["theta", "beta_bar", "sigma_p", "intercept",
                    "xi", "xi_bar", "eta", "delta", "res_stage1", "res_stage2"]:
            assert key in out, f"Missing key '{key}' in estimator output."


    def test_output_shapes(self, small_data):
        data, u_cost, _ = small_data
        T, J, _         = data["X"].shape
        out = blp_estimator_fast(data, u_cost, iv_type=1, N_draw=100, seed=0)
        assert out["theta"].shape   == (3,)
        assert out["beta_bar"].shape == (2,)
        assert out["xi"].shape      == (T, J)
        assert out["xi_bar"].shape  == (T,)
        assert out["eta"].shape     == (T, J)
        assert out["delta"].shape   == (T, J)


    def test_sigma_within_bounds(self, small_data):
        data, u_cost, _ = small_data
        bounds = [(0.01, 2.5)]
        out    = blp_estimator_fast(data, u_cost, bounds=bounds, N_draw=100, seed=0)
        lo, hi = bounds[0]
        assert lo <= out["sigma_p"] <= hi, \
            f"sigma_hat={out['sigma_p']:.4f} outside bounds [{lo}, {hi}]"


    def test_stage2_sigma_close_to_stage1(self, small_data):
        """Stage 2 is a refinement; sigma_2 should stay near sigma_1"""
        data, u_cost, _ = small_data
        out    = blp_estimator_fast(data, u_cost, N_draw=100, seed=0)
        sigma1 = out["res_stage1"].x[0]
        sigma2 = out["sigma_p"]
        assert abs(sigma2 - sigma1) < 0.5, \
            f"sigma_2={sigma2:.4f} too far from sigma_1={sigma1:.4f}"


    def test_xi_decomposition_identity(self, small_data):
        """xi must equal eta + intercept (the structural decomposition)"""
        data, u_cost, _ = small_data
        out = blp_estimator_fast(data, u_cost, N_draw=100, seed=0)
        np.testing.assert_allclose(
            out["xi"], out["eta"] + out["intercept"], atol=1e-8,
            err_msg="xi != eta + intercept: decomposition violated.")


# ===========================================================================
# 9. EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_single_market_no_crash(self):
        """Estimator must not crash with T=1 (single market)"""
        data, u_cost, _ = make_synthetic_data(T=1, J=8, Nt=2000, seed=77)
        blp = FastBLP(data, u_cost, iv_type=1)
        val = blp.obj(np.array([1.5]), blp.W1)
        assert np.isfinite(val)

    def test_single_product_no_crash(self):
        """Estimator must not crash with J=1 (single product per market)"""
        data, u_cost, _ = make_synthetic_data(T=20, J=1, Nt=2000, seed=88)
        blp = FastBLP(data, u_cost, iv_type=1)
        val = blp.obj(np.array([1.0]), blp.W1)
        assert np.isfinite(val)

    def test_very_small_sigma_finite(self, small_data):
        """At sigma=0.001 (above penalty threshold) obj() must be finite"""
        data, u_cost, _ = small_data
        blp = FastBLP(data, u_cost)
        assert np.isfinite(blp.obj(np.array([0.001]), blp.W1))

    def test_very_large_sigma_finite(self, small_data):
        """At sigma=5.0 obj() must be finite"""
        data, u_cost, _ = small_data
        blp = FastBLP(data, u_cost)
        assert np.isfinite(blp.obj(np.array([5.0]), blp.W1))

    def test_near_zero_shares_no_crash(self):
        """Near-zero shares must not cause log(0) or division-by-zero"""
        T, J = 5, 4
        rng  = default_rng(200)
        X    = np.stack([rng.uniform(1, 2, (T, J)),
                         rng.uniform(1, 2, (T, J))], axis=-1)
        s    = np.full((T, J + 1), 1.0 / (J + 1))
        s[:, 1] = 1e-10
        s[:, 0] = 1.0 - s[:, 1:].sum(axis=1)
        s    = np.abs(s) / s.sum(axis=1, keepdims=True)
        q    = np.maximum(np.round(s * 1000).astype(int), 0)
        data   = {"X": X, "q": q, "Nt": 1000}
        u_cost = rng.normal(0, 0.7, (T, J))
        blp    = FastBLP(data, u_cost)
        assert np.isfinite(blp.obj(np.array([1.0]), blp.W1))

    def test_reproducibility_same_seed(self, small_data):
        """Two FastBLP instances with the same seed must return identical g"""
        data, u_cost, _ = small_data
        blp1 = FastBLP(data, u_cost, seed=42)
        g1   = blp1.compute_g_xi(1.5, blp1.W1)
        blp2 = FastBLP(data, u_cost, seed=42)
        g2   = blp2.compute_g_xi(1.5, blp2.W1)
        np.testing.assert_array_equal(g1, g2,
            err_msg="Same seed must produce identical results")


# ===========================================================================
# 10. NUMERICAL PROPERTIES
# ===========================================================================

class TestNumericalProperties:

    def test_warm_start_not_slower_than_cold(self):
        """
        A warm-start from delta_true should converge in less iterations
        compared to starting from zero
        """
        T, J, sigma = 5, 5, 1.0
        rng         = default_rng(50)
        X           = np.stack([rng.uniform(1, 2, (T, J)),
                                 rng.uniform(1, 2, (T, J))], axis=-1)
        delta_true  = rng.uniform(-1, 1, (T, J))
        v           = rng.normal(0, 1, 200)
        s           = _exact_mc_shares(delta_true, X[:, :, 0], sigma, v)

        def count_iters(start, tol=1e-10):
            s_in = np.clip(s[:, 1:], 1e-300, 1.0)
            d    = start.copy()
            for it in range(5000):
                d_old = d.copy()
                maxd  = 0.0
                for t in range(T):
                    probs = _rc_logit_draw_probs(d[t], X[t, :, 0], sigma, v)
                    sp    = np.clip(probs.mean(axis=0)[1:], 1e-300, 1.0)
                    d[t]  = d[t] + np.log(s_in[t]) - np.log(sp)
                    maxd  = max(maxd, np.abs(d[t] - d_old[t]).max())
                if maxd < tol:
                    return it + 1
            return 5000

        iters_cold = count_iters(np.zeros((T, J)))
        iters_warm = count_iters(delta_true.copy())
        assert iters_warm <= iters_cold, \
            f"Warm start ({iters_warm} iters) should not be slower than cold ({iters_cold})."

    def test_2sls_coefficients_invariant_to_w_scaling(self, small_data):
        """
        Scaling w by a constant c should multiply beta_w by 1/c and leave
        beta_p unchanged. Because w enters only linearly through X_mat and the contraction mapping depends
        only on price and sigma
        """
        data, u_cost, _ = small_data
        blp1 = FastBLP(data, u_cost)
        blp1.compute_g_xi(1.5, blp1.W1)

        data2 = {**data, "X": data["X"].copy()}
        data2["X"][:, :, 1] *= 2.0  # double w only, NOT price
        blp2 = FastBLP(data2, u_cost)
        blp2.compute_g_xi(1.5, blp2.W1)

        assert abs(blp1.beta_last[1] - blp2.beta_last[1]) < 1e-5, \
            (f"beta_p should be invariant to w-scaling: "
             f"{blp1.beta_last[1]:.6f} vs {blp2.beta_last[1]:.6f}")
        assert abs(blp1.beta_last[2] - 2.0 * blp2.beta_last[2]) < 1e-5, \
            (f"beta_w should halve when w doubles: "
             f"{blp1.beta_last[2]:.6f} vs 2x{blp2.beta_last[2]:.6f}")

    def test_w1_condition_number_acceptable(self, small_data):
        """W1 = (Z'Z/N)^{-1} must be well-conditioned (kappa < 1e10)"""
        data, u_cost, _ = small_data
        blp  = FastBLP(data, u_cost)
        cond = np.linalg.cond(blp.W1)
        assert cond < 1e10, f"W1 condition number {cond:.2e} is too large."

    def test_contraction_output_is_finite(self, small_data):
        """Contraction mapping output must not contain NaN or Inf"""
        data, u_cost, _ = small_data
        blp   = FastBLP(data, u_cost)
        delta = blp_contraction_all_markets(
            blp.s_obs, blp.X_tensor, 1.5, blp.v_draws, blp.delta_last)
        assert np.all(np.isfinite(delta)), \
            "Contraction mapping returned non-finite values"

    def test_objective_convex_shaped_around_truth(self, small_data):
        """
        A grid of sigma values centred on the grounf truth should form a convex shape:
        obj(0.8) > obj(1.5) and obj(2.2) > obj(1.5)
        """
        data, u_cost, _ = small_data
        blp     = FastBLP(data, u_cost, N_draw=1500)
        v_truth = blp.obj(np.array([1.5]), blp.W1)
        v_below = blp.obj(np.array([0.8]), blp.W1)
        v_above = blp.obj(np.array([2.2]), blp.W1)
        assert v_below > v_truth, \
            f"obj(0.8)={v_below:.2f} should exceed obj(1.5)={v_truth:.2f}"
        assert v_above > v_truth, \
            f"obj(2.2)={v_above:.2f} should exceed obj(1.5)={v_truth:.2f}"

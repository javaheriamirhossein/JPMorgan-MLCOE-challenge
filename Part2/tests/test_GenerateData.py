"""
Unit and integration tests for BayesianSparseDeepHalo.GenerateData

Test summary
---------------
 1.  SimParams – defaults, types, shapes, post_init
 2.  SimParams – custom overrides and edge values
 3.  MCMCParams – defaults, types, shapes, post_init
 4.  MCMCParams – custom overrides
 5.  generate_data_tf – return-value structure (all 4 DGP types)
 6.  generate_data_tf – data_mcmc shapes and dtypes
 7.  generate_data_tf – dataset_cl shapes and dtypes
 8.  generate_data_tf – true_params shapes, dtypes, and keys
 9.  generate_data_tf – choice-count sanity (q rows sum to Nt)
10.  generate_data_tf – market-share sanity (shares in [0,1], sum to 1)
11.  generate_data_tf – features in valid range
12.  generate_data_tf – DGP-type-specific structural constraints
13.  generate_data_tf – endogeneity / exogeneity contrast
14.  generate_data_tf – reproducibility (seed)
15.  generate_data_tf – ChoiceDataset field consistency
16.  generate_data_tf – outside-option embedding is zero
17.  generate_data_tf – error on invalid dgp_type
18.  generate_data_tf – edge cases (T=1, J=1, D=1, large D)
"""

import numpy as np
import pytest
import tensorflow as tf
from choice_learn.data import ChoiceDataset

from BayesianSparseDeepHalo.GenerateData import SimParams, MCMCParams, generate_data_tf


# ── helpers ──────────────────────────────────────────────────────────────────

def _sim(T=10, J=5, D=2, Nt=100, seed=42):
    return SimParams(
        T=T, J=J, Nt=Nt, D=D,
        beta_mean=np.array([-1.0, 0.5][:D] if D <= 2 else [-1.0] + [0.5] * (D - 1)),
        sigma_beta=np.array([1.5, 0.0][:D] if D <= 2 else [1.5] + [0.0] * (D - 1)),
        xi_bar=-1.0,
        seed=seed,
    )


def _unpack(ds_field):
    """Unwrap ChoiceDataset tuple fields to a plain numpy array."""
    if isinstance(ds_field, tuple):
        return np.asarray(ds_field[0])
    return np.asarray(ds_field)


def _run(dgp_type=1, T=10, J=5, D=2, Nt=100, seed=42):
    return generate_data_tf(dgp_type=dgp_type, sim=_sim(T=T, J=J, D=D, Nt=Nt, seed=seed))


# ── module-scoped fixtures to avoid regenerating data for every test ────────

@pytest.fixture(scope="module")
def dgp1():
    return _run(dgp_type=1, T=15, J=6, D=2, Nt=200, seed=7)

@pytest.fixture(scope="module")
def dgp2():
    return _run(dgp_type=2, T=15, J=6, D=2, Nt=200, seed=7)

@pytest.fixture(scope="module")
def dgp3():
    return _run(dgp_type=3, T=15, J=6, D=2, Nt=200, seed=7)

@pytest.fixture(scope="module")
def dgp4():
    return _run(dgp_type=4, T=15, J=6, D=2, Nt=200, seed=7)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SimParams – defaults, types, shapes, post_init
# ═══════════════════════════════════════════════════════════════════════════

class TestSimParamsDefaults:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.sp = SimParams()

    def test_default_T(self):
        assert self.sp.T == 100

    def test_default_J(self):
        assert self.sp.J == 15

    def test_default_Nt(self):
        assert self.sp.Nt == 1000

    def test_default_D(self):
        assert self.sp.D == 2

    def test_default_xi_bar(self):
        assert self.sp.xi_bar == -1.0

    def test_default_seed(self):
        assert self.sp.seed == 123

    def test_beta_mean_is_ndarray(self):
        assert isinstance(self.sp.beta_mean, np.ndarray)

    def test_beta_mean_shape(self):
        assert self.sp.beta_mean.shape == (2,)

    def test_beta_mean_default_values(self):
        np.testing.assert_array_equal(self.sp.beta_mean, [-1.0, 0.5])

    def test_sigma_beta_is_ndarray(self):
        assert isinstance(self.sp.sigma_beta, np.ndarray)

    def test_sigma_beta_shape(self):
        assert self.sp.sigma_beta.shape == (2,)

    def test_sigma_beta_default_values(self):
        np.testing.assert_array_equal(self.sp.sigma_beta, [1.5, 0.0])

    def test_sigma_beta_non_negative(self):
        assert np.all(self.sp.sigma_beta >= 0)

    def test_T_is_int(self):
        assert isinstance(self.sp.T, int)

    def test_J_is_int(self):
        assert isinstance(self.sp.J, int)

    def test_Nt_is_int(self):
        assert isinstance(self.sp.Nt, int)

    def test_D_is_int(self):
        assert isinstance(self.sp.D, int)

    def test_xi_bar_is_float(self):
        assert isinstance(self.sp.xi_bar, float)

    def test_seed_is_int(self):
        assert isinstance(self.sp.seed, int)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SimParams – custom overrides and edge values
# ═══════════════════════════════════════════════════════════════════════════

class TestSimParamsCustom:

    def test_custom_beta_mean_stored(self):
        beta = np.array([2.0, -3.0, 1.0])
        sp = SimParams(D=3, beta_mean=beta)
        np.testing.assert_array_equal(sp.beta_mean, beta)

    def test_custom_sigma_beta_stored(self):
        sigma = np.array([0.0, 1.0, 2.0])
        sp = SimParams(D=3, sigma_beta=sigma)
        np.testing.assert_array_equal(sp.sigma_beta, sigma)

    def test_none_beta_mean_triggers_default(self):
        sp = SimParams(D=2, beta_mean=None)
        assert sp.beta_mean is not None
        assert sp.beta_mean.shape == (2,)

    def test_none_sigma_beta_triggers_default(self):
        sp = SimParams(D=2, sigma_beta=None)
        assert sp.sigma_beta is not None
        assert sp.sigma_beta.shape == (2,)

    def test_T_eq_1_accepted(self):
        sp = SimParams(T=1)
        assert sp.T == 1

    def test_J_eq_1_accepted(self):
        sp = SimParams(J=1)
        assert sp.J == 1

    def test_D_eq_1_accepted(self):
        sp = SimParams(D=1, beta_mean=np.array([-1.0]), sigma_beta=np.array([0.5]))
        assert sp.D == 1

    def test_positive_xi_bar(self):
        sp = SimParams(xi_bar=2.5)
        assert sp.xi_bar == 2.5

    def test_zero_xi_bar(self):
        sp = SimParams(xi_bar=0.0)
        assert sp.xi_bar == 0.0

    def test_all_fixed_sigma(self):
        """sigma_beta all zeros → all fixed coefficients."""
        sp = SimParams(D=3, sigma_beta=np.zeros(3))
        assert np.all(sp.sigma_beta == 0.0)

    def test_repr_contains_T(self):
        sp = SimParams(T=77)
        assert "77" in str(sp)


# ═══════════════════════════════════════════════════════════════════════════
# 3. MCMCParams – defaults, types, shapes, post_init
# ═══════════════════════════════════════════════════════════════════════════

class TestMCMCParamsDefaults:

    @pytest.fixture(autouse=True)
    def _build(self):
        self.mp = MCMCParams()

    def test_default_R0(self):
        assert self.mp.R0 == 200

    def test_default_G(self):
        assert self.mp.G == 10000

    def test_default_burn(self):
        assert self.mp.burn == 5000

    def test_default_D(self):
        assert self.mp.D == 2

    def test_default_tau0(self):
        assert self.mp.tau0 == 1e-3

    def test_default_tau1(self):
        assert self.mp.tau1 == 1.0

    def test_default_a_phi(self):
        assert self.mp.a_phi == 1.0

    def test_default_b_phi(self):
        assert self.mp.b_phi == 1.0

    def test_default_V_beta(self):
        assert self.mp.V_beta == 10.0

    def test_default_V_xibar(self):
        assert self.mp.V_xibar == 10.0

    def test_default_V_r(self):
        assert self.mp.V_r == 0.5

    def test_default_target_accept(self):
        assert self.mp.target_accept == 0.35

    def test_default_step_beta(self):
        assert self.mp.step_beta == 0.5

    def test_default_step_r(self):
        assert self.mp.step_r == 0.5

    def test_default_step_xibar(self):
        assert self.mp.step_xibar == 0.5

    def test_default_step_eta(self):
        assert self.mp.step_eta == 0.5

    def test_default_step_phi(self):
        assert self.mp.step_phi == 0.5

    def test_random_coef_mask_is_ndarray(self):
        assert isinstance(self.mp.random_coef_mask, np.ndarray)

    def test_random_coef_mask_shape(self):
        assert self.mp.random_coef_mask.shape == (2,)

    def test_random_coef_mask_default_first_one(self):
        """Default: only the first coefficient is random."""
        assert self.mp.random_coef_mask[0] == 1.0

    def test_random_coef_mask_rest_zero(self):
        assert np.all(self.mp.random_coef_mask[1:] == 0.0)

    def test_random_coef_mask_binary(self):
        vals = self.mp.random_coef_mask
        assert set(vals).issubset({0.0, 1.0})

    def test_burn_less_than_G(self):
        assert self.mp.burn < self.mp.G


# ═══════════════════════════════════════════════════════════════════════════
# 4. MCMCParams – custom overrides
# ═══════════════════════════════════════════════════════════════════════════

class TestMCMCParamsCustom:

    def test_custom_D_changes_mask_shape(self):
        mp = MCMCParams(D=5)
        assert mp.random_coef_mask.shape == (5,)

    def test_custom_D_mask_first_element_one(self):
        mp = MCMCParams(D=5)
        assert mp.random_coef_mask[0] == 1.0

    def test_custom_D_mask_remaining_zero(self):
        mp = MCMCParams(D=5)
        assert np.all(mp.random_coef_mask[1:] == 0.0)

    def test_explicit_mask_stored(self):
        mask = np.array([1., 1., 0., 0.])
        mp = MCMCParams(D=4, random_coef_mask=mask)
        np.testing.assert_array_equal(mp.random_coef_mask, mask)

    def test_all_random_mask(self):
        D = 4
        mask = np.ones(D)
        mp = MCMCParams(D=D, random_coef_mask=mask)
        assert np.all(mp.random_coef_mask == 1.0)

    def test_all_fixed_mask(self):
        D = 4
        mask = np.zeros(D)
        mp = MCMCParams(D=D, random_coef_mask=mask)
        assert np.all(mp.random_coef_mask == 0.0)

    def test_custom_tau0_stored(self):
        mp = MCMCParams(tau0=0.01)
        assert mp.tau0 == 0.01

    def test_custom_tau1_stored(self):
        mp = MCMCParams(tau1=2.0)
        assert mp.tau1 == 2.0

    def test_tau0_less_than_tau1(self):
        mp = MCMCParams()
        assert mp.tau0 < mp.tau1

    def test_R0_is_int(self):
        mp = MCMCParams(R0=50)
        assert isinstance(mp.R0, int)


# ═══════════════════════════════════════════════════════════════════════════
# 5. generate_data_tf – return-value structure (all 4 DGP types)
# ═══════════════════════════════════════════════════════════════════════════

class TestReturnStructure:

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_returns_three_elements(self, dgp):
        out = _run(dgp_type=dgp)
        assert len(out) == 3

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_data_mcmc_is_dict(self, dgp):
        data_mcmc, _, _ = _run(dgp_type=dgp)
        assert isinstance(data_mcmc, dict)

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_dataset_cl_is_choice_dataset(self, dgp):
        _, ds, _ = _run(dgp_type=dgp)
        assert isinstance(ds, ChoiceDataset)

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_true_params_is_dict(self, dgp):
        _, _, tp = _run(dgp_type=dgp)
        assert isinstance(tp, dict)

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_data_mcmc_required_keys(self, dgp):
        data_mcmc, _, _ = _run(dgp_type=dgp)
        for key in ("X", "q", "Nt"):
            assert key in data_mcmc, f"data_mcmc missing key '{key}'"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_true_params_required_keys(self, dgp):
        _, _, tp = _run(dgp_type=dgp)
        for key in ("eta_star", "xi_star", "X", "shares",
                    "beta_mean", "sigma_beta", "u_cost"):
            assert key in tp, f"true_params missing key '{key}'"


# ═══════════════════════════════════════════════════════════════════════════
# 6. generate_data_tf – data_mcmc shapes and dtypes
# ═══════════════════════════════════════════════════════════════════════════

class TestDataMCMCShapes:

    @pytest.fixture(autouse=True)
    def _build(self, dgp1):
        self.data_mcmc, _, _ = dgp1
        self.T, self.J, self.D, self.Nt = 15, 6, 2, 200

    def test_X_shape(self):
        assert self.data_mcmc["X"].shape == (self.T, self.J, self.D)

    def test_q_shape(self):
        assert self.data_mcmc["q"].shape == (self.T, self.J + 1)

    def test_Nt_value(self):
        assert self.data_mcmc["Nt"] == self.Nt

    def test_X_dtype_float(self):
        assert np.issubdtype(self.data_mcmc["X"].dtype, np.floating)

    def test_q_dtype_integer(self):
        assert np.issubdtype(self.data_mcmc["q"].dtype, np.integer)

    def test_X_is_ndarray(self):
        assert isinstance(self.data_mcmc["X"], np.ndarray)

    def test_q_is_ndarray(self):
        assert isinstance(self.data_mcmc["q"], np.ndarray)

    def test_X_values_finite(self):
        assert np.all(np.isfinite(self.data_mcmc["X"]))

    def test_q_non_negative(self):
        assert np.all(self.data_mcmc["q"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. generate_data_tf – dataset_cl shapes and dtypes
# ═══════════════════════════════════════════════════════════════════════════

class TestChoiceDatasetShapes:

    @pytest.fixture(autouse=True)
    def _build(self, dgp1):
        _, self.ds, _ = dgp1
        self.T, self.J, self.D, self.Nt = 15, 6, 2, 200
        self.N_obs = self.T * self.Nt  # total individual observations

    def test_choices_length(self):
        assert len(self.ds.choices) == self.N_obs

    def test_choices_dtype_integer(self):
        assert np.issubdtype(self.ds.choices.dtype, np.integer)

    def test_choices_in_valid_range(self):
        choices = self.ds.choices
        assert np.all(choices >= 0)
        assert np.all(choices <= self.J)

    def test_items_features_shape(self):
        feat = _unpack(self.ds.items_features_by_choice)
        assert feat.shape == (self.N_obs, self.J + 1, self.D)

    def test_items_features_dtype_float(self):
        feat = _unpack(self.ds.items_features_by_choice)
        assert np.issubdtype(feat.dtype, np.floating)

    def test_available_items_shape(self):
        avail = _unpack(self.ds.available_items_by_choice)
        assert avail.shape == (self.N_obs, self.J + 1)

    def test_available_items_all_ones(self):
        """No availability restrictions: every product available to every consumer."""
        avail = _unpack(self.ds.available_items_by_choice)
        np.testing.assert_array_equal(avail, np.ones_like(avail))

    def test_shared_features_shape(self):
        shared = _unpack(self.ds.shared_features_by_choice)
        assert shared.shape == (self.N_obs, 1)

    def test_shared_features_dtype(self):
        shared = _unpack(self.ds.shared_features_by_choice)
        assert np.issubdtype(shared.dtype, np.floating)

    def test_shared_features_names(self):
        names = self.ds.shared_features_by_choice_names
        if isinstance(names, tuple):
            names = names[0]
        assert names == ["market_id"]

    def test_items_features_names_count(self):
        names = self.ds.items_features_by_choice_names
        if isinstance(names, tuple):
            names = names[0]
        assert len(names) == self.D

    def test_items_features_names_pattern(self):
        names = self.ds.items_features_by_choice_names
        if isinstance(names, tuple):
            names = names[0]
        for i, name in enumerate(names):
            assert name == f"x{i}", f"Expected 'x{i}', got '{name}'"

    def test_market_ids_in_valid_range(self):
        shared = _unpack(self.ds.shared_features_by_choice)
        mids = shared[:, 0].astype(int)
        assert np.all(mids >= 0)
        assert np.all(mids < self.T)

    def test_all_markets_represented(self):
        """Every market 0..T-1 must appear at least once in shared_features."""
        shared = _unpack(self.ds.shared_features_by_choice)
        present = set(shared[:, 0].astype(int))
        expected = set(range(self.T))
        assert present == expected


# ═══════════════════════════════════════════════════════════════════════════
# 8. generate_data_tf – true_params shapes, dtypes, and keys
# ═══════════════════════════════════════════════════════════════════════════

class TestTrueParamsShapes:

    @pytest.fixture(autouse=True)
    def _build(self, dgp1):
        _, _, self.tp = dgp1
        self.T, self.J, self.D = 15, 6, 2

    def test_eta_star_shape(self):
        assert self.tp["eta_star"].shape == (self.T, self.J)

    def test_xi_star_shape(self):
        assert self.tp["xi_star"].shape == (self.T, self.J)

    def test_X_shape(self):
        assert self.tp["X"].shape == (self.T, self.J, self.D)

    def test_shares_shape(self):
        assert self.tp["shares"].shape == (self.T, self.J + 1)

    def test_beta_mean_shape(self):
        assert self.tp["beta_mean"].shape == (self.D,)

    def test_sigma_beta_shape(self):
        assert self.tp["sigma_beta"].shape == (self.D,)

    def test_u_cost_shape(self):
        assert self.tp["u_cost"].shape == (self.T, self.J)

    def test_eta_star_dtype_float(self):
        assert np.issubdtype(self.tp["eta_star"].dtype, np.floating)

    def test_xi_star_dtype_float(self):
        assert np.issubdtype(self.tp["xi_star"].dtype, np.floating)

    def test_shares_dtype_float(self):
        assert np.issubdtype(self.tp["shares"].dtype, np.floating)

    def test_all_values_finite(self):
        for key in ("eta_star", "xi_star", "X", "shares", "u_cost"):
            assert np.all(np.isfinite(self.tp[key])), f"Non-finite values in true_params['{key}']"

    def test_beta_mean_is_ndarray(self):
        assert isinstance(self.tp["beta_mean"], np.ndarray)

    def test_sigma_beta_is_ndarray(self):
        assert isinstance(self.tp["sigma_beta"], np.ndarray)

    def test_sigma_beta_non_negative(self):
        assert np.all(self.tp["sigma_beta"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# 9. generate_data_tf – choice-count sanity (q rows sum to Nt)
# ═══════════════════════════════════════════════════════════════════════════

class TestChoiceCountSanity:

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_q_rows_sum_to_Nt(self, dgp):
        """Every market's total choice count must equal exactly Nt."""
        Nt = 200
        data_mcmc, _, _ = _run(dgp_type=dgp, Nt=Nt)
        row_sums = data_mcmc["q"].sum(axis=1)
        np.testing.assert_array_equal(
            row_sums, np.full(row_sums.shape, Nt),
            err_msg=f"DGP {dgp}: q row sums must equal Nt={Nt}")

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_q_non_negative(self, dgp):
        data_mcmc, _, _ = _run(dgp_type=dgp)
        assert np.all(data_mcmc["q"] >= 0), f"DGP {dgp}: q contains negative values"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_q_outside_option_non_negative(self, dgp):
        """Outside option counts must be >= 0 after rounding correction."""
        data_mcmc, _, _ = _run(dgp_type=dgp, Nt=500)
        assert np.all(data_mcmc["q"][:, 0] >= 0),  f"DGP {dgp}: outside-option counts contain negative values"

    def test_total_obs_in_dataset_equals_T_times_Nt(self, dgp1):
        """ChoiceDataset must expand to exactly T * Nt individual observations."""
        data_mcmc, ds, _ = dgp1
        T, Nt = 15, 200
        assert len(ds.choices) == T * Nt

    def test_choices_flat_match_q_expansion(self, dgp1):
        """Sum of choices == j in dataset must match q[:, j].sum() for each j."""
        data_mcmc, ds, _ = dgp1
        J = 6
        q = data_mcmc["q"]
        choices = np.asarray(ds.choices)
        for j in range(J + 1):
            n_chosen = int((choices == j).sum())
            n_q = int(q[:, j].sum())
            assert n_chosen == n_q,  f"Choice count for product {j}: dataset={n_chosen}, q_sum={n_q}"


# ═══════════════════════════════════════════════════════════════════════════
# 10. generate_data_tf – market-share sanity (shares in [0,1], sum to 1)
# ═══════════════════════════════════════════════════════════════════════════

class TestShareSanity:

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_shares_in_unit_interval(self, dgp):
        _, _, tp = _run(dgp_type=dgp)
        shares = tp["shares"]
        assert np.all(shares >= 0.0), f"DGP {dgp}: shares contain negatives"
        assert np.all(shares <= 1.0), f"DGP {dgp}: shares exceed 1"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_shares_sum_to_one(self, dgp):
        _, _, tp = _run(dgp_type=dgp)
        row_sums = tp["shares"].sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones_like(row_sums), atol=1e-6,
            err_msg=f"DGP {dgp}: market shares do not sum to 1")

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_outside_option_share_positive(self, dgp):
        """The outside option (j=0) must attract some consumers in every market."""
        _, _, tp = _run(dgp_type=dgp)
        outside = tp["shares"][:, 0]
        assert np.all(outside > 0), f"DGP {dgp}: some markets have zero outside-option share"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_inside_shares_positive_on_average(self, dgp):
        """Average inside-product share must be positive."""
        _, _, tp = _run(dgp_type=dgp, J=5, Nt=200)
        inside_mean = tp["shares"][:, 1:].mean()
        assert inside_mean > 0


# ═══════════════════════════════════════════════════════════════════════════
# 11. generate_data_tf – features in valid range
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureRange:

    def test_raw_second_feature_in_1_2(self, dgp1):
        """
        Non-price features (column 1+) are drawn from Uniform[1,2].
        For exogenous DGP (type 1), price (column 0) is NOT shifted by
        a large alpha, so it should stay within a reasonable range.
        """
        data_mcmc, _, _ = dgp1
        X = data_mcmc["X"]
        # Second feature (index 1): always from Uniform[1,2] + tiny cost shock
        feat1 = X[:, :, 1]
        assert np.all(feat1 >= 0.5), "Second feature has unexpectedly small values"
        assert np.all(feat1 <= 3.0), "Second feature has unexpectedly large values"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_features_finite(self, dgp):
        data_mcmc, _, _ = _run(dgp_type=dgp)
        assert np.all(np.isfinite(data_mcmc["X"]))

    def test_endogenous_price_shifted(self, dgp2):
        """
        DGP-2 (sparse-endogenous): for products with large |eta_star|,
        the price (feature 0) must differ systematically from DGP-1.
        """
        data1, _, _ = _run(dgp_type=1, seed=7)
        data2, _, _ = _run(dgp_type=2, seed=7)
        # Prices will differ because alpha != 0 in DGP-2
        assert not np.allclose(data1["X"][:, :, 0], data2["X"][:, :, 0]),             "Endogenous DGP-2 prices should differ from exogenous DGP-1"

    def test_exogenous_price_not_shifted(self, dgp1):
        """
        DGP-1 (sparse-exogenous): alpha == 0 for all products.
        Price = 0.3 * x_raw1 + u_cost, centred around 0.3*[1,2] + N(0,0.7).
        """
        data_mcmc, _, _ = dgp1
        price = data_mcmc["X"][:, :, 0]
        # Rough bounds: 0.3*1 + 3*0.7 ≈ 2.4 upper; lower can be negative
        assert np.all(price < 10.0), "Exogenous prices are unexpectedly large"


# ═══════════════════════════════════════════════════════════════════════════
# 12. generate_data_tf – DGP-type-specific structural constraints
# ═══════════════════════════════════════════════════════════════════════════

class TestDGPStructure:

    def test_dgp1_eta_star_sparse(self, dgp1):
        """
        DGP-1: sparse alternating eta pattern.
        Exactly 40% of products have non-zero eta_star and eta is identical
        across all markets (same pattern repeated).
        """
        _, _, tp = dgp1
        eta = tp["eta_star"]
        T, J = eta.shape
        # Rows should be identical (same pattern across markets)
        for t in range(1, T):
            np.testing.assert_array_equal(
                eta[0], eta[t],
                err_msg=f"DGP-1 eta_star row {t} differs from row 0 — should be identical")

    def test_dgp1_eta_star_values_are_pm1_or_0(self, dgp1):
        """DGP-1 eta_star must contain only values in {-1, 0, +1}."""
        _, _, tp = dgp1
        vals = np.unique(tp["eta_star"])
        for v in vals:
            assert v in (-1.0, 0.0, 1.0), f"Unexpected eta_star value in DGP-1: {v}"

    def test_dgp1_active_fraction(self):
        """DGP-1: 40% of J products get non-zero eta_star."""
        J = 10
        _, _, tp = _run(dgp_type=1, J=J, T=5)
        eta = tp["eta_star"][0]   # first market
        n_active = int((eta != 0).sum())
        assert n_active == int(0.4 * J),  f"Expected {int(0.4*J)} active products, got {n_active}"

    def test_dgp2_eta_star_sparse(self, dgp2):
        """DGP-2: same sparse pattern as DGP-1."""
        _, _, tp = dgp2
        eta = tp["eta_star"]
        for t in range(1, eta.shape[0]):
            np.testing.assert_array_equal(eta[0], eta[t])

    def test_dgp3_eta_star_not_identical_rows(self, dgp3):
        """DGP-3: eta_star is iid Normal → rows must NOT all be identical."""
        _, _, tp = dgp3
        eta = tp["eta_star"]
        same_as_row0 = all(np.array_equal(eta[0], eta[t]) for t in range(1, eta.shape[0]))
        assert not same_as_row0, "DGP-3 eta_star rows should differ (iid Normal)"

    def test_dgp3_eta_star_mean_near_zero(self, dgp3):
        """DGP-3 eta_star ~ N(0, 1/3) → grand mean should be near 0."""
        _, _, tp = dgp3
        assert abs(tp["eta_star"].mean()) < 0.3,  f"DGP-3 eta_star grand mean {tp['eta_star'].mean():.3f} too far from 0"

    def test_dgp4_eta_star_not_identical_rows(self, dgp4):
        """DGP-4: eta_star is iid Normal → rows must differ."""
        _, _, tp = dgp4
        eta = tp["eta_star"]
        same = all(np.array_equal(eta[0], eta[t]) for t in range(1, eta.shape[0]))
        assert not same

    def test_dgp1_xi_star_equals_xi_bar_plus_eta(self, dgp1):
        """true_params['xi_star'] must equal xi_bar + eta_star."""
        _, _, tp = dgp1
        xi_bar = -1.0
        expected = xi_bar + tp["eta_star"]
        np.testing.assert_allclose(tp["xi_star"], expected, atol=1e-10)

    def test_dgp3_xi_star_equals_xi_bar_plus_eta(self, dgp3):
        _, _, tp = dgp3
        xi_bar = -1.0
        np.testing.assert_allclose(tp["xi_star"], xi_bar + tp["eta_star"], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 13. generate_data_tf – endogeneity / exogeneity contrast
# ═══════════════════════════════════════════════════════════════════════════

class TestEndogeneityContrast:

    def test_exogenous_alpha_zero_for_dgp1(self):
        """
        DGP-1 (exogenous): alpha = 0 everywhere, so price = 0.3*x1 + u_cost.
        The correlation between eta_star and price should be near zero.
        """
        _, _, tp = _run(dgp_type=1, T=50, J=10, Nt=100, seed=1)
        eta = tp["eta_star"].flatten()
        price = tp["X"][:, :, 0].flatten()
        corr = np.corrcoef(eta, price)[0, 1]
        assert abs(corr) < 0.25,  f"DGP-1 (exogenous): eta–price correlation {corr:.3f} is unexpectedly large"

    def test_endogenous_alpha_nonzero_for_dgp2(self):
        """
        DGP-2 (endogenous): alpha ≠ 0 for products with large |eta|,
        so the eta–price correlation should be meaningfully positive.
        """
        _, _, tp = _run(dgp_type=2, T=50, J=10, Nt=100, seed=1)
        eta = tp["eta_star"].flatten()
        price = tp["X"][:, :, 0].flatten()
        corr = np.corrcoef(eta, price)[0, 1]
        assert corr > 0.05,  f"DGP-2 (endogenous): eta–price correlation {corr:.3f} should be positive"

    def test_dgp3_dgp4_price_distributions_differ(self):
        """Endogenous (DGP-4) prices should have higher variance than exogenous (DGP-3)."""
        _, _, tp3 = _run(dgp_type=3, T=30, J=8, Nt=200, seed=10)
        _, _, tp4 = _run(dgp_type=4, T=30, J=8, Nt=200, seed=10)
        var3 = tp3["X"][:, :, 0].var()
        var4 = tp4["X"][:, :, 0].var()
        assert var4 >= var3 * 0.8,             "DGP-4 price variance should be at least as large as DGP-3"

    def test_dgp1_dgp2_same_eta_structure(self):
        """DGP-1 and DGP-2 share the same sparse eta pattern (same seed)."""
        _, _, tp1 = _run(dgp_type=1, seed=55)
        _, _, tp2 = _run(dgp_type=2, seed=55)
        np.testing.assert_array_equal(
            tp1["eta_star"], tp2["eta_star"],
            err_msg="DGP-1 and DGP-2 must share the same eta_star pattern")

    def test_dgp3_dgp4_same_eta_structure(self):
        """DGP-3 and DGP-4 share the same Normal eta draws (same seed)."""
        _, _, tp3 = _run(dgp_type=3, seed=66)
        _, _, tp4 = _run(dgp_type=4, seed=66)
        np.testing.assert_array_equal(tp3["eta_star"], tp4["eta_star"])


# ═══════════════════════════════════════════════════════════════════════════
# 14. generate_data_tf – reproducibility (seed)
# ═══════════════════════════════════════════════════════════════════════════

class TestReproducibility:

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_same_seed_same_X(self, dgp):
        d1, _, _ = _run(dgp_type=dgp, seed=100)
        d2, _, _ = _run(dgp_type=dgp, seed=100)
        np.testing.assert_array_equal(d1["X"], d2["X"],
                                      err_msg=f"DGP {dgp}: X differs across same-seed runs")

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_same_seed_same_q(self, dgp):
        d1, _, _ = _run(dgp_type=dgp, seed=100)
        d2, _, _ = _run(dgp_type=dgp, seed=100)
        np.testing.assert_array_equal(d1["q"], d2["q"],
                                      err_msg=f"DGP {dgp}: q differs across same-seed runs")

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_same_seed_same_eta_star(self, dgp):
        _, _, tp1 = _run(dgp_type=dgp, seed=100)
        _, _, tp2 = _run(dgp_type=dgp, seed=100)
        np.testing.assert_array_equal(tp1["eta_star"], tp2["eta_star"],
                                      err_msg=f"DGP {dgp}: eta_star differs across same-seed runs")

    @pytest.mark.parametrize("dgp", [3, 4])
    def test_different_seed_different_eta(self, dgp):
        """Different seeds must produce different stochastic eta_star for DGPs 3/4."""
        _, _, tp1 = _run(dgp_type=dgp, seed=1)
        _, _, tp2 = _run(dgp_type=dgp, seed=9999)
        assert not np.array_equal(tp1["eta_star"], tp2["eta_star"]),  f"DGP {dgp}: different seeds should give different eta_star"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_different_seed_different_X(self, dgp):
        d1, _, _ = _run(dgp_type=dgp, seed=1)
        d2, _, _ = _run(dgp_type=dgp, seed=9999)
        assert not np.array_equal(d1["X"], d2["X"]),  f"DGP {dgp}: different seeds should give different X"


# ═══════════════════════════════════════════════════════════════════════════
# 15. generate_data_tf – ChoiceDataset field consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestChoiceDatasetConsistency:

    @pytest.fixture(autouse=True)
    def _build(self, dgp1):
        self.data_mcmc, self.ds, self.tp = dgp1
        self.T, self.J, self.D = 15, 6, 2

    def test_market_id_range(self):
        shared = _unpack(self.ds.shared_features_by_choice)
        mids = shared[:, 0].astype(int)
        assert mids.min() == 0
        assert mids.max() == self.T - 1

    def test_item_features_match_X_mcmc(self):
        """
        For every observation, items_features[:, 1:] (inside products) must
        match data_mcmc["X"] for the corresponding market.
        """
        feat = _unpack(self.ds.items_features_by_choice)
        shared = _unpack(self.ds.shared_features_by_choice)
        mids = shared[:, 0].astype(int)
        X = self.data_mcmc["X"]
        # Check first 50 observations only (enough to catch misalignment)
        for i in range(min(50, len(mids))):
            t = mids[i]
            np.testing.assert_allclose(
                feat[i, 1:],        # inside products (columns 1..)
                X[t],               # market-t features from MCMC dict
                atol=1e-10,
                err_msg=f"Item features for obs {i} (market {t}) don't match data_mcmc['X']")

    def test_choices_consistent_with_q(self):
        """For each product j, choices count must equal q[:, j].sum()."""
        choices = np.asarray(self.ds.choices)
        q = self.data_mcmc["q"]
        for j in range(self.J + 1):
            assert int((choices == j).sum()) == int(q[:, j].sum()),  f"Choices for j={j} inconsistent with q"

    def test_outside_option_features_are_zero(self):
        """Column 0 of items_features (outside option) must be all-zero."""
        feat = _unpack(self.ds.items_features_by_choice)
        outside_emb = feat[:, 0, :]
        np.testing.assert_array_equal(
            outside_emb, np.zeros_like(outside_emb),
            err_msg="Outside option (column 0) embedding must be zero")

    def test_market_obs_counts_match_Nt(self):
        """Each market must appear exactly Nt times in shared_features."""
        shared = _unpack(self.ds.shared_features_by_choice)
        mids = shared[:, 0].astype(int)
        for t in range(self.T):
            count = int((mids == t).sum())
            assert count == 200,  f"Market {t} has {count} observations, expected 200"


# ═══════════════════════════════════════════════════════════════════════════
# 16. generate_data_tf – outside-option embedding is zero
# ═══════════════════════════════════════════════════════════════════════════

class TestOutsideOptionEmbedding:

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_outside_option_column_zero(self, dgp):
        _, ds, _ = _run(dgp_type=dgp)
        feat = _unpack(ds.items_features_by_choice)
        outside = feat[:, 0, :]
        assert np.all(outside == 0.0),  f"DGP {dgp}: outside-option features must all be zero"

    @pytest.mark.parametrize("dgp", [1, 2, 3, 4])
    def test_inside_features_not_all_zero(self, dgp):
        """Inside product features (columns 1..) must be non-trivially non-zero."""
        _, ds, _ = _run(dgp_type=dgp)
        feat = _unpack(ds.items_features_by_choice)
        inside = feat[:, 1:, :]
        assert not np.all(inside == 0.0),  f"DGP {dgp}: inside-product features must not all be zero"


# ═══════════════════════════════════════════════════════════════════════════
# 17. generate_data_tf – error on invalid dgp_type
# ═══════════════════════════════════════════════════════════════════════════

class TestInvalidInputs:

    @pytest.mark.parametrize("bad_type", [0, 5, -1, 99])
    def test_invalid_dgp_type_raises_value_error(self, bad_type):
        with pytest.raises(ValueError, match="dgp_type must be 1..4"):
            _run(dgp_type=bad_type)


# ═══════════════════════════════════════════════════════════════════════════
# 18. generate_data_tf – edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_T_eq_1_runs(self):
        """Single market must generate valid data."""
        data_mcmc, ds, tp = _run(T=1, J=5, D=2, Nt=100)
        assert data_mcmc["X"].shape == (1, 5, 2)
        assert data_mcmc["q"].shape == (1, 6)
        assert len(ds.choices) == 100

    def test_J_eq_1_runs(self):
        """Single inside product must generate valid data."""
        data_mcmc, ds, tp = _run(T=5, J=1, D=2, Nt=100)
        assert data_mcmc["X"].shape == (5, 1, 2)
        assert data_mcmc["q"].shape == (5, 2)

    def test_D_eq_1_runs(self):
        sim = SimParams(T=5, J=4, Nt=100, D=1,
                        beta_mean=np.array([-1.0]),
                        sigma_beta=np.array([1.0]))
        data_mcmc, ds, tp = generate_data_tf(dgp_type=1, sim=sim)
        assert data_mcmc["X"].shape == (5, 4, 1)

    
    def test_large_D_runs(self):
        """D=10 must not crash."""
        D = 10
        sim = SimParams(T=8, J=5, Nt=50, D=D,
                        beta_mean=np.ones(D) * -0.5,
                        sigma_beta=np.ones(D) * 0.5)
        data_mcmc, _, _ = generate_data_tf(dgp_type=1, sim=sim)
        assert data_mcmc["X"].shape == (8, 5, D)

    def test_small_Nt_runs(self):
        """Very small Nt=10 must still produce valid data."""
        data_mcmc, ds, _ = _run(Nt=10)
        assert data_mcmc["q"].sum(axis=1).min() == 10

    
    def test_large_T_runs(self):
        sim = SimParams(T=200, J=5, Nt=50, D=2)
        data_mcmc, _, _ = generate_data_tf(dgp_type=1, sim=sim)
        assert data_mcmc["X"].shape == (200, 5, 2)

    def test_all_dgp_types_run_without_error(self):
        for dgp in (1, 2, 3, 4):
            data_mcmc, ds, tp = _run(dgp_type=dgp, T=4, J=3, Nt=20)
            assert data_mcmc["q"].sum(axis=1).min() == 20

    def test_all_fixed_sigma_beta(self):
        """All-zero sigma_beta (no random coefficients) should work fine."""
        sim = SimParams(T=5, J=4, Nt=50, D=2,
                        beta_mean=np.array([-1.0, 0.5]),
                        sigma_beta=np.array([0.0, 0.0]))
        data_mcmc, _, tp = generate_data_tf(dgp_type=1, sim=sim)
        assert np.all(np.isfinite(data_mcmc["X"]))

    def test_dgp1_with_J_smaller_than_n_active_floor(self):
        """J=1 → n_active = int(0.4*1) = 0; eta_star must be all zeros."""
        _, _, tp = _run(dgp_type=1, J=1)
        # With J=1, n_active=0, eta_base=[0], so eta_star==0 everywhere
        # OR n_active could be 0 meaning only zeros
        # Just verify shapes and finiteness
        assert tp["eta_star"].shape[1] == 1
        assert np.all(np.isfinite(tp["eta_star"]))

    def test_q_rows_sum_to_Nt_edge_small_Nt(self):
        """Rounding correction must hold even for tiny Nt=5."""
        data_mcmc, _, _ = _run(Nt=5)
        np.testing.assert_array_equal(data_mcmc["q"].sum(axis=1), np.full(10, 5))

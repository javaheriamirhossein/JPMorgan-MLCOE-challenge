"""
Berry-Levinsohn-Pakes (BLP) Estimator
================================================================
Implements the two-step efficient GMM estimator for the random-coefficient
logit demand model (BLP).

Key idea — "Concentration":
  The model has two types of parameters:
    * sigma  : the std-dev of the random coefficient on price.
    * beta   : linear taste parameters [beta_p, beta_w].
    
  Given sigma, these are solved analytically via 2SLS, so they
  are filtered out of the outer optimization.
  This reduces the GMM to a 1-D search over sigma only, which is faster and
  than a joint 3-D search over (sigma, beta_p, beta_w).

Two-step GMM:
  Stage 1: Optimize sigma using the identity weighting matrix W_1 = (Z'Z/N)ˆ{-1}.
  Stage 2: Re-weight with the efficient matrix W_2 = Sˆ{-1} = (Z'xi xi'Z/N)ˆ{-1},
           where xi are the structural errors from Stage 1.
"""

import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp
from scipy.optimize import minimize

# -----------------------------------------------------------------------
# tqdm progress bar 
# -----------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def trange(n, desc=None, leave=True):
        print(f"Starting {desc}...")
        for i in range(n):
            yield i
    tqdm = None


# =======================================================================
# PROGRESS BAR WRAPPER
# =======================================================================
class ObjectiveWithPbar:
    """
    Decorator that wraps any scalar objective function and attaches a 
    tqdm progress bar.
    Parameters
    ----------
    f    : callable — the objective function f(x) → float
    desc : str      — label shown on the left of the progress bar
    """

    def __init__(self, f: callable, desc: str = "BLP objective") -> None:
        self.f = f
        self.calls = 0
        self.pbar = tqdm(desc=desc)

    def __call__(self, x: np.ndarray) -> float:
        val = self.f(x)
        self.calls += 1
        self.pbar.update(1)
        self.pbar.set_postfix({"sigma": f"{x[0]:.4f}", "obj": f"{val:.4e}"})
        return val

    def close(self) -> None:
        """Must be called after minimize() to cleanly close the progress bar."""
        self.pbar.close()


# =======================================================================
# BLP HELPER FUNCTIONS
# =======================================================================

def _rc_logit_draw_probs(
    delta: np.ndarray,       # (J,)   — mean utility for each inside good
    price: np.ndarray,       # (J,)   — price of each inside good
    sigma_p: float,          # scalar — std-dev of random coefficient on price
    v_draws: np.ndarray,     # (R,)   — standard-normal simulation draws
) -> np.ndarray:             # (R, J+1) — choice probabilities incl. outside option
    """
    Compute individual-level choice probabilities for all simulation draws.

    For each draw vᵣ ~ N(0,1), consumer r's utility for product j is:
        u_{rj} = delta_j + sigma_p * v_r * p_j

    delta_j is the mean utility and 
    sigma_p * v_r * p_j is the individual-specific deviation 

    Probabilities are computed via the softmax formula, using log-sum-exp 
    Pr_{rj} = exp(u_{rj}) / [1 + sigma_k exp(u_{rk})]

    The outside option (product 0) has utility normalized to 0.
    """
    N_draw = v_draws.shape[0]

    # Individual-specific price deviation: shape (R, J)
    # Each row r corresponds to one simulation draw vᵣ
    mu = (sigma_p * v_draws)[:, None] * price[None, :]

    # Add mean utility: u_{rj} = delta_j + μ_{rj}, shape (R, J)
    U_inside = delta[None, :] + mu

    # Stack the outside option (utility = 0) as column 0: shape (R, J+1)
    U_all = np.concatenate([np.zeros((N_draw, 1)), U_inside], axis=1)

    # Numerically-stable softmax via log-sum-exp trick
    denom = logsumexp(U_all, axis=1, keepdims=True)
    return np.exp(U_all - denom)


def blp_build_ivs(
    w_t: np.ndarray,       # (J,) — non-price characteristic for market t
    u_cost_t: np.ndarray,  # (J,) — cost shock for market t (used in iv_type=1 only)
    iv_type: int,          # 1=cost-shock IVs, 2=weak polynomial IVs, 3=BLP differentiation IVs
) -> np.ndarray:           # (J, K) — instrument matrix for market t
    """
    Construct the instrument matrix Z for a single market.

    The moment condition for GMM is E[Z'xi] = 0, where xi are the
    structural demand shocks.

    Three instrument sets are considered:

    iv_type = 1 — Cost-Shock IVs (Strong)
        Uses [1, w, wˆ2, u_cost, u_costˆ2].

    iv_type = 2 — Weak Polynomial IVs (No cost data, but weak)
        Uses [1, w, wˆ2, wˆ3, wˆ4] where w is standardized.

    iv_type = 3 — BLP Differentiation IVs (No cost data, strong)
        Uses [1, w_own, sigma_{k!=j} w_k, sigma_{k} (w_j - w_k)ˆ2].
        Exploits own price and competitor products's prices. 
    """
    J = len(w_t)

    if iv_type == 1:
        # --- Cost-Shock IVs ---
        # Polynomial terms of own characteristic and cost shock
        return np.column_stack([np.ones_like(w_t), w_t, w_t**2,
                                 u_cost_t, u_cost_t**2])

    elif iv_type == 2:
        # --- Weak Polynomial IVs ---
        # Only use own characteristic w, standardized 
        w_norm = (w_t - np.mean(w_t)) / (np.std(w_t) + 1e-8)
        return np.column_stack([np.ones_like(w_norm), w_norm, w_norm**2,
                                 w_norm**3, w_norm**4])

    elif iv_type == 3:
        # --- BLP Differentiation IVs (Competitor Characteristics) ---

        w_own = w_t

        # Sum of all competitors' characteristics for product j:
        #   sigma_{k != j} w_k = sigma_k w_k - w_j
        w_sum_others = np.sum(w_t) - w_t

        # Sum of squared characteristic distances for product j:
        #   sigma_k (w_j - w_k)ˆ2
        w_diff_sq = np.zeros(J)
        for j in range(J):
            w_diff_sq[j] = np.sum((w_t[j] - w_t)**2)

        # Standardize each instrument to prevent scale-driven multicollinearity in GMM
        w_own_std       = (w_own       - np.mean(w_own))       / (np.std(w_own)       + 1e-8)
        sum_others_std  = (w_sum_others - np.mean(w_sum_others)) / (np.std(w_sum_others) + 1e-8)
        diff_sq_std     = (w_diff_sq    - np.mean(w_diff_sq))    / (np.std(w_diff_sq)    + 1e-8)

        return np.column_stack([np.ones_like(w_t), w_own_std,
                                 sum_others_std, diff_sq_std])

    else:
        raise ValueError("iv_type must be 1, 2, or 3")


def blp_contraction_all_markets(
    s_obs: np.ndarray,       # (T, J+1)  — observed market shares (col 0 = outside option)
    X: np.ndarray,           # (T, J, D) — product characteristics [price, w, ...]
    sigma_p: float,          # scalar    — current guess for price std-dev
    v_draws: np.ndarray,     # (R,)      — fixed simulation draws (same across iterations)
    delta_start: np.ndarray, # (T, J)    — warm-start values for mean utility
    tol: float = 1e-10,      # convergence tolerance (max absolute change in delta)
    max_iter: int = 2000,    # maximum contraction iterations
) -> np.ndarray:             # (T, J)    — converged mean utilities delta
    """
    BLP Contraction Mapping — recover mean utilities delta from observed shares.

    This is done via the fixed-point iteration:
        delta^{(n+1)}_j = delta^{(n)}_j + log(s_j^obs) - log(sigma_j(delta^{(n)}, sigma_p))

    The mapping is run independently for each market t.
    """
    T, J, _ = X.shape
    price = X[:, :, 0]  # (T, J) — price is the first characteristic

    # Clip to avoid log(0); outside option share at index 0 is excluded
    s_in = np.clip(s_obs[:, 1:], 1e-300, 1.0)  # (T, J) — inside goods only
    delta = delta_start.copy()                  # (T, J) — mutable working copy

    for it in range(max_iter):
        delta_old = delta.copy()
        maxdiff = 0.0

        for t in range(T):
            # Compute predicted shares at current delta for market t: shape (R, J+1)
            probs = _rc_logit_draw_probs(delta[t], price[t], sigma_p, v_draws)

            # Average over simulation draws to approximate the integral: shape (J,)
            s_pred_in = np.clip(probs.mean(axis=0)[1:], 1e-300, 1.0)

            # Fixed-point update: delta ← delta + log(s_obs) - log(s_pred)
            delta[t] = delta[t] + (np.log(s_in[t]) - np.log(s_pred_in))

            # Track the maximum change across all products in this market
            maxdiff = max(maxdiff, float(np.max(np.abs(delta[t] - delta_old[t]))))

        # Converged when all elements of delta change by less than tol
        if maxdiff < tol:
            break

    return delta  # (T, J)


# =======================================================================
# CONCENTRATED BLP CLASS
# =======================================================================
class FastBLP:
    """
    The Concentrated BLP estimator.

    We analytically filter out the linear parameters beta
    (via closed-form 2SLS) for any given sigma. This reduces the GMM optimization
    to a 1-D search over sigma alone.

    The GMM objective is:
        Q(sigma) = N g(sigma)' W g(sigma)
    where g(sigma) = (1/N) Z' xi(sigma, hat{beta}(sigma)) are the average moments,
    and hat{beta}(sigma) is the 2SLS estimator of beta given sigma.

    Parameters
    ----------
    data     : dict with keys:
                 "X" : (T, J, D) array of product characteristics
                 "q" : (T, J+1) array of observed quantities (col 0 = outside)
                 "Nt": int, number of consumers per market
    u_cost   : (T, J) array of cost shocks (used only by iv_type=1)
    iv_type  : 1, 2, or 3 — instrument set (see blp_build_ivs)
    N_draw   : number of simulation draws for approximating the share integral
    seed     : random seed for reproducibility of simulation draws
    tol      : contraction mapping convergence tolerance
    max_iter : maximum contraction mapping iterations
    """

    def __init__(
        self,
        data: dict,
        u_cost: np.ndarray,  # (T, J)
        iv_type: int = 1,
        N_draw: int = 200,
        seed: int = 1234,
        tol: float = 1e-10,
        max_iter: int = 2000,
    ) -> None:

        self.T, self.J, _ = data["X"].shape  # T markets, J inside goods, D features
        self.N = self.T * self.J             # total number of market-product observations

        # ------------------------------------------------------------------
        # Build the regressor matrix X_mat — shape (N, 3)
        # Columns: [Intercept (1), Price (p_jt), Characteristic (w_jt)]
        # This is the "X" in the 2SLS formula: hat{beta} = (X'Z W Z'X)ˆ{-1} X'Z W Z'delta
        # ------------------------------------------------------------------
        price_flat = data["X"][:, :, 0].flatten()  # (N,)
        w_flat     = data["X"][:, :, 1].flatten()  # (N,)
        self.X_mat = np.column_stack([np.ones(self.N), price_flat, w_flat])  # (N, 3)

        # ------------------------------------------------------------------
        # Build the instrument matrix Z_mat — shape (N, K)
        # Each row corresponds to one market-product observation.
        # Z_mat is constructed market-by-market and stacked vertically.
        # ------------------------------------------------------------------
        Z_list = []
        for t in range(self.T):
            w_t      = data["X"][t, :, 1]  # (J,)
            u_cost_t = u_cost[t]           # (J,)
            Z_list.append(blp_build_ivs(w_t, u_cost_t, iv_type))
        self.Z_mat = np.concatenate(Z_list, axis=0)  # (N, K)
        self.K = self.Z_mat.shape[1]                 # number of instruments

        # ------------------------------------------------------------------
        # Stage 1 Weighting Matrix: W_1 = (Z'Z / N)ˆ{-1}
        # This is the standard initial GMM weight matrix. It equals the inverse of the sample covariance
        # of the instruments, up to a constant.
        # ------------------------------------------------------------------
        Z_cov   = (self.Z_mat.T @ self.Z_mat) / self.N  # (K, K)
        self.W1 = np.linalg.pinv(Z_cov, rcond=1e-8)     # (K, K)

        # Convert quantities to market shares: s_obs[t, 0] = outside, s_obs[t, 1:] = inside
        self.s_obs    = data["q"] / data["Nt"]   # (T, J+1)
        self.X_tensor = data["X"]                # (T, J, D)

        # ------------------------------------------------------------------
        # Simulation draws — drawn once and fixed throughout estimation.
        # Using the same draws across iterations ensures that the objective
        # function is smooth in sigma (no simulation noise from re-drawing).
        # R = N_draw draws approximate: integral Pr(·|v) phi(v) dv approx (1/R) sigma_r Pr(·|vᵣ)
        # ------------------------------------------------------------------
        rng          = default_rng(seed)
        self.v_draws = rng.normal(0.0, 1.0, size=N_draw)  # (R,)

        # ------------------------------------------------------------------
        # Warm-start delta with the simple logit inversion (sigma=0 solution):
        #   delta_j^{(0)} = log(s_j) - log(s_0)
        # This is exact when sigma=0 and provides a good starting point for sigma>0.
        # ------------------------------------------------------------------
        s0            = np.clip(self.s_obs[:, 0],  1e-300, 1.0)   # (T,)
        s_in          = np.clip(self.s_obs[:, 1:], 1e-300, 1.0)   # (T, J)
        self.delta_last = np.log(s_in) - np.log(s0[:, None])       # (T, J)

        self.tol      = tol
        self.max_iter = max_iter

        # Will be populated after the first call to compute_g_xi()
        self.beta_last: np.ndarray | None = None  # (3,) — [intercept, beta_p, beta_w]
        self.xi_last:   np.ndarray | None = None  # (N,) — structural errors xi_jt

    def compute_g_xi(
        self,
        sigma: float,      # current sigma guess
        W: np.ndarray,     # (K, K) — current GMM weighting matrix
    ) -> np.ndarray:       # (K,)   — average moment vector g = (1/N) Z'xi
        """
        Core computation for a given sigma and weighting matrix W.

        Steps:
          1. Run contraction mapping to get delta(sigma) from observed shares.
          2. Solve for beta analytically via closed-form 2SLS.
          3. Compute structural errors xi = delta - X·beta.
          4. Compute and return the average moment vector g = Z'xi / N.

        -------------------------------------------------------------------
        Closed-Form 2SLS for Linear Parameters
        -------------------------------------------------------------------
        The structural demand equation (in mean-utility space) is:
            delta_jt = X_jt beta + xi_jt

        Given sigma, delta is known from the contraction mapping, and the 2SLS
        estimator of beta treats price as endogenous with instruments Z:

           hat{beta}(sigma) = (X'Z W Z'X)ˆ{-1} X'Z W Z'delta

        Letting:
            A = X'Z W Z'X    (3×3 matrix, the "sandwich" denominator)
            b = X'Z W Z'delta    (3×1 vector, the "sandwich" numerator)

        we solve the linear system A hat{beta} = b. This is an analytic closed-form
        solution
        -------------------------------------------------------------------
        """

        # Step 1: Contraction mapping — recover delta(sigma) from observed shares
        delta_tensor = blp_contraction_all_markets(
            self.s_obs, self.X_tensor, sigma, self.v_draws,
            self.delta_last, self.tol, self.max_iter
        )
        self.delta_last = delta_tensor          # (T, J) — warm-start for next sigma
        delta_flat      = delta_tensor.flatten()  # (N,)

        # Step 2: Closed-form 2SLS — solve for hat{beta}(sigma)
        XZ = self.X_mat.T @ self.Z_mat  # (3, K)
        ZX = self.Z_mat.T @ self.X_mat  # (K, 3)
        A  = XZ @ W @ ZX               # (3, 3) — 2SLS denominator matrix

        Z_delta = self.Z_mat.T @ delta_flat  # (K,)
        b       = XZ @ W @ Z_delta           # (3,)  — 2SLS numerator vector

        # Solve A hat{beta} = b analytically; rcond=None uses full numerical precision
        self.beta_last = np.linalg.lstsq(A, b, rcond=None)[0]  # (3,)

        # Step 3: Compute structural errors xi = delta - X·hat{beta}
        xi_flat       = delta_flat - self.X_mat @ self.beta_last  # (N,)
        self.xi_last  = xi_flat

        # Step 4: Average moment vector g = (1/N) Z' xi
        # This is the sample analogue of E[Z xi] = 0 (the GMM moment condition)
        g = (self.Z_mat.T @ xi_flat) / self.N  # (K,)
        return g

    def obj(
        self,
        sigma_array: np.ndarray,  # (1,) — optimizer passes sigma as a 1-element array
        W: np.ndarray,            # (K, K) — current GMM weighting matrix
    ) -> float:
        """
        1-D GMM objective function Q(sigma) = N g(sigma)' W g(sigma).
        Returns a large penalty (1e12) when sigma < 1e-6 to enforce sigma > 0.
        """
        sigma = sigma_array[0]
        if sigma < 1e-6:
            return 1e12  # penalty: sigma must be strictly positive

        g = self.compute_g_xi(sigma, W)
        return float(g.T @ W @ g) * self.N  # scalar GMM criterion


# =======================================================================
# TWO-STEP GMM ESTIMATOR (Main Entry Point)
# =======================================================================
def blp_estimator_fast(
    data: dict,              # see FastBLP docstring for format
    u_cost: np.ndarray,      # (T, J) — cost shocks for instrument construction
    iv_type: int = 1,        # instrument type: 1=cost-shock, 2=weak-poly, 3=BLP-diff
    N_draw: int = 200,       # number of simulation draws for share integral
    seed: int = 1234,        # RNG seed for reproducibility
    bounds: list = [(1e-6, 3.0)],  # search bounds for sigma
) -> dict:
    """
    Two-Step Efficient GMM estimator for the concentrated BLP model.

    -------------------------------------------------------------------
    Two-Stage GMM Optimization
    -------------------------------------------------------------------
    Stage 1 — Consistent (but inefficient) estimate:
        Minimize Q₁(sigma) = N g(sigma)' W_1 g(sigma)
        using the initial weighting matrix W_1 = (Z'Z/N)ˆ{-1}.
        This gives a consistent hat{sigma}_1 and first-step residuals hat{xi}_1.

    Stage 2 — Asymptotically efficient estimate:
        Re-weight using the optimal matrix W_2 =  hat{S}ˆ{-1}, where:
             hat{S} = (1/N) sigma_jt (Z_jt hat{xi1}_jt)' (Z_jt hat{xi1}_jt)
        Minimize Q₂(sigma) = N g(sigma)' W_2 g(sigma) starting from hat{sigma}_1.

    Both stages use L-BFGS-B (a bounded quasi-Newton method) since the
    objective is smooth (same simulation draws are reused).
    -------------------------------------------------------------------

    Returns
    -------
    dict with keys:
        theta     : (3,) — [beta_p, beta_w, sigma]
        beta_bar  : (2,) — [beta_p, beta_w]
        sigma_p   : float — estimated sigma (price random-coefficient std-dev)
        intercept : float — estimated intercept (= mean market-level utility bar{xi})
        xi        : (T, J) — estimated demand shocks xi_jt = bar{xi}_t + η_jt
        xi_bar    : (T,)   — market-level intercepts (= intercept for each t)
        eta       : (T, J) — product-level deviations from market mean
        delta     : (T, J) — converged mean utilities delta_jt
        res_stage1, res_stage2 : scipy OptimizeResult objects
    """

    print(f"\n--- Starting BLP IV Type {iv_type} ---")
    blp = FastBLP(data, u_cost, iv_type, N_draw, seed)


    # ------------------------------------------------------------------
    # STAGE 1: Initial consistent estimate of sigma
    # Weighting matrix: W_1 = (Z'Z/N)ˆ{-1}  (identity-style, not efficient)
    # ------------------------------------------------------------------
    stage1_wrapped = ObjectiveWithPbar(
        lambda sig: blp.obj(sig, blp.W1),
        desc=f"Stage 1 (IV={iv_type})"
    )
    res1 = minimize(stage1_wrapped, x0=[1.5], bounds=bounds, method="L-BFGS-B")
    stage1_wrapped.close()

    sigma1 = res1.x[0]

    # Recompute at hat{sigma}_1 to get consistent first-step structural errors hat{xi}_1
    blp.compute_g_xi(sigma1, blp.W1)
    xi1 = blp.xi_last  # (N,) — first-step residuals

    # ------------------------------------------------------------------
    # STAGE 2 WEIGHTING MATRIX: W_2 =  hat{S}ˆ{-1}  (optimal / efficient)
    #  hat{S} = (1/N) Z' diag(hat{xi}_1ˆ2) Z  — heteroskedasticity-robust sandwich
    # Using the first-step hat{xi}_1 to estimate the long-run variance of the moments.
    # ------------------------------------------------------------------
    Z_xi = blp.Z_mat * xi1[:, None]       # (N, K) — each row: Z_jt hat{xi}_1_jt
    S    = (Z_xi.T @ Z_xi) / blp.N        # (K, K) — estimated moment variance
    W2   = np.linalg.pinv(S, rcond=1e-8)  # (K, K) — optimal weighting matrix

    # ------------------------------------------------------------------
    # STAGE 2: Efficient re-estimation of sigma
    # Warm-start from hat{sigma}_1 to reduce the number of optimizer evaluations.
    # ------------------------------------------------------------------
    stage2_wrapped = ObjectiveWithPbar(
        lambda sig: blp.obj(sig, W2),
        desc=f"Stage 2 (IV={iv_type})"
    )
    res2 = minimize(stage2_wrapped, x0=[sigma1], bounds=bounds, method="L-BFGS-B")
    stage2_wrapped.close()

    sigma2 = res2.x[0]

    # Compute final hat{beta} at the Stage 2 optimal sigmâ₂
    blp.compute_g_xi(sigma2, W2)

    # ------------------------------------------------------------------
    # Unpack and organize output
    # ------------------------------------------------------------------
    beta_0 = blp.beta_last[0]  # intercept = mean market-level shock bar{xi}
    beta_p = blp.beta_last[1]  # price coefficient beta_p
    beta_w = blp.beta_last[2]  # characteristic coefficient beta_w

    theta_hat = np.array([beta_p, beta_w, sigma2])   # (3,) — main parameters

    # Reshape structural errors from flat (N,) → market-product (T, J)
    xi_tensor   = blp.xi_last.reshape(blp.T, blp.J)   # (T, J) — η_jt deviations
    xi_bar_hat  = np.full(blp.T, beta_0)              # (T,)   — market intercepts
    eta_hat     = xi_tensor                           # (T, J) — product deviations

    out = {
        "theta"     : theta_hat,           # [beta_p, beta_w, sigma]
        "beta_bar"  : theta_hat[:2],       # [beta_p, beta_w]
        "sigma_p"   : sigma2,              # sigma (price random-coefficient std-dev)
        "intercept" : beta_0,              # bar{xi} (mean market-level quality)
        "xi"        : xi_tensor + beta_0,  # full xi_jt = bar{xi} + η_jt
        "xi_bar"    : xi_bar_hat,          # bar{xi}_t (market-level component)
        "eta"       : eta_hat,             # η_jt (product-level deviation)
        "delta"     : blp.delta_last.copy(), # delta_jt (mean utilities at sigmâ₂)
        "res_stage1": res1,
        "res_stage2": res2,
    }

    print(f"BLP {iv_type} final beta: {theta_hat[:2]}")
    print(f"BLP {iv_type} final sigma (std): {float(theta_hat[2])}") 

    return out

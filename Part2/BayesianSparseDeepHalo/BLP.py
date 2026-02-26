import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp
from scipy.optimize import minimize

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
# BLP IMPLEMENTATION
# =====================================

# Helper function to compute logit probabilities for a single market given a set of random draws.
def _rc_logit_draw_probs(delta, price, beta_p_mean, sigma_p, v_draws):
    N_draw = v_draws.shape[0]
    beta_p_draw = beta_p_mean + sigma_p * v_draws
    mu = (beta_p_draw - beta_p_mean)[:, None] * price[None, :]
    U_inside = delta[None, :] + mu
    U_all = np.concatenate([np.zeros((N_draw, 1)), U_inside], axis=1)
    denom = logsumexp(U_all, axis=1, keepdims=True)
    return np.exp(U_all - denom)


def blp_build_ivs(w_t, u_cost_t, iv_type):
    if iv_type == 1:
        return np.column_stack([np.ones_like(w_t), w_t, w_t**2, u_cost_t, u_cost_t**2])
    elif iv_type == 2:
        return np.column_stack([np.ones_like(w_t), w_t, w_t**2, w_t**3, w_t**4])
    else:
        raise ValueError("iv_type must be 1 or 2")


def blp_contraction_all_markets(
    s_obs, X, beta_bar, sigma_p, v_draws,
    delta_start, tol=1e-10, max_iter=2000,
    pbar_iters=None
):
    """
    Contraction for all markets
    """
    T, J, _ = X.shape
    price = X[:, :, 0]
    s_in = np.clip(s_obs[:, 1:], 1e-300, 1.0)

    delta = delta_start.copy()

    for it in range(max_iter):
        delta_old = delta
        maxdiff = 0.0

        for t in range(T):
            probs = _rc_logit_draw_probs(delta[t], price[t], beta_bar[0], sigma_p, v_draws)
            s_pred_in = np.clip(probs.mean(axis=0)[1:], 1e-300, 1.0)
            delta[t] = delta[t] + (np.log(s_in[t]) - np.log(s_pred_in))
            maxdiff = max(maxdiff, float(np.max(np.abs(delta[t] - delta_old[t]))))

        if pbar_iters is not None:
            pbar_iters.update(1)
            pbar_iters.set_postfix({"maxdiff": f"{maxdiff:.2e}"})

        if maxdiff < tol:
            break

    return delta


class ObjectiveWithPbar:
    """
    Wraps an objective function f(x) and updates a tqdm progress bar on each call.
    """
    def __init__(self, f, total_calls=None, desc="BLP objective"):
        self.f = f
        self.calls = 0
        self.pbar = None
        if tqdm is not None:
            self.pbar = tqdm(total=total_calls, desc=desc, leave=True)

    def __call__(self, x):
        val = self.f(x)
        self.calls += 1
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"calls": self.calls, "obj": f"{val:.4e}"})
        return val

    def close(self):
        if self.pbar is not None:
            self.pbar.close()


class FastBLP:
    def __init__(self, data, u_cost, iv_type=1, N_draw=200, seed=1234,
                 tol=1e-10, max_iter=2000,
                 show_contraction_iters=False):
        self.X = data["X"]
        self.q = data["q"]
        self.Nt = data["Nt"]
        self.s_obs = self.q / self.Nt

        self.u_cost = u_cost
        self.iv_type = iv_type

        self.T, self.J, self.d = self.X.shape
        assert self.d == 2

        rng = default_rng(seed)
        self.v_draws = rng.normal(0.0, 1.0, size=N_draw)

        self.Z_list = []
        for t in range(self.T):
            w_t = self.X[t, :, 1]
            self.Z_list.append(blp_build_ivs(w_t, self.u_cost[t], iv_type))
        self.K = self.Z_list[0].shape[1]

        s0 = np.clip(self.s_obs[:, 0], 1e-300, 1.0)  # 1e-300
        s_in = np.clip(self.s_obs[:, 1:], 1e-300, 1.0)
        self.delta_last = np.log(s_in) - np.log(s0[:, None])  # warm start

        self.tol = tol
        self.max_iter = max_iter
        self.show_contraction_iters = show_contraction_iters

        self.xi_last = None
        self.g_last = None

    def compute_g_xi(self, params):
        beta_p, beta_w, sigma = params
        beta_bar = np.array([beta_p, beta_w])

       
        pbar_iters = None
        if self.show_contraction_iters and tqdm is not None:
            pbar_iters = tqdm(total=min(self.max_iter, 500), desc="Contraction iters", leave=False)

        delta = blp_contraction_all_markets(
            self.s_obs, self.X, beta_bar, sigma, self.v_draws,
            delta_start=self.delta_last, tol=self.tol, max_iter=self.max_iter,
            pbar_iters=pbar_iters
        )
        if pbar_iters is not None:
            pbar_iters.close()

        self.delta_last = delta

        XB = np.einsum("tjd,d->tj", self.X, beta_bar)
        xi = delta - XB

        g = np.zeros(self.K)
        for t in range(self.T):
            g += self.Z_list[t].T @ xi[t]

        self.xi_last = xi
        self.g_last = g
        return g, xi

    def get_xi_for_params(self, params):
        g, xi = self.compute_g_xi(params)   # updates self.delta_last, self.xi_last
        delta = self.delta_last.copy()
        return delta, xi

    def stage1_obj(self, params):
        beta_p, beta_w, sigma = params
        if sigma < 0 or beta_p > 0:
            return 1e12
        g, _ = self.compute_g_xi(params)
        return float(g @ g)

    def stage2_obj(self, params, Weight):
        beta_p, beta_w, sigma = params
        if sigma < 0 or beta_p > 0:
            return 1e12
        g, _ = self.compute_g_xi(params)
        return float(g.T @ Weight @ g)


# 2-stage least squares
def blp_estimator_fast(
    data, u_cost, iv_type=1, N_draw=200, seed=1234,
    bounds=((-2, 0), (0, 1), (1e-6, 3.0)),
    maxiter=120,
    show_obj_progress=True,
    show_contraction_iters=False
):
    blp = FastBLP(
        data, u_cost, iv_type=iv_type, N_draw=N_draw, seed=seed,
        show_contraction_iters=show_contraction_iters
    )

    rng2 = default_rng(int(1e6 * default_rng().random()))
    init = np.array([
        rng2.uniform(-2.0, -0.5),
        rng2.uniform(0.2, 0.8),
        rng2.uniform(0.5, 3.0),
    ])

    
    stage1 = blp.stage1_obj
    stage2 = None

    stage1_wrapped = ObjectiveWithPbar(stage1, total_calls=None, desc=f"BLP IV{iv_type} stage1") \
        if (show_obj_progress and tqdm is not None) else stage1

    res1 = minimize(stage1_wrapped, init, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": maxiter, "ftol": 1e-8})
    if hasattr(stage1_wrapped, "close"):
        stage1_wrapped.close()

    theta1 = res1.x

    g1 = blp.g_last.reshape(-1, 1)
    K = g1.shape[0]
    Weight = np.linalg.inv((g1 @ g1.T) / (blp.T * blp.J) + 1e-6 * np.eye(K))

    stage2 = lambda th: blp.stage2_obj(th, Weight)
    stage2_wrapped = ObjectiveWithPbar(stage2, total_calls=None, desc=f"BLP IV{iv_type} stage2") \
        if (show_obj_progress and tqdm is not None) else stage2

    res2 = minimize(stage2_wrapped, theta1, bounds=bounds, method="L-BFGS-B",
                    options={"maxiter": maxiter, "ftol": 1e-8})
    if hasattr(stage2_wrapped, "close"):
        stage2_wrapped.close()
        
    theta_hat = res2.x

    # Ensure xi_last corresponds to theta_hat (minimize doesn't guarantee last callback == optimum)
    _, xi_hat = blp.get_xi_for_params(theta_hat)   # (T,J) inside only

    # Decompose xi into market mean + residual (optional)
    xi_bar_hat = xi_hat.mean(axis=1)               # (T,)
    eta_hat = xi_hat - xi_bar_hat[:, None]         # (T,J)

    out = {
        "theta": theta_hat,        # (3,)
        "beta_bar": theta_hat[:2], # (2,)
        "sigma_p": float(theta_hat[2]),
        "xi": xi_hat,              # (T,J)
        "xi_bar": xi_bar_hat,      # (T,)
        "eta": eta_hat,            # (T,J)
        "delta": blp.delta_last.copy(),  # (T,J) inside delta (after contraction)
        "res_stage1": res1,
        "res_stage2": res2,
    }
    print(f"BLP {iv_type} final beta: {theta_hat[:2]}")
    print(f"BLP {iv_type} final sigma (std): {float(theta_hat[2])}")
    return out


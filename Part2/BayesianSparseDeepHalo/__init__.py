"""BayesianSparseDeepHalo — package."""

from .GenerateData import SimParams, generate_data_tf
from .BLP import _rc_logit_draw_probs, blp_build_ivs, FastBLP, blp_estimator_fast
from .DeepHalo import NonlinearMap, DeepHaloEncoder
from .LuSparseRandomLogit import (
    BayesianSparseRandomLogit,
    adapt_step_size,
    gibbs_update_gamma_phi_tf,
    mh_update_beta_cl,
    mh_update_xi_cl,
    mh_update_r_cl,
    tmh_update_beta_cl,
    calibrate_stepsizes_cl,
)
from .DeepHalo_MCEM_Core import (
    SparseDeepHaloMCEM,
    build_choice_dataset_from_market_counts,
    compute_probs_and_ll_batch_masked,
)

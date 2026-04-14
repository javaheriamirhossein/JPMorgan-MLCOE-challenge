"""BayesianSparseDeepHalo — package."""

from .GenerateData import SimParams, MCMCParams, generate_data_tf, generate_teacher_dgp
from .BLP import _rc_logit_draw_probs, blp_build_ivs, FastBLP, blp_estimator_fast
from .DeepHalo import NonlinearMap, DeepHaloEncoder
from .LuSparseRandomLogit import BayesianSparseRandomLogit
from .DeepHalo_MCEM_Core import (
    SparseDeepHaloMCEM,
    build_choice_dataset_from_market_counts,
    compute_probs_and_ll_batch_masked,
)

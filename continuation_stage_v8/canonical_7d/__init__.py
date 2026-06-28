"""Canonical 7D world model package for V8."""
from .config import (
    STATE_NAMES_7D, STATE_DIM, ACTION_DIM, IDX_7D_FROM_8D,
    DataConfig, TrainingConfig, RDEConfig, EvalConfig, CalibConfig, PlanConfig, StatisticalConfig
)
from .data_loader import load_7d_data, get_test_segments
from .models import BaseModel, GPModel, SINDyModel
from .residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid
from .evaluation_modes import EvalResult, run_mode_a, run_mode_b, run_mode_c
from .metrics import compute_metrics, compute_cost
from .runner import train_all_models, evaluate_model_on_segments, aggregate_results
from .extended_models import GPEnsemble, AdaptiveResidualGP, NeuralODEModel, NeuralODEMultiStep

"""Canonical 7D configuration for V8 world model."""
from dataclasses import dataclass, field
from typing import List
import numpy as np

# 7D state: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
# NO kappa (index 5 in original 8D is always zero)
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1

# Indices to extract 7D from original 8D: [0,1,2,3,4,6,7] (skip index 5=kappa)
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]


@dataclass(frozen=True)
class DataConfig:
    data_path: str = "data/stage2_dataset_150k.npz"
    train_ratio: float = 0.75
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    gp_max_samples: int = 2000
    gp_n_restarts: int = 2
    n_models: int = 5
    n_epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    residual_scale: float = 0.3


@dataclass(frozen=True)
class RDEConfig:
    dagger_rounds_L: int = 0
    dagger_rounds_T: int = 1
    dagger_rounds_M: int = 2
    n_segments_dagger: int = 3
    dagger_segment_length: int = 500


@dataclass(frozen=True)
class EvalConfig:
    eval_horizons: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    )
    n_eval_segments: int = 10
    segment_length: int = 200


@dataclass(frozen=True)
class CalibConfig:
    target_coverage: float = 0.95
    scale_search_range: List[float] = field(
        default_factory=lambda: [float(x) / 100 for x in range(10, 500, 5)]
    )
    n_bins_ece: int = 10
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.68, 0.90, 0.95, 0.99]
    )


@dataclass(frozen=True)
class PlanConfig:
    n_candidate_sequences: int = 50
    cost_weights: dict = field(default_factory=lambda: {
        'e_y': 100.0, 'e_psi': 100.0, 'v': 1.0,
        'theta': 10.0, 'theta_dot': 1.0,
        'delta': 10.0, 'delta_dot': 1.0, 'u': 0.1
    })
    plan_horizons: List[int] = field(
        default_factory=lambda: [10, 20, 50]
    )


@dataclass(frozen=True)
class StatisticalConfig:
    n_train_seeds: int = 5
    n_eval_seeds: int = 10
    eval_horizons_stat: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    )

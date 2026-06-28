"""Canonical 4D configuration with fixed seeds and reproducibility."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    v0: float = 3.5
    g: float = 9.81
    dt: float = 1.0 / 30.0
    dt_sub: int = 5
    phi_limit: float = np.pi / 3
    state_limit: float = 100.0


@dataclass(frozen=True)
class TrainingConfig:
    n_samples: int = 5000
    seed: int = 42
    gp_max_samples: int = 2000
    gp_n_restarts: int = 2
    n_models: int = 5
    n_epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    use_scheduler: bool = True
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
    eval_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100, 200, 500, 1000])
    mode_b_segments: int = 5
    mode_b_segment_length: int = 500
    failure_threshold: float = np.pi / 3


@dataclass(frozen=True)
class CalibConfig:
    target_coverage: float = 0.95
    scale_search_range: List[float] = field(default_factory=lambda: [float(x)/100 for x in range(10, 200, 5)])
    n_bins_ece: int = 10
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95, 0.99])


@dataclass(frozen=True)
class PlanConfig:
    n_candidate_sequences: int = 50
    cost_weights: dict = field(default_factory=lambda: {
        'phi': 100.0, 'delta': 100.0, 'phi_dot': 10.0, 'delta_dot': 1.0, 'u': 0.1
    })
    plan_horizons: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])


@dataclass(frozen=True)
class StatisticalConfig:
    n_train_seeds: int = 5
    n_eval_seeds: int = 10
    eval_horizons_stat: List[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    )

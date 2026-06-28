"""Training and evaluation runner for canonical 7D models."""
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict
from .config import TrainingConfig, EvalConfig, STATE_NAMES_7D, STATE_DIM
from .data_loader import load_7d_data, get_test_segments
from .models import GPModel, SINDyModel
from .residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid
from .evaluation_modes import run_mode_a, run_mode_b, run_mode_c
from .metrics import compute_metrics


def train_all_models(data: dict, train_config=None, data_segments=None):
    """Train all models on 7D real data."""
    cfg = train_config or TrainingConfig()
    states = data['train_states']
    actions = data['train_actions']
    deltas = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    models = {}

    # GP
    t0 = time.time()
    gp = GPModel(max_samples=cfg.gp_max_samples, n_restarts=cfg.gp_n_restarts)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    models['gp'] = gp
    print(f"  GP trained in {time.time()-t0:.1f}s")

    # E1 (offline NN)
    t0 = time.time()
    e1 = E1OfflineNN(n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale)
    e1.train(states, actions, deltas, state_std, action_std, delta_std)
    models['e1'] = e1
    print(f"  E1 trained in {time.time()-t0:.1f}s")

    # RDE-L
    t0 = time.time()
    rde_l = RDELocal(n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std)
    models['rde_l'] = rde_l
    print(f"  RDE-L trained in {time.time()-t0:.1f}s")

    # RDE-T
    t0 = time.time()
    rde_t = RDETrajectory(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=1, n_segments=3, segment_length=500
    )
    rde_t.train(states, actions, deltas, state_std, action_std, delta_std,
                data_segments=data_segments)
    models['rde_t'] = rde_t
    print(f"  RDE-T trained in {time.time()-t0:.1f}s")

    # RDE-M
    t0 = time.time()
    rde_m = RDEHybrid(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=2, n_segments=3, segment_length=500
    )
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                data_segments=data_segments)
    models['rde_m'] = rde_m
    print(f"  RDE-M trained in {time.time()-t0:.1f}s")

    # SINDy
    t0 = time.time()
    sindy = SINDyModel()
    sindy.train(states, actions, deltas, state_std, action_std, delta_std)
    models['sindy'] = sindy
    print(f"  SINDy trained in {time.time()-t0:.1f}s")

    return models


def evaluate_model_on_segments(model, segments: list, mode: str = 'b'):
    """Evaluate a model on test segments.

    Args:
        model: world model
        segments: list of segment dicts from get_test_segments
        mode: evaluation mode ('a', 'b', 'c')
    Returns:
        list of EvalResult
    """
    results = []
    for seg in segments:
        s0 = seg['states'][0]
        actions = seg['actions']
        real_states = seg['states']
        n_steps = len(actions)

        if mode == 'a':
            result = run_mode_a(model, s0, actions, n_steps, real_states)
        elif mode == 'b':
            result = run_mode_b(model, s0, actions, n_steps, real_states)
        elif mode == 'c':
            result = run_mode_c(model, s0, actions, n_steps, real_states)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        results.append(result)

    return results


def aggregate_results(results: list, eval_horizons: list, state_std: np.ndarray):
    """Aggregate results across segments for given horizons."""
    aggregated = {}
    for h in eval_horizons:
        metrics_list = []
        for result in results:
            n = min(h + 1, len(result.states_model))
            m = compute_metrics(
                result.states_model[:n], result.states_real[:n],
                state_std, h,
                survival_steps=min(result.survival_steps, h),
                failure_before_horizon=result.failure_before_horizon
            )
            metrics_list.append(m)
        aggregated[h] = metrics_list
    return aggregated

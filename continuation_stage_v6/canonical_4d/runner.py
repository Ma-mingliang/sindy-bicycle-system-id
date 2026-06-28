"""Training and evaluation runner for canonical 4D models."""
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict
from .config import TrainingConfig, EvalConfig
from .dynamics import BicycleDynamics
from .models import GPModel, SINDyModel
from .residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid
from .evaluation_modes import run_mode_a, run_mode_b, run_mode_c
from .metrics import compute_metrics, STATE_NAMES


def generate_training_data(n_samples=5000, seed=42, dynamics=None):
    """Generate random (s, tau) -> delta_s training data."""
    rng = np.random.RandomState(seed)
    phis = rng.uniform(-0.5, 0.5, n_samples)
    deltas = rng.uniform(-0.3, 0.3, n_samples)
    phi_dots = rng.uniform(-2.0, 2.0, n_samples)
    delta_dots = rng.uniform(-1.0, 1.0, n_samples)
    taus = rng.uniform(-50, 50, n_samples)

    states = np.column_stack([phis, deltas, phi_dots, delta_dots])
    delta_s = np.empty_like(states)
    for i in range(n_samples):
        s_next = dynamics.step(states[i], taus[i])
        delta_s[i] = s_next - states[i]

    state_std = np.std(states, axis=0)
    action_std = float(np.std(taus))
    delta_std = np.std(delta_s, axis=0)

    return states, taus, delta_s, state_std, action_std, delta_std


def train_all_models(states, actions, deltas, state_std, action_std, delta_std,
                     dynamics, train_config=None):
    """Train all 5 models with same budget."""
    cfg = train_config or TrainingConfig()

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

    # RDE-L (local residual, 0 DAgger rounds)
    t0 = time.time()
    rde_l = RDELocal(n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std)
    models['rde_l'] = rde_l
    print(f"  RDE-L trained in {time.time()-t0:.1f}s")

    # RDE-T (trajectory sync, 1 DAgger round)
    t0 = time.time()
    make_tau = dynamics.generate_lqr_tau
    rde_t = RDETrajectory(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=1, n_segments=3, segment_length=500
    )
    rde_t.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    models['rde_t'] = rde_t
    print(f"  RDE-T trained in {time.time()-t0:.1f}s")

    # RDE-M (hybrid, 2 DAgger rounds)
    t0 = time.time()
    rde_m = RDEHybrid(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=2, n_segments=3, segment_length=500
    )
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    models['rde_m'] = rde_m
    print(f"  RDE-M trained in {time.time()-t0:.1f}s")

    # SINDy
    t0 = time.time()
    sindy = SINDyModel()
    sindy.train(states, actions, deltas, state_std, action_std, delta_std)
    models['sindy'] = sindy
    print(f"  SINDy trained in {time.time()-t0:.1f}s")

    return models


def evaluate_model(model, dynamics, eval_config=None, state_std=None,
                   n_segments=5, segment_length=500, mode='b'):
    """Evaluate a model across multiple segments."""
    cfg = eval_config or EvalConfig()
    all_results = []
    tau_func_source = dynamics.generate_lqr_tau

    for seg in range(n_segments):
        tau_func = tau_func_source(seg, segment_length)
        phi0 = np.random.uniform(-0.2, 0.2)
        s0 = np.array([phi0, 0.0, 0.0, 0.0])

        if mode == 'a':
            result = run_mode_a(model, s0, tau_func, segment_length, dynamics)
        elif mode == 'b':
            result = run_mode_b(model, s0, tau_func, segment_length, dynamics)
        elif mode == 'c':
            result = run_mode_c(model, s0, tau_func, segment_length, dynamics)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_results.append(result)

    return all_results


def aggregate_results(all_results, eval_horizons, state_std):
    """Aggregate results across segments for given horizons."""
    aggregated = {}
    for h in eval_horizons:
        metrics_list = []
        for result in all_results:
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

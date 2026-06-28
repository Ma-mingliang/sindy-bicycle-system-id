"""V6 Agent C: Horizon Crossover Analysis.

Determines the exact horizon where RDE models start outperforming pure GP
in Mode B open-loop evaluation. Uses a single rollout per segment and
computes NMAE at all horizons from the same trajectory (optimized).
"""
import sys
import os
import json
import time
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig, EvalConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d import models as _models_mod
from canonical_4d.models import GPModel
from canonical_4d.residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid
from canonical_4d.evaluation_modes import run_mode_b
from canonical_4d.runner import generate_training_data
from canonical_4d.metrics import compute_metrics

# Monkey-patch GPModel to cap max_samples at 1000 (avoid OOM on this machine)
_ORIG_GP_INIT = GPModel.__init__
def _PATCHED_GP_INIT(self, max_samples=2000, n_restarts=2):
    _ORIG_GP_INIT(self, max_samples=min(max_samples, 1000), n_restarts=n_restarts)
GPModel.__init__ = _PATCHED_GP_INIT

OUTPUT = Path(__file__).parent
OUTPUT.mkdir(exist_ok=True)

FINE_GRAINED_HORIZONS = [
    1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75,
    100, 125, 150, 200, 250, 300, 400, 500, 750, 1000,
]

N_EVAL_SEGMENTS = 3
N_TRAIN_SAMPLES = 5000
GP_DIVERGE_THRESHOLD = 1.0


def train_all_models(states, actions, deltas, state_std, action_std, delta_std, dynamics):
    """Train GP, E1, RDE-L, RDE-T, RDE-M with same budget."""
    cfg = TrainingConfig()
    models = {}

    # GP
    t0 = time.time()
    gp = GPModel(max_samples=cfg.gp_max_samples, n_restarts=cfg.gp_n_restarts)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    models['gp'] = gp
    print(f"  GP trained in {time.time()-t0:.1f}s")

    # E1
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
    make_tau = dynamics.generate_lqr_tau
    rde_t = RDETrajectory(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=1, n_segments=3, segment_length=500
    )
    rde_t.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    models['rde_t'] = rde_t
    print(f"  RDE-T trained in {time.time()-t0:.1f}s")

    # RDE-M
    t0 = time.time()
    rde_m = RDEHybrid(
        n_models=cfg.n_models, n_epochs=cfg.n_epochs, residual_scale=cfg.residual_scale,
        dagger_rounds=2, n_segments=3, segment_length=500
    )
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    models['rde_m'] = rde_m
    print(f"  RDE-M trained in {time.time()-t0:.1f}s")

    return models


def evaluate_model_all_horizons(model, dynamics, max_horizon, n_segments, state_std, horizons):
    """Evaluate a model in Mode B at ALL horizons using a single long rollout per segment.

    For each segment, we run Mode B once up to max_horizon+1 steps, then compute
    NMAE at each requested horizon from the same trajectory.
    """
    nmae_by_h = {str(h): [] for h in horizons}

    for seg in range(n_segments):
        tau_func = dynamics.generate_lqr_tau(seg + 600, max_horizon + 1)
        phi0 = np.random.uniform(-0.2, 0.2)
        s0 = np.array([phi0, 0.0, 0.0, 0.0])
        result = run_mode_b(model, s0, tau_func, max_horizon + 1, dynamics)

        sm = result.states_model
        sr = result.states_real
        n_available = len(sm)

        for h in horizons:
            n = min(h + 1, n_available)
            if n < 2:
                nmae_by_h[str(h)].append(float('inf'))
                continue
            metrics = compute_metrics(
                sm[:n], sr[:n], state_std, h,
                survival_steps=min(result.survival_steps, h),
                failure_before_horizon=result.failure_before_horizon
            )
            nmae_val = metrics.get('overall', {}).get('nmae', float('inf'))
            nmae_by_h[str(h)].append(float(nmae_val) if np.isfinite(nmae_val) else float('inf'))

    # Average across segments
    avg_nmae = {}
    for h_str, vals in nmae_by_h.items():
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            avg_nmae[h_str] = float(np.mean(finite_vals))
        else:
            avg_nmae[h_str] = float('inf')
    return avg_nmae


def main():
    print("=" * 80)
    print("V6 AGENT C: Horizon Crossover Analysis")
    print("=" * 80)
    t_start = time.time()

    dyn = BicycleDynamics()

    # --- Phase 1: Train ---
    print(f"\n[Phase 1] Generating {N_TRAIN_SAMPLES} training samples...")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        N_TRAIN_SAMPLES, 42, dyn
    )
    print(f"  state_std: {state_std}")
    print(f"  action_std: {action_std:.4f}")
    print(f"  delta_std:  {delta_std}")

    print("\n[Phase 2] Training all models...")
    models = train_all_models(states, actions, deltas, state_std, action_std, delta_std, dyn)

    # --- Phase 3: Evaluate at fine-grained horizons (single rollout per segment) ---
    max_h = max(FINE_GRAINED_HORIZONS)
    print(f"\n[Phase 3] Evaluating {len(models)} models at {len(FINE_GRAINED_HORIZONS)} horizons "
          f"(max={max_h}, {N_EVAL_SEGMENTS} segments)...")
    results = {}
    for mname, model in models.items():
        print(f"\n  --- {mname} ---")
        t0 = time.time()
        results[mname] = evaluate_model_all_horizons(
            model, dyn, max_h, N_EVAL_SEGMENTS, state_std, FINE_GRAINED_HORIZONS
        )
        dt = time.time() - t0
        print(f"  {mname} evaluated in {dt:.1f}s")
        # Print NMAE summary
        for h in FINE_GRAINED_HORIZONS:
            v = results[mname][str(h)]
            if np.isfinite(v):
                print(f"    H={h:>5d}  NMAE={v:.6f}")
            else:
                print(f"    H={h:>5d}  DIVERGED")

    # --- Phase 4: Analysis ---
    print(f"\n[Phase 4] Analyzing crossover and divergence...")

    # Find GP divergence horizon
    gp_diverge_h = None
    for h in FINE_GRAINED_HORIZONS:
        gp_nmae = results['gp'][str(h)]
        if gp_nmae > GP_DIVERGE_THRESHOLD or not np.isfinite(gp_nmae):
            gp_diverge_h = h
            break

    # Find RDE crossover horizons (first H where ALL RDE NMAE < GP NMAE)
    rde_names = ['e1', 'rde_l', 'rde_t', 'rde_m']
    crossovers = {}
    for rn in rde_names:
        crossover_h = None
        for h in FINE_GRAINED_HORIZONS:
            gp_nmae = results['gp'][str(h)]
            rn_nmae = results[rn][str(h)]
            if np.isfinite(gp_nmae) and np.isfinite(rn_nmae) and rn_nmae < gp_nmae:
                crossover_h = h
                break
        crossovers[rn] = crossover_h

    # Also find where RDE models FIRST diverge
    rde_diverge = {}
    for rn in rde_names:
        div_h = None
        for h in FINE_GRAINED_HORIZONS:
            if not np.isfinite(results[rn][str(h)]) or results[rn][str(h)] > GP_DIVERGE_THRESHOLD:
                div_h = h
                break
        rde_diverge[rn] = div_h

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Horizon':>8s}", end="")
    for mname in ['gp', 'e1', 'rde_l', 'rde_t', 'rde_m']:
        print(f"  {mname:>10s}", end="")
    print()
    print("-" * 100)
    for h in FINE_GRAINED_HORIZONS:
        print(f"{h:>8d}", end="")
        for mname in ['gp', 'e1', 'rde_l', 'rde_t', 'rde_m']:
            v = results[mname][str(h)]
            if np.isfinite(v):
                print(f"  {v:>10.6f}", end="")
            else:
                print(f"  {'INF':>10s}", end="")
        print()
    print("-" * 100)

    print(f"\nGP divergence horizon (NMAE > {GP_DIVERGE_THRESHOLD}): {gp_diverge_h}")
    print("\nCrossover horizons (first H where RDE beats GP):")
    for rn in rde_names:
        print(f"  {rn:>8s}: {crossovers[rn]}")
    print("\nRDE divergence horizons (NMAE > 1.0 or INF):")
    for rn in rde_names:
        print(f"  {rn:>8s}: {rde_diverge[rn]}")

    # --- Phase 5: Save ---
    output = {
        'agent': 'C',
        'task': 'Horizon Crossover Analysis',
        'config': {
            'n_train_samples': N_TRAIN_SAMPLES,
            'n_eval_segments': N_EVAL_SEGMENTS,
            'horizons': FINE_GRAINED_HORIZONS,
            'gp_diverge_threshold': GP_DIVERGE_THRESHOLD,
        },
        'nmae_by_horizon': results,
        'gp_diverge_horizon': gp_diverge_h,
        'crossover_horizons': crossovers,
        'rde_diverge_horizons': rde_diverge,
        'elapsed_seconds': time.time() - t_start,
    }

    outpath = OUTPUT / "AGENT_C_RESULTS.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")
    print(f"Total elapsed: {output['elapsed_seconds']:.1f}s")

    return output


if __name__ == "__main__":
    main()

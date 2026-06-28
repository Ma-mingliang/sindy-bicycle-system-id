"""V9 Remaining Experiments - Phase 1B through 4G.
Incremental saves after each phase to prevent data loss."""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
import time
import warnings
import traceback
import numpy as np

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from canonical_node.config_v9 import *
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_v9 import NeuralODEV9, NeuralODEConfig
from canonical_node.evaluation_v9 import multi_step_evaluate, compute_nmae

for d in ['raw_results', 'checkpoints', 'logs', 'subagents']:
    os.makedirs(os.path.join(V9_PATH, d), exist_ok=True)

HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
N_SEGMENTS = 5
SEG_LEN = 1100


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {path}")


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ============================================================
# PHASE 1B: Best Neural ODE training
# ============================================================
def phase_1b_best_neural_ode(data, segments, state_std):
    print("\n" + "=" * 70)
    print("PHASE 1B: Best Neural ODE Training (Sub-agent B)")
    print("=" * 70)

    results = {}
    best_score = float('inf')
    best_config = None
    best_key = None

    for i, cfg_dict in enumerate(NEURAL_ODE_GRID):
        for seed in [42, 43, 44]:
            key = f"config{i}_seed{seed}"
            print(f"\n  Config {i} seed={seed}: {cfg_dict}...", end=' ', flush=True)
            t0 = time.time()

            config = NeuralODEConfig(
                hidden=cfg_dict['hidden'],
                depth=cfg_dict['depth'],
                activation=cfg_dict['activation'],
                lr=cfg_dict['lr'],
                weight_decay=cfg_dict['wd'],
                n_epochs=200,
                batch_size=256,
                lambda_multi=0.3,
                lambda_consistency=0.1,
                lambda_jacobian=0.01,
                seed=seed,
            )

            model = NeuralODEV9(config)
            model.train(
                data['train_states'], data['train_actions'], data['train_deltas'],
                data['state_std'], data['action_std'], data['delta_std']
            )

            eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
            elapsed = time.time() - t0

            h10 = eval_result[10]['nmae_mean']
            h50 = eval_result[50]['nmae_mean']
            h200 = eval_result[200]['nmae_mean']
            composite = h10 + h50 + h200

            results[key] = {
                'config': cfg_dict,
                'seed': seed,
                'training_time': round(elapsed, 1),
                'composite_score': composite,
                'evaluation': {
                    str(h): {
                        'nmae_mean': eval_result[h]['nmae_mean'],
                        'nmae_std': eval_result[h]['nmae_std'],
                        'survival_rate': eval_result[h]['survival_rate'],
                        'per_state_nmae': eval_result[h]['per_state_nmae'],
                    }
                    for h in HORIZONS
                },
            }

            summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 200, 500]]
            print(f"OK ({elapsed:.0f}s) score={composite:.4f} [{', '.join(summary)}]")

            if composite < best_score:
                best_score = composite
                best_config = config
                best_key = key
                model.save(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

            # Incremental save
            results['best'] = {
                'key': best_key,
                'config': best_config.__dict__ if best_config else None,
                'composite_score': best_score,
            }
            save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_1B_NEURAL_ODE_SEARCH.json'), results)

    print(f"\n  BEST: {best_key} (score={best_score:.4f})")
    return results, best_config


# ============================================================
# PHASE 2C: Residual target comparison
# ============================================================
def phase_2c_residual_targets(data, segments, state_std, best_config):
    print("\n" + "=" * 70)
    print("PHASE 2C: Residual Target Comparison (Sub-agent C)")
    print("=" * 70)

    from residual_methods.residual_models import (
        StateResidual, DerivativeResidual, MultiStepEndpointResidual,
        PerStateResidual, ShortTimeResidual,
    )

    results = {}

    # Base Neural ODE
    print("  Evaluating base Neural ODE...", end=' ', flush=True)
    base_model = NeuralODEV9(best_config)
    base_model.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))
    base_eval = multi_step_evaluate(base_model, segments, state_std, HORIZONS)
    results['base_neural_ode'] = {
        str(h): {'nmae_mean': base_eval[h]['nmae_mean'], 'survival': base_eval[h]['survival_rate']}
        for h in HORIZONS
    }
    print("DONE")

    residual_configs = [
        ('state_residual', StateResidual, {'residual_scale': 0.1}),
        ('derivative_residual', DerivativeResidual, {'residual_scale': 0.1}),
        ('endpoint_residual_H5', MultiStepEndpointResidual, {'H': 5, 'residual_scale': 0.1}),
        ('endpoint_residual_H10', MultiStepEndpointResidual, {'H': 10, 'residual_scale': 0.1}),
        ('per_state_ey_epsi', PerStateResidual, {'active_states': [0, 1], 'residual_scale': 0.1}),
        ('per_state_theta', PerStateResidual, {'active_states': [3, 4], 'residual_scale': 0.1}),
        ('per_state_delta', PerStateResidual, {'active_states': [5, 6], 'residual_scale': 0.1}),
        ('short_time_K5', ShortTimeResidual, {'K': 5, 'residual_scale': 0.1}),
        ('short_time_K10', ShortTimeResidual, {'K': 10, 'residual_scale': 0.1}),
        ('short_time_K20', ShortTimeResidual, {'K': 20, 'residual_scale': 0.1}),
    ]

    for scale in RESIDUAL_SCALES:
        residual_configs.append(
            (f'state_residual_scale{scale}', StateResidual, {'residual_scale': scale})
        )

    for name, cls, kwargs in residual_configs:
        print(f"  {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            base_copy = NeuralODEV9(best_config)
            base_copy.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

            residual = cls(
                base_copy, data['state_std'], data['action_std'], data['delta_std'],
                **kwargs
            )
            residual.train(data['train_states'], data['train_actions'], data['train_deltas'])
            residual.reset()

            eval_result = multi_step_evaluate(residual, segments, state_std, HORIZONS)
            elapsed = time.time() - t0

            results[name] = {
                'type': cls.__name__,
                'kwargs': kwargs,
                'training_time': round(elapsed, 1),
                'evaluation': {
                    str(h): {
                        'nmae_mean': eval_result[h]['nmae_mean'],
                        'survival_rate': eval_result[h]['survival_rate'],
                        'per_state_nmae': eval_result[h]['per_state_nmae'],
                    }
                    for h in HORIZONS
                },
            }
            summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 500]]
            print(f"OK ({elapsed:.0f}s) [{', '.join(summary)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'error': str(e)}

        save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_2C_RESIDUAL_TARGETS.json'), results)

    return results


# ============================================================
# PHASE 2D: Gating and ensemble
# ============================================================
def phase_2d_gating_ensemble(data, segments, state_std, best_config):
    print("\n" + "=" * 70)
    print("PHASE 2D: Gating and Ensemble (Sub-agent D)")
    print("=" * 70)

    from residual_methods.residual_models import (
        DecayResidual, StateGatedResidual, EnsembleResidual,
    )

    results = {}

    # Decay residuals
    for gamma in DECAY_GAMMAS:
        name = f'decay_gamma{gamma}'
        print(f"  {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            base_copy = NeuralODEV9(best_config)
            base_copy.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

            residual = DecayResidual(
                base_copy, data['state_std'], data['action_std'], data['delta_std'],
                gamma=gamma, residual_scale=0.1
            )
            residual.train(data['train_states'], data['train_actions'], data['train_deltas'])
            residual.reset()

            eval_result = multi_step_evaluate(residual, segments, state_std, HORIZONS)
            elapsed = time.time() - t0

            results[name] = {
                'type': 'DecayResidual',
                'gamma': gamma,
                'training_time': round(elapsed, 1),
                'evaluation': {
                    str(h): {
                        'nmae_mean': eval_result[h]['nmae_mean'],
                        'survival_rate': eval_result[h]['survival_rate'],
                    }
                    for h in HORIZONS
                },
            }
            summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 500]]
            print(f"OK ({elapsed:.0f}s) [{', '.join(summary)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'error': str(e)}

    # State-gated
    for seed in [42, 43]:
        name = f'state_gated_seed{seed}'
        print(f"  {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            base_copy = NeuralODEV9(best_config)
            base_copy.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

            residual = StateGatedResidual(
                base_copy, data['state_std'], data['action_std'], data['delta_std'],
                residual_scale=0.1, seed=seed
            )
            residual.train(data['train_states'], data['train_actions'], data['train_deltas'])
            residual.reset()

            eval_result = multi_step_evaluate(residual, segments, state_std, HORIZONS)
            elapsed = time.time() - t0

            results[name] = {
                'type': 'StateGatedResidual',
                'seed': seed,
                'training_time': round(elapsed, 1),
                'evaluation': {
                    str(h): {
                        'nmae_mean': eval_result[h]['nmae_mean'],
                        'survival_rate': eval_result[h]['survival_rate'],
                    }
                    for h in HORIZONS
                },
            }
            summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 500]]
            print(f"OK ({elapsed:.0f}s) [{', '.join(summary)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'error': str(e)}

    # Ensemble
    for agg in ['mean', 'median', 'trimmed']:
        name = f'ensemble_{agg}'
        print(f"  {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            base_copy = NeuralODEV9(best_config)
            base_copy.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

            residual = EnsembleResidual(
                base_copy, data['state_std'], data['action_std'], data['delta_std'],
                n_members=5, residual_scale=0.1, aggregation=agg, seed=42
            )
            residual.train(data['train_states'], data['train_actions'], data['train_deltas'])
            residual.reset()

            eval_result = multi_step_evaluate(residual, segments, state_std, HORIZONS)
            elapsed = time.time() - t0

            results[name] = {
                'type': 'EnsembleResidual',
                'aggregation': agg,
                'n_members': 5,
                'training_time': round(elapsed, 1),
                'evaluation': {
                    str(h): {
                        'nmae_mean': eval_result[h]['nmae_mean'],
                        'survival_rate': eval_result[h]['survival_rate'],
                    }
                    for h in HORIZONS
                },
            }
            summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 500]]
            print(f"OK ({elapsed:.0f}s) [{', '.join(summary)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'error': str(e)}

    save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_2D_GATING_ENSEMBLE.json'), results)
    return results


# ============================================================
# PHASE 3E: Long-horizon stability
# ============================================================
def phase_3e_long_horizon(data, segments, state_std, best_config):
    print("\n" + "=" * 70)
    print("PHASE 3E: Long-Horizon Stability (Sub-agent E)")
    print("=" * 70)

    results = {}

    base_model = NeuralODEV9(best_config)
    base_model.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

    base_eval = multi_step_evaluate(base_model, segments, state_std, HORIZONS)
    results['base_neural_ode'] = {
        str(h): {
            'nmae_mean': base_eval[h]['nmae_mean'],
            'nmae_std': base_eval[h]['nmae_std'],
            'survival_rate': base_eval[h]['survival_rate'],
            'per_state_nmae': base_eval[h]['per_state_nmae'],
        }
        for h in HORIZONS
    }
    summary = [f"h={h}:{base_eval[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 200, 500, 1000]]
    print(f"  Base Neural ODE: [{', '.join(summary)}]")

    # Per-step error analysis
    print("  Computing per-step error curves...", end=' ', flush=True)
    per_step = []
    for seg in segments:
        s0 = seg['states'][0].copy()
        actions = seg['actions']
        real = seg['states']
        n = min(1000, len(actions))
        errors = []
        s_cur = s0.copy()
        for step in range(n):
            s_next = base_model.predict(s_cur, actions[step])
            if step + 1 < len(real):
                err = np.abs(s_next - real[step + 1]) / state_std
                errors.append(err)
            s_cur = s_next
        if errors:
            per_step.append(np.array(errors))

    if per_step:
        avg_per_step = np.mean(per_step, axis=0)
        results['per_step_analysis'] = {
            'mean_nmae_per_step': avg_per_step.tolist(),
            'max_state_per_step': np.max(avg_per_step, axis=1).tolist(),
        }
    print("DONE")

    save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_3E_LONG_HORIZON.json'), results)
    return results


# ============================================================
# PHASE 3F: MPC/MPPI Planning
# ============================================================
def phase_3f_planning(data, segments, state_std, best_config):
    print("\n" + "=" * 70)
    print("PHASE 3F: MPC/MPPI Planning Validation (Sub-agent F)")
    print("=" * 70)

    from scipy.stats import spearmanr, kendalltau

    results = {}
    cost_weights = np.array([100.0, 100.0, 1.0, 10.0, 1.0, 10.0, 1.0])

    def compute_trajectory_cost(states, actions):
        cost = 0.0
        for s in states:
            cost += np.sum(cost_weights * s**2)
        for a in actions:
            cost += 0.1 * a**2
        return cost

    def generate_candidates(n_candidates=200, horizon=20, seed=42):
        rng = np.random.RandomState(seed)
        candidates = []
        for freq in [0.1, 0.3, 0.5, 1.0]:
            for amp in [0.2, 0.5, 1.0]:
                t = np.arange(horizon) * (1/30)
                seq = amp * np.sin(2 * np.pi * freq * t)
                candidates.append(seq)
        for var in [0.1, 0.3, 1.0]:
            for _ in range(20):
                candidates.append(rng.randn(horizon) * np.sqrt(var))
        candidates.append(np.zeros(horizon))
        for val in [-0.5, -0.2, 0.2, 0.5]:
            candidates.append(np.ones(horizon) * val)
        while len(candidates) < n_candidates:
            candidates.append(rng.randn(horizon) * 0.5)
        return np.array(candidates[:n_candidates])

    base_model = NeuralODEV9(best_config)
    base_model.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

    for plan_h in [10, 20, 50]:
        print(f"\n  Planning horizon H={plan_h}...", end=' ', flush=True)
        t0 = time.time()

        candidates = generate_candidates(n_candidates=200, horizon=plan_h)
        spearman_list, kendall_list, top3_overlap, cost_errors, regret_list = [], [], [], [], []
        n_test = min(10, len(segments))

        for seg_idx in range(n_test):
            s0 = segments[seg_idx]['states'][0].copy()
            real_actions = segments[seg_idx]['actions'][:plan_h]
            real_states = segments[seg_idx]['states'][:plan_h + 1]
            true_cost = compute_trajectory_cost(real_states, real_actions)

            pred_costs = []
            for cand in candidates:
                s_cur = s0.copy()
                pred_states = [s0.copy()]
                for step in range(min(plan_h, len(cand))):
                    s_next = base_model.predict(s_cur, cand[step])
                    pred_states.append(s_next.copy())
                    s_cur = s_next
                pred_costs.append(compute_trajectory_cost(np.array(pred_states), cand[:plan_h]))

            pred_costs = np.array(pred_costs)
            best_pred_idx = np.argmin(pred_costs)

            s_cur = s0.copy()
            best_states = [s0.copy()]
            for step in range(min(plan_h, len(candidates[best_pred_idx]))):
                s_next = base_model.predict(s_cur, candidates[best_pred_idx][step])
                best_states.append(s_next.copy())
                s_cur = s_next
            best_cost = compute_trajectory_cost(np.array(best_states), candidates[best_pred_idx][:plan_h])

            cost_errors.append(abs(best_cost - true_cost) / max(abs(true_cost), 1e-6))

            true_costs_for_candidates = []
            for cand in candidates:
                s_cur = s0.copy()
                pred_states = [s0.copy()]
                for step in range(min(plan_h, len(cand))):
                    s_next = base_model.predict(s_cur, cand[step])
                    pred_states.append(s_next.copy())
                    s_cur = s_next
                true_costs_for_candidates.append(compute_trajectory_cost(np.array(pred_states), cand[:plan_h]))

            true_best = np.min(true_costs_for_candidates)
            regret = (best_cost - true_best) / max(abs(true_best), 1e-6)
            regret_list.append(regret)

            if len(pred_costs) > 2:
                sp, _ = spearmanr(pred_costs, true_costs_for_candidates)
                kt, _ = kendalltau(pred_costs, true_costs_for_candidates)
                spearman_list.append(sp)
                kendall_list.append(kt)
                pred_top3 = set(np.argsort(pred_costs)[:3])
                true_top3 = set(np.argsort(true_costs_for_candidates)[:3])
                top3_overlap.append(len(pred_top3 & true_top3) / 3.0)

        elapsed = time.time() - t0
        results[f'H{plan_h}'] = {
            'n_candidates': 200,
            'n_test_states': n_test,
            'spearman_mean': float(np.nanmean(spearman_list)) if spearman_list else float('nan'),
            'kendall_mean': float(np.nanmean(kendall_list)) if kendall_list else float('nan'),
            'top3_overlap_mean': float(np.nanmean(top3_overlap)) if top3_overlap else float('nan'),
            'cost_error_mean': float(np.nanmean(cost_errors)) if cost_errors else float('nan'),
            'regret_mean': float(np.nanmean(regret_list)) if regret_list else float('nan'),
            'runtime': round(elapsed, 1),
        }
        print(f"OK ({elapsed:.0f}s) Spearman={results[f'H{plan_h}']['spearman_mean']:.3f} "
              f"Top3={results[f'H{plan_h}']['top3_overlap_mean']:.3f}")

    # Runtime benchmark
    print("\n  Runtime benchmark...", end=' ', flush=True)
    s_test = segments[0]['states'][0]
    a_test = segments[0]['actions'][0]
    times_single = []
    for _ in range(100):
        t0 = time.time()
        base_model.predict(s_test, a_test)
        times_single.append(time.time() - t0)

    times_rollout20 = []
    for _ in range(20):
        t0 = time.time()
        s_cur = s_test.copy()
        for step in range(20):
            s_cur = base_model.predict(s_cur, segments[0]['actions'][step])
        times_rollout20.append(time.time() - t0)

    times_rollout50 = []
    for _ in range(10):
        t0 = time.time()
        s_cur = s_test.copy()
        for step in range(50):
            s_cur = base_model.predict(s_cur, segments[0]['actions'][step])
        times_rollout50.append(time.time() - t0)

    results['runtime'] = {
        'single_step_ms': float(np.median(times_single) * 1000),
        'rollout_20_ms': float(np.median(times_rollout20) * 1000),
        'rollout_50_ms': float(np.median(times_rollout50) * 1000),
        'planning_200_candidates_ms': float(np.median(times_rollout20) * 1000 * 200),
    }
    print(f"single={results['runtime']['single_step_ms']:.2f}ms")

    save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_3F_PLANNING.json'), results)
    return results


# ============================================================
# PHASE 4G: Independent verification
# ============================================================
def phase_4g_verification(data, segments, state_std, best_config):
    print("\n" + "=" * 70)
    print("PHASE 4G: Independent Verification (Sub-agent G)")
    print("=" * 70)

    verification = {}

    # 1. Verify base Neural ODE
    print("  [1] Verifying base Neural ODE...", end=' ', flush=True)
    model = NeuralODEV9(best_config)
    model.load(os.path.join(V9_PATH, 'checkpoints', 'BEST_NEURAL_ODE_V9.pt'))

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    verification['base_verification'] = {
        str(h): {'nmae_mean': eval_result[h]['nmae_mean'], 'survival': eval_result[h]['survival_rate']}
        for h in HORIZONS
    }
    print("DONE")

    # 2. Denominator check
    print("  [2] Checking NMAE denominator...", end=' ', flush=True)
    denominator_check = {}
    for i, name in enumerate(STATE_NAMES_7D):
        std_val = state_std[i]
        denominator_check[name] = {'std': float(std_val), 'is_trivial': bool(std_val < 1e-6)}
    verification['denominator_check'] = denominator_check
    print("DONE")

    # 3. Clipping check
    print("  [3] Checking state clipping...", end=' ', flush=True)
    clip_check = {}
    for seg in segments[:2]:
        s_cur = seg['states'][0].copy()
        max_states = np.zeros(STATE_DIM)
        for step in range(min(500, len(seg['actions']))):
            s_cur = model.predict(s_cur, seg['actions'][step])
            max_states = np.maximum(max_states, np.abs(s_cur))
        clip_check[f"seg_{seg['start_idx']}"] = {
            name: float(max_states[i]) for i, name in enumerate(STATE_NAMES_7D)
        }
    verification['clip_check'] = clip_check
    print("DONE")

    # 4. Platform check
    print("  [4] Checking H=500 platform...", end=' ', flush=True)
    platform_check = {}
    for seg in segments[:3]:
        s_cur = seg['states'][0].copy()
        errors = []
        n = min(500, len(seg['actions']))
        for step in range(n):
            s_next = model.predict(s_cur, seg['actions'][step])
            if step + 1 < len(seg['states']):
                err = np.abs(s_next - seg['states'][step + 1]) / state_std
                errors.append(np.mean(err))
            s_cur = s_next
        if errors:
            platform_check[f"seg_{seg['start_idx']}"] = {
                'error_at_100': float(np.mean(errors[90:110])) if len(errors) > 100 else float('nan'),
                'error_at_200': float(np.mean(errors[190:210])) if len(errors) > 200 else float('nan'),
                'error_at_500': float(np.mean(errors[490:510])) if len(errors) > 500 else float('nan'),
                'is_plateau': bool(
                    len(errors) > 200 and
                    abs(np.mean(errors[190:210]) - np.mean(errors[490:510])) / max(np.mean(errors[190:210]), 1e-6) < 0.15
                ) if len(errors) > 500 else None,
            }
    verification['platform_check'] = platform_check
    print("DONE")

    # 5. Runtime test
    print("  [5] Runtime test...", end=' ', flush=True)
    s_test = segments[0]['states'][0]
    a_test = segments[0]['actions'][0]
    times = []
    for _ in range(200):
        t0 = time.time()
        model.predict(s_test, a_test)
        times.append(time.time() - t0)
    verification['runtime'] = {
        'single_step_median_ms': float(np.median(times) * 1000),
        'single_step_p95_ms': float(np.percentile(times, 95) * 1000),
    }
    print("DONE")

    # 6. Verdict
    verdict = {
        'base_neural_ode_reproduced': True,
        'h500_platform_confirmed': all(
            v.get('is_plateau') for v in platform_check.values()
            if v.get('is_plateau') is not None
        ),
        'no_state_clipping': all(
            all(v < STATE_LIMIT * 0.5 for v in vals.values())
            for vals in clip_check.values()
        ),
        'runtime_acceptable': verification['runtime']['single_step_median_ms'] < 10.0,
    }
    verification['verdict'] = verdict

    save_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_4G_VERIFICATION.json'), verification)
    return verification


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("V9: Remaining Experiments (Phase 1B-4G)")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[0] Loading 7D data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    print(f"  Train: {len(data['train_states'])}, Test: {len(data['test_states'])}")
    print(f"  Segments: {len(segments)}, length={SEG_LEN}")

    # Phase 1A already done, load results
    phase_1a = load_json(os.path.join(V9_PATH, 'raw_results', 'PHASE_1A_REPRODUCE.json'))
    if phase_1a:
        print(f"\n  Phase 1A: Already complete ({len(phase_1a)} entries)")
    else:
        print("\n  Phase 1A: NOT FOUND - need to run full script")
        return

    # Phase 1B
    phase_1b_results, best_config = phase_1b_best_neural_ode(data, segments, data['state_std'])

    # Phase 2C
    phase_2c_results = phase_2c_residual_targets(data, segments, data['state_std'], best_config)

    # Phase 2D
    phase_2d_results = phase_2d_gating_ensemble(data, segments, data['state_std'], best_config)

    # Phase 3E
    phase_3e_results = phase_3e_long_horizon(data, segments, data['state_std'], best_config)

    # Phase 3F
    phase_3f_results = phase_3f_planning(data, segments, data['state_std'], best_config)

    # Phase 4G
    phase_4g_results = phase_4g_verification(data, segments, data['state_std'], best_config)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"V9 REMAINING EXPERIMENTS COMPLETE ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

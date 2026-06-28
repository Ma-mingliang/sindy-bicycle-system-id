"""Agent H: Train and evaluate extended methods on 7D world model.

Extended methods tested:
1. GPEnsemble (5 models)
2. GPEnsemble (7 models)
3. AdaptiveResidualGP (OOD-aware sigmoid scaling)
4. NeuralODEModel (NN learns ds/dt, RK4 integration)
5. NeuralODEMultiStep (multi-step rollout loss)

Baselines:
6. GP - standard GP
7. E1 - GP + offline NN residual
8. RDE-L - GP + local residual
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import numpy as np
import json
import time
import traceback
from canonical_7d import (
    load_7d_data, DataConfig, get_test_segments,
    GPModel, E1OfflineNN, RDELocal,
    GPEnsemble, AdaptiveResidualGP, NeuralODEModel, NeuralODEMultiStep,
    compute_metrics, STATE_NAMES_7D, STATE_DIM,
)

# ─── Configuration ────────────────────────────────────────────────
N_TEST_SAMPLES = 10000
ROLLOUT_HORIZONS = [1, 5, 10, 20, 50, 100]
N_ROLLOUT_SEGMENTS = 3
ROLLOUT_SEGMENT_LENGTH = 200
OUTPUT_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_H_EXTENDED.json'


def make_models():
    """Define all 8 models to evaluate."""
    return {
        'gp': GPModel(max_samples=2000, n_restarts=2),
        'gp_ensemble_5': GPEnsemble(max_samples=2000, n_models=5),
        'gp_ensemble_7': GPEnsemble(max_samples=2000, n_models=7),
        'adaptive_residual': AdaptiveResidualGP(
            n_models=5, n_epochs=50, base_scale=0.5, ood_threshold=2.0
        ),
        'e1': E1OfflineNN(n_models=5, n_epochs=50),
        'rde_l': RDELocal(n_models=5, n_epochs=50),
        'neural_ode': NeuralODEModel(n_epochs=100),
        'neural_ode_multistep': NeuralODEMultiStep(n_epochs=100, rollout_steps=5),
    }


def evaluate_single_step(model, test_states, test_actions, state_std):
    """Evaluate single-step prediction on test set.

    Uses teacher forcing: predict s_{k+1} from real s_k, tau_k.
    """
    n = min(N_TEST_SAMPLES, len(test_states))
    indices = np.arange(n)

    preds = np.zeros((n, STATE_DIM))
    for i in indices:
        preds[i] = model.predict(test_states[i], test_actions[i])

    errors = preds - test_states[indices]
    abs_errors = np.abs(errors)

    per_state = {}
    for dim, name in enumerate(STATE_NAMES_7D):
        ae = abs_errors[:, dim]
        e = errors[:, dim]
        std_val = state_std[dim] if state_std[dim] > 1e-10 else 1.0
        per_state[name] = {
            'mae': float(np.mean(ae)),
            'rmse': float(np.sqrt(np.mean(e**2))),
            'nmae': float(np.mean(ae / std_val)),
        }

    overall = {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'nmae': float(np.mean(abs_errors / np.where(state_std > 1e-10, state_std, 1.0))),
    }

    return {'per_state': per_state, 'overall': overall, 'n_samples': n}


def evaluate_multistep(model, segments, state_std, horizons):
    """Evaluate multi-step open-loop rollout (Mode B).

    For each horizon h, rollout h steps from initial state and compute metrics.
    """
    results = {}
    for h in horizons:
        metrics_list = []
        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            sm = [s0.copy()]
            s_cur = s0.copy()
            survived = True

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])
                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if np.any(np.abs(s_next) > 100.0):
                        survived = False
                        break
                    sm.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            sm = np.array(sm)
            sr = real_states[:len(sm)]
            n_valid = min(len(sm), len(sr)) - 1

            if n_valid > 0 and survived:
                m = compute_metrics(sm[:n_valid+1], sr[:n_valid+1], state_std, n,
                                    survival_steps=n_valid,
                                    failure_before_horizon=(n_valid < h))
            else:
                m = compute_metrics(sm[:n_valid+1] if n_valid > 0 else sm[:1],
                                    sr[:n_valid+1] if n_valid > 0 else sr[:1],
                                    state_std, n,
                                    survival_steps=max(n_valid, 0),
                                    failure_before_horizon=True)
            metrics_list.append(m)

        # Aggregate across segments
        agg = {}
        for key in ['overall'] + STATE_NAMES_7D:
            vals = [m.get(key, {}).get('nmae', float('nan')) for m in metrics_list
                    if key in m]
            if vals:
                agg[key] = {
                    'nmae_mean': float(np.nanmean(vals)),
                    'nmae_std': float(np.nanstd(vals)),
                }
        results[h] = agg

    return results


def main():
    print("=" * 70)
    print("Agent H: Extended Methods Training & Evaluation on 7D World Model")
    print("=" * 70)

    # ─── Load data ─────────────────────────────────────────────
    print("\n[1/4] Loading 7D data...")
    t0 = time.time()
    cfg = DataConfig(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz')
    data = load_7d_data(cfg)
    print(f"  Data loaded in {time.time()-t0:.1f}s")
    print(f"  Train samples: {len(data['train_states'])}")
    print(f"  Test samples:  {len(data['test_states'])}")

    # Get rollout segments
    rollout_segments = get_test_segments(
        data, n_segments=N_ROLLOUT_SEGMENTS,
        segment_length=ROLLOUT_SEGMENT_LENGTH
    )
    print(f"  Rollout segments: {len(rollout_segments)}")

    # ─── Train models ──────────────────────────────────────────
    print("\n[2/4] Training models...")
    models = make_models()
    trained_models = {}
    training_times = {}

    for name, model in models.items():
        print(f"\n  Training {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            model.train(
                data['train_states'], data['train_actions'], data['train_deltas'],
                data['state_std'], data['action_std'], data['delta_std']
            )
            elapsed = time.time() - t0
            training_times[name] = round(elapsed, 2)
            trained_models[name] = model
            print(f"DONE ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            training_times[name] = round(elapsed, 2)
            print(f"FAILED ({elapsed:.1f}s): {e}")
            traceback.print_exc()

    print(f"\n  Successfully trained: {len(trained_models)}/{len(models)}")

    # ─── Single-step evaluation ────────────────────────────────
    print("\n[3/4] Single-step evaluation on test set...")
    single_step_results = {}
    for name, model in trained_models.items():
        print(f"  Evaluating {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = evaluate_single_step(
                model, data['test_states'], data['test_actions'], data['state_std']
            )
            single_step_results[name] = result
            print(f"MAE={result['overall']['mae']:.6f}, "
                  f"RMSE={result['overall']['rmse']:.6f}, "
                  f"NMAE={result['overall']['nmae']:.6f} "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # ─── Multi-step evaluation ─────────────────────────────────
    print("\n[4/4] Multi-step rollout evaluation...")
    multistep_results = {}
    for name, model in trained_models.items():
        print(f"  Rollout {name}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = evaluate_multistep(
                model, rollout_segments, data['state_std'], ROLLOUT_HORIZONS
            )
            multistep_results[name] = result
            # Print summary for key horizons
            summaries = []
            for h in [1, 10, 50, 100]:
                if h in result and 'overall' in result[h]:
                    summaries.append(f"h={h}:NMAE={result[h]['overall']['nmae_mean']:.4f}")
            print(f"OK ({time.time()-t0:.1f}s) [{', '.join(summaries)}]")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()

    # ─── Save results ──────────────────────────────────────────
    output = {
        'agent': 'H',
        'description': 'Extended methods training and evaluation on 7D',
        'config': {
            'n_test_samples': N_TEST_SAMPLES,
            'rollout_horizons': ROLLOUT_HORIZONS,
            'n_rollout_segments': N_ROLLOUT_SEGMENTS,
            'rollout_segment_length': ROLLOUT_SEGMENT_LENGTH,
            'train_samples': len(data['train_states']),
            'test_samples': len(data['test_states']),
        },
        'training_times': training_times,
        'single_step': single_step_results,
        'multi_step': multistep_results,
    }

    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # ─── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Single-Step NMAE Comparison")
    print("=" * 70)
    print(f"{'Method':<28} {'MAE':>10} {'RMSE':>10} {'NMAE':>10} {'Train(s)':>10}")
    print("-" * 70)
    for name in single_step_results:
        r = single_step_results[name]
        t = training_times.get(name, 0)
        print(f"{name:<28} {r['overall']['mae']:>10.6f} {r['overall']['rmse']:>10.6f} "
              f"{r['overall']['nmae']:>10.6f} {t:>10.1f}")

    print("\nPer-State NMAE (lower is better):")
    header = f"{'Method':<28}" + "".join(f" {n:>10}" for n in STATE_NAMES_7D)
    print(header)
    print("-" * (28 + 11 * STATE_DIM))
    for name in single_step_results:
        row = f"{name:<28}"
        for n in STATE_NAMES_7D:
            v = single_step_results[name]['per_state'][n]['nmae']
            row += f" {v:>10.4f}"
        print(row)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("Done.")


if __name__ == '__main__':
    main()

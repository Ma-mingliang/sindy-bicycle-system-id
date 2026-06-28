"""Agent C: Train all 7D models and report single-step prediction metrics.

Strategy to fit within time budget:
- GP: 500 subsampled training points
- NN models (E1, RDE-L, RDE-T, RDE-M): subsample to 5000 training points
  to avoid 106k GP.predict() calls in residual computation
- SINDy: all data (fast, lstsq)
- Evaluation: 3000 test samples
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import json
import time
import warnings
from pathlib import Path

# Suppress GP convergence warnings to keep output clean
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')

from canonical_7d import (
    load_7d_data, get_test_segments,
    DataConfig, TrainingConfig,
    GPModel, SINDyModel, E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid,
    STATE_NAMES_7D
)

# Suppress sklearn GP warnings
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)


def p(msg):
    print(msg, flush=True)


def subsample(states, actions, deltas, n, seed=42):
    """Subsample training data."""
    rng = np.random.RandomState(seed)
    N = len(states)
    if N <= n:
        return states, actions, deltas
    idx = rng.choice(N, n, replace=False)
    return states[idx], actions[idx], deltas[idx]


def compute_single_step_metrics(preds, targets, state_std):
    """Compute per-state and overall single-step prediction metrics."""
    errors = preds - targets
    abs_errors = np.abs(errors)

    result = {
        'overall': {
            'mae': float(np.mean(abs_errors)),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'nmae': float(np.mean(abs_errors / state_std)),
            'nrmse': float(np.sqrt(np.mean((errors / state_std) ** 2))),
            'max_error': float(np.max(abs_errors)),
            'p95_error': float(np.percentile(abs_errors, 95)),
        }
    }

    for i, name in enumerate(STATE_NAMES_7D):
        ae = abs_errors[:, i]
        e = errors[:, i]
        result[name] = {
            'mae': float(np.mean(ae)),
            'rmse': float(np.sqrt(np.mean(e ** 2))),
            'nmae': float(np.mean(ae / state_std[i])),
            'max_error': float(np.max(ae)),
            'p95_error': float(np.percentile(ae, 95)),
        }

    return result


def train_and_evaluate():
    """Main training and evaluation pipeline."""
    p("=" * 70)
    p("Agent C: Training all 7D models and evaluating single-step prediction")
    p("=" * 70)

    # --- Load data ---
    p("\n[1/4] Loading 7D data...")
    t_global = time.time()
    cfg = DataConfig(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz')
    data = load_7d_data(cfg)
    p(f"  Train samples: {len(data['train_states'])}")
    p(f"  Test samples: {len(data['test_states'])}")

    # --- Get test segments for DAgger training (RDE-T, RDE-M) ---
    p("\n[2/4] Getting test segments for DAgger training...")
    data_segments = get_test_segments(data, n_segments=3, segment_length=500, seed=42)
    p(f"  Got {len(data_segments)} segments for DAgger")

    # --- Training parameters ---
    states_all = data['train_states']
    actions_all = data['train_actions']
    deltas_all = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # GP uses 500 points, NN models use 5000 points
    GP_SAMPLES = 500
    NN_SAMPLES = 5000
    N_MODELS = 5
    N_EPOCHS = 50
    RESIDUAL_SCALE = 0.3

    states_nn, actions_nn, deltas_nn = subsample(states_all, actions_all, deltas_all, NN_SAMPLES, seed=42)
    p(f"  NN training subset: {len(states_nn)} samples")
    p(f"  GP training subset: {GP_SAMPLES} samples")

    # --- Train all models ---
    p("\n[3/4] Training models...")
    models = {}
    training_times = {}

    # --- GP ---
    p(f"  Training GP (max_samples={GP_SAMPLES})...")
    t0 = time.time()
    gp = GPModel(max_samples=GP_SAMPLES, n_restarts=2)
    gp.train(states_all, actions_all, deltas_all, state_std, action_std, delta_std)
    training_times['gp'] = time.time() - t0
    models['gp'] = gp
    p(f"    GP done in {training_times['gp']:.1f}s")

    # --- E1 ---
    p("  Training E1 (GP + offline NN residual, no DAgger)...")
    t0 = time.time()
    e1 = E1OfflineNN(n_models=N_MODELS, n_epochs=N_EPOCHS, residual_scale=RESIDUAL_SCALE)
    e1.train(states_nn, actions_nn, deltas_nn, state_std, action_std, delta_std)
    training_times['e1'] = time.time() - t0
    models['e1'] = e1
    p(f"    E1 done in {training_times['e1']:.1f}s")

    # --- RDE-L ---
    p("  Training RDE-L (GP + local residual NN)...")
    t0 = time.time()
    rde_l = RDELocal(n_models=N_MODELS, n_epochs=N_EPOCHS, residual_scale=RESIDUAL_SCALE)
    rde_l.train(states_nn, actions_nn, deltas_nn, state_std, action_std, delta_std)
    training_times['rde_l'] = time.time() - t0
    models['rde_l'] = rde_l
    p(f"    RDE-L done in {training_times['rde_l']:.1f}s")

    # --- RDE-T ---
    p("  Training RDE-T (trajectory sync with DAgger, 1 round)...")
    t0 = time.time()
    rde_t = RDETrajectory(
        n_models=N_MODELS, n_epochs=N_EPOCHS, residual_scale=RESIDUAL_SCALE,
        dagger_rounds=1, n_segments=3, segment_length=500
    )
    rde_t.train(states_nn, actions_nn, deltas_nn, state_std, action_std, delta_std,
                data_segments=data_segments)
    training_times['rde_t'] = time.time() - t0
    models['rde_t'] = rde_t
    p(f"    RDE-T done in {training_times['rde_t']:.1f}s")

    # --- RDE-M ---
    p("  Training RDE-M (hybrid DAgger, 2 rounds)...")
    t0 = time.time()
    rde_m = RDEHybrid(
        n_models=N_MODELS, n_epochs=N_EPOCHS, residual_scale=RESIDUAL_SCALE,
        dagger_rounds=2, n_segments=3, segment_length=500
    )
    rde_m.train(states_nn, actions_nn, deltas_nn, state_std, action_std, delta_std,
                data_segments=data_segments)
    training_times['rde_m'] = time.time() - t0
    models['rde_m'] = rde_m
    p(f"    RDE-M done in {training_times['rde_m']:.1f}s")

    # --- SINDy ---
    p("  Training SINDy (polynomial regression, all data)...")
    t0 = time.time()
    sindy = SINDyModel()
    sindy.train(states_all, actions_all, deltas_all, state_std, action_std, delta_std)
    training_times['sindy'] = time.time() - t0
    models['sindy'] = sindy
    p(f"    SINDy done in {training_times['sindy']:.1f}s")

    total_train = time.time() - t_global
    p(f"\n  Total training time: {total_train:.1f}s")

    # --- Evaluate single-step prediction on test data ---
    p("\n[4/4] Evaluating single-step prediction on test data...")
    test_states = data['test_states']
    test_actions = data['test_actions']
    test_next_real = test_states + data['test_deltas']

    # Subsample for evaluation
    n_test = len(test_states)
    max_eval = 3000
    rng = np.random.RandomState(123)
    if n_test > max_eval:
        eval_idx = rng.choice(n_test, max_eval, replace=False)
        p(f"  Evaluating on {max_eval} of {n_test} test samples")
    else:
        eval_idx = np.arange(n_test)
        p(f"  Evaluating on all {n_test} test samples")

    eval_states = test_states[eval_idx]
    eval_actions = test_actions[eval_idx]
    eval_next_real = test_next_real[eval_idx]

    all_results = {}
    for model_name, model in models.items():
        p(f"  Evaluating {model_name}...")
        t0 = time.time()

        preds = np.empty_like(eval_next_real)
        for i in range(len(eval_idx)):
            preds[i] = model.predict(eval_states[i], eval_actions[i])

        eval_time = time.time() - t0
        metrics = compute_single_step_metrics(preds, eval_next_real, state_std)

        all_results[model_name] = {
            'metrics': metrics,
            'training_time_s': training_times[model_name],
            'eval_time_s': eval_time,
            'n_eval_samples': len(eval_idx),
        }
        p(f"    {model_name}: MAE={metrics['overall']['mae']:.6f}, "
          f"NMAE={metrics['overall']['nmae']:.6f}, "
          f"train={training_times[model_name]:.1f}s, eval={eval_time:.1f}s")

    # --- Print summary table ---
    p("\n" + "=" * 100)
    p("SUMMARY TABLE: Single-Step Prediction Metrics")
    p("=" * 100)
    header = (f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'NMAE':>10} "
              f"{'NRMSE':>10} {'MaxErr':>10} {'P95Err':>10} {'Train(s)':>10}")
    p(header)
    p("-" * 100)

    for model_name in ['gp', 'e1', 'rde_l', 'rde_t', 'rde_m', 'sindy']:
        r = all_results[model_name]
        m = r['metrics']['overall']
        line = (f"{model_name:<12} "
                f"{m['mae']:>10.6f} "
                f"{m['rmse']:>10.6f} "
                f"{m['nmae']:>10.6f} "
                f"{m['nrmse']:>10.6f} "
                f"{m['max_error']:>10.6f} "
                f"{m['p95_error']:>10.6f} "
                f"{r['training_time_s']:>10.1f}")
        p(line)

    # --- Per-state NMAE table ---
    p("\n" + "=" * 120)
    p("PER-STATE NMAE TABLE")
    p("=" * 120)
    header = f"{'Model':<12}" + "".join(f"  {name:>12}" for name in STATE_NAMES_7D) + f"  {'Overall':>12}"
    p(header)
    p("-" * 120)

    for model_name in ['gp', 'e1', 'rde_l', 'rde_t', 'rde_m', 'sindy']:
        r = all_results[model_name]
        line = f"{model_name:<12}"
        for name in STATE_NAMES_7D:
            line += f"  {r['metrics'][name]['nmae']:>12.6f}"
        line += f"  {r['metrics']['overall']['nmae']:>12.6f}"
        p(line)

    # --- Save results ---
    output_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_C_TRAINING.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    p(f"\nResults saved to: {output_path}")
    p(f"Total wall time: {time.time() - t_global:.1f}s")
    p("Done!")

    return all_results


if __name__ == '__main__':
    train_and_evaluate()

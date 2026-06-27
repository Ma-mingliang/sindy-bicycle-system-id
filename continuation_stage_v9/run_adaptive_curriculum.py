"""V10 Experiment: Adaptive & Frequency-Domain Curriculum Learning.

Tests two fundamentally different curriculum approaches against baseline:

Approach A: Error-Adaptive Curriculum
  - Monitors per-horizon validation NMAE during training
  - Advances rollout length only when model demonstrates competence
  - Uses EMA smoothing to prevent oscillation

Approach B: Frequency-Domain Curriculum
  - Trains on slow states (v, theta) first, then medium (e_y, e_psi, delta),
    then fast (theta_dot, delta_dot)
  - Orthogonal to rollout-length curriculum
  - Can be combined with mild rollout schedule

Both approaches address the root cause of why fixed rollout curriculum fails:
- Fixed curriculum forces the model to handle long rollouts before it's ready
- This produces destabilizing gradients that harm short-horizon accuracy
- Adaptive approaches let the model's actual capability guide training

Expected improvements over baseline (1,5,10,20):
- Better short-horizon NMAE (h=1,5) because training isn't destabilized
- Better long-horizon NMAE (h=50,100) because model learns stable representations
- Better survival rate at long horizons
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from canonical_node.config_v9 import *
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_v9 import NeuralODEV9, NeuralODEConfig
from canonical_node.neural_ode_adaptive_curriculum import (
    AdaptiveCurriculumNeuralODE, AdaptiveCurriculumConfig
)
from canonical_node.neural_ode_freq_curriculum import (
    FreqCurriculumNeuralODE, FreqCurriculumConfig
)
from canonical_node.evaluation_v9 import multi_step_evaluate

V10_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V10_PATH, 'raw_results')

HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500]
N_SEGMENTS = 5
SEG_LEN = 1100


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {path}")


def run_experiment(data, segments, state_std, name, model, config_dict):
    """Train and evaluate any Neural ODE variant."""
    print(f"\n  Training {name}...", flush=True)
    t0 = time.time()

    model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_path = os.path.join(V10_PATH, 'checkpoints', f'CURRICULUM_{name}.pt')
    model.save(ckpt_path)

    result = {
        'config': config_dict,
        'training_time': round(elapsed, 1),
        'evaluation': {
            str(h): {
                'nmae_mean': eval_result[h]['nmae_mean'],
                'nmae_std': eval_result[h]['nmae_std'],
                'survival_rate': eval_result[h]['survival_rate'],
                'per_state_nmae': eval_result[h]['per_state_nmae'],
            }
            for h in HORIZONS
        },
        'training_log': model._training_log[-5:],
    }

    summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 200, 500]]
    print(f"  {name}: {elapsed:.0f}s [{', '.join(summary)}]")

    return result


def main():
    print("=" * 70)
    print("V10 Experiment: Adaptive & Frequency-Domain Curriculum Learning")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}")

    # ============================================================
    # Define all experiment configurations
    # ============================================================
    print("\n[1] Running curriculum experiments...")

    # --- BASELINE: Fixed curriculum 1,5,10,20 (the current approach) ---
    baseline_config = NeuralODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20',
        n_epochs=200, seed=43,
    )

    # --- APPROACH A: Error-Adaptive Curriculum ---
    # A1: Conservative thresholds (advance only when very confident)
    adaptive_conservative = AdaptiveCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20',  # Fallback if adaptive fails
        n_epochs=200, seed=43,
        max_rollout=50,
        error_thresholds='5:0.02,10:0.04,20:0.08,50:0.15',
        ema_decay=0.9,
        eval_interval=5,
        min_epochs_per_level=15,
        max_epochs_per_level=60,
        warmup_epochs=5,
        lambda_multi_start=0.1,
        lambda_multi_end=0.4,
    )

    # A2: Moderate thresholds (balanced advance speed)
    adaptive_moderate = AdaptiveCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20',
        n_epochs=200, seed=43,
        max_rollout=50,
        error_thresholds='5:0.03,10:0.06,20:0.12,50:0.25',
        ema_decay=0.95,
        eval_interval=5,
        min_epochs_per_level=10,
        max_epochs_per_level=50,
        warmup_epochs=3,
        lambda_multi_start=0.15,
        lambda_multi_end=0.5,
    )

    # A3: Aggressive thresholds (advance quickly)
    adaptive_aggressive = AdaptiveCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20',
        n_epochs=200, seed=43,
        max_rollout=50,
        error_thresholds='5:0.05,10:0.10,20:0.20,50:0.40',
        ema_decay=0.85,
        eval_interval=5,
        min_epochs_per_level=8,
        max_epochs_per_level=40,
        warmup_epochs=2,
        lambda_multi_start=0.2,
        lambda_multi_end=0.5,
    )

    # --- APPROACH B: Frequency-Domain Curriculum ---
    # B1: Strong frequency weighting (large contrast between slow/fast)
    freq_strong = FreqCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, seed=43,
        freq_schedule='slow:3.0,med:1.0,fast:0.2->slow:0.5,med:1.0,fast:2.0',
        min_state_weight=0.1,
        freq_transition_epochs=150,
        use_mild_rollout=True,
        mild_rollout_curriculum='1,3,5,10',
        lambda_multi_freq=0.2,
        lambda_consistency=0.1,
    )

    # B2: Moderate frequency weighting
    freq_moderate = FreqCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, seed=43,
        freq_schedule='slow:2.0,med:1.0,fast:0.5->slow:0.8,med:1.0,fast:1.5',
        min_state_weight=0.2,
        freq_transition_epochs=120,
        use_mild_rollout=True,
        mild_rollout_curriculum='1,3,5,10',
        lambda_multi_freq=0.2,
        lambda_consistency=0.1,
    )

    # B3: No mild rollout (pure frequency curriculum, no rollout extension)
    freq_pure = FreqCurriculumConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, seed=43,
        freq_schedule='slow:2.5,med:1.0,fast:0.3->slow:0.5,med:1.0,fast:2.0',
        min_state_weight=0.15,
        freq_transition_epochs=150,
        use_mild_rollout=False,  # Pure frequency, no rollout
        lambda_multi_freq=0.0,  # No rollout loss at all
        lambda_consistency=0.1,
    )

    # --- BASELINE VARIANTS for comparison ---
    # Longer training with fixed curriculum
    baseline_long = NeuralODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20',
        n_epochs=400, seed=43,
    )

    # Extended fixed curriculum (that we know performs worse)
    baseline_extended = NeuralODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        rollout_curriculum='1,5,10,20,50',
        n_epochs=250, seed=43,
    )

    # --- Experiment definitions ---
    experiments = [
        # Baselines
        ('baseline_1_5_10_20', NeuralODEV9(baseline_config),
         {'hidden': 64, 'depth': 3, 'rollout': '1,5,10,20', 'epochs': 200}),
        ('baseline_long_400ep', NeuralODEV9(baseline_long),
         {'hidden': 64, 'depth': 3, 'rollout': '1,5,10,20', 'epochs': 400}),
        ('baseline_extended_1_5_10_20_50', NeuralODEV9(baseline_extended),
         {'hidden': 64, 'depth': 3, 'rollout': '1,5,10,20,50', 'epochs': 250}),

        # Approach A: Adaptive Curriculum
        ('adaptive_conservative', AdaptiveCurriculumNeuralODE(adaptive_conservative),
         {'hidden': 64, 'depth': 3, 'approach': 'adaptive', 'thresholds': 'conservative'}),
        ('adaptive_moderate', AdaptiveCurriculumNeuralODE(adaptive_moderate),
         {'hidden': 64, 'depth': 3, 'approach': 'adaptive', 'thresholds': 'moderate'}),
        ('adaptive_aggressive', AdaptiveCurriculumNeuralODE(adaptive_aggressive),
         {'hidden': 64, 'depth': 3, 'approach': 'adaptive', 'thresholds': 'aggressive'}),

        # Approach B: Frequency-Domain Curriculum
        ('freq_strong', FreqCurriculumNeuralODE(freq_strong),
         {'hidden': 64, 'depth': 3, 'approach': 'frequency', 'weighting': 'strong'}),
        ('freq_moderate', FreqCurriculumNeuralODE(freq_moderate),
         {'hidden': 64, 'depth': 3, 'approach': 'frequency', 'weighting': 'moderate'}),
        ('freq_pure', FreqCurriculumNeuralODE(freq_pure),
         {'hidden': 64, 'depth': 3, 'approach': 'frequency', 'weighting': 'pure'}),
    ]

    # Run all experiments
    results = {}
    for name, model, config_dict in experiments:
        try:
            results[name] = run_experiment(
                data, segments, state_std, name, model, config_dict
            )
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

        # Save intermediate results
        save_json(os.path.join(RESULTS_PATH, 'V10_ADAPTIVE_FREQ.json'), results)

    # ============================================================
    # Summary comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'H=1':>8} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 85)

    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<35} ERROR: {res['error'][:40]}")
            continue
        eval_data = res.get('evaluation', {})
        vals = []
        for h in [1, 10, 50, 100, 200, 500]:
            h_str = str(h)
            if h_str in eval_data:
                vals.append(f"{eval_data[h_str]['nmae_mean']:.4f}")
            else:
                vals.append("N/A")
        time_str = f"{res.get('training_time', 0):.0f}s"
        print(f"{name:<35} {'  '.join(vals)}  [{time_str}]")

    # Survival rate comparison
    print(f"\n{'Survival Rates':}")
    print(f"{'Config':<35} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 70)
    for name, res in results.items():
        if 'error' in res:
            continue
        eval_data = res.get('evaluation', {})
        vals = []
        for h in [50, 100, 200, 500]:
            h_str = str(h)
            if h_str in eval_data:
                vals.append(f"{eval_data[h_str]['survival_rate']:.0%}")
            else:
                vals.append("N/A")
        print(f"{name:<35} {'  '.join(vals)}")

    # Per-state NMAE breakdown for best performers
    print(f"\n{'Per-State NMAE at H=50 (lower is better)':}")
    print(f"{'Config':<30} {'e_y':>8} {'e_psi':>8} {'v':>8} {'theta':>8} {'theta_d':>8} {'delta':>8} {'del_d':>8}")
    print("-" * 95)
    for name, res in results.items():
        if 'error' in res:
            continue
        eval_data = res.get('evaluation', {})
        if '50' not in eval_data:
            continue
        psn = eval_data['50'].get('per_state_nmae', {})
        vals = []
        for state_name in ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']:
            if state_name in psn:
                vals.append(f"{psn[state_name]['mean']:.4f}")
            else:
                vals.append("N/A")
        print(f"{name:<30} {'  '.join(vals)}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min")
    print(f"Results saved to: {os.path.join(RESULTS_PATH, 'V10_ADAPTIVE_FREQ.json')}")


if __name__ == '__main__':
    main()

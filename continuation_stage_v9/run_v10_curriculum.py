"""V10 Experiment 2: Extended Curriculum Learning.

Tests longer rollout curriculum to improve long-horizon prediction.
Baseline: 1,5,10,20 → New: 1,5,10,20,50 and 1,5,10,20,50,100"""
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


def run_experiment(data, segments, state_std, name, config):
    """Train and evaluate a Neural ODE with given config."""
    print(f"\n  Training {name}...", flush=True)
    t0 = time.time()

    model = NeuralODEV9(config)
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
        'config': {
            'hidden': config.hidden,
            'depth': config.depth,
            'activation': config.activation,
            'lr': config.lr,
            'rollout_curriculum': config.rollout_curriculum,
            'n_epochs': config.n_epochs,
            'seed': config.seed,
        },
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
    print("V10 Experiment 2: Extended Curriculum Learning")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}")

    # Configurations to test
    configs = {
        # Baseline: original V9 curriculum
        'baseline_1_5_10_20': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,5,10,20',
            n_epochs=200, seed=43,
        ),
        # Extended to 50
        'curriculum_1_5_10_20_50': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,5,10,20,50',
            n_epochs=250, seed=43,
        ),
        # Extended to 100
        'curriculum_1_5_10_20_50_100': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,5,10,20,50,100',
            n_epochs=300, seed=43,
        ),
        # More steps early: 1,3,5,10,20,50
        'curriculum_1_3_5_10_20_50': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,3,5,10,20,50',
            n_epochs=250, seed=43,
        ),
        # Aggressive: jump to 100 early
        'curriculum_1_10_50_100': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,10,50,100',
            n_epochs=300, seed=43,
        ),
        # Longer training with original curriculum
        'baseline_long_400ep': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            rollout_curriculum='1,5,10,20',
            n_epochs=400, seed=43,
        ),
    }

    # Run experiments
    print("\n[1] Running curriculum experiments...")
    results = {}

    for name, config in configs.items():
        try:
            results[name] = run_experiment(data, segments, state_std, name, config)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

        save_json(os.path.join(RESULTS_PATH, 'V10_CURRICULUM.json'), results)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 70)

    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<35} ERROR: {res['error'][:40]}")
            continue
        eval_data = res.get('evaluation', {})
        vals = []
        for h in [10, 50, 100, 200, 500]:
            h_str = str(h)
            if h_str in eval_data:
                vals.append(f"{eval_data[h_str]['nmae_mean']:.4f}")
            else:
                vals.append("N/A")
        print(f"{name:<35} {'  '.join(vals)}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min")


if __name__ == '__main__':
    main()

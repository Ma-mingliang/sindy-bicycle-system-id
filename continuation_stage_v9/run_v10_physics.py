"""V10 Experiment 1: Physics-Informed Loss Enhancement.

Compares physics-constrained Neural ODE against V9 baseline.
Tests multiple physics loss weights to find optimal configuration."""
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
from canonical_node.neural_ode_physics import PhysicsNeuralODE, PhysicsODEConfig
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


def load_baseline():
    """Load V9 baseline results."""
    path = os.path.join(RESULTS_PATH, 'PHASE_1B_NEURAL_ODE_SEARCH.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        # Get best config results
        best_key = data.get('best', {}).get('key', 'config3_seed43')
        if best_key in data:
            return data[best_key]
    return None


def run_physics_experiment(data, segments, state_std, config_name, physics_config):
    """Train and evaluate a physics-informed Neural ODE."""
    print(f"\n  Training {config_name}...", flush=True)
    t0 = time.time()

    model = PhysicsNeuralODE(physics_config)
    model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_path = os.path.join(V10_PATH, 'checkpoints', f'PHYSICS_{config_name}.pt')
    model.save(ckpt_path)

    result = {
        'config': {
            'hidden': physics_config.hidden,
            'depth': physics_config.depth,
            'activation': physics_config.activation,
            'lr': physics_config.lr,
            'lambda_physics_ey': physics_config.lambda_physics_ey,
            'lambda_physics_theta': physics_config.lambda_physics_theta,
            'lambda_physics_delta': physics_config.lambda_physics_delta,
            'seed': physics_config.seed,
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
        'training_log': model._training_log[-5:],  # Last 5 epochs
    }

    summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 200, 500]]
    print(f"  {config_name}: {elapsed:.0f}s [{', '.join(summary)}]")

    return result


def main():
    print("=" * 70)
    print("V10 Experiment 1: Physics-Informed Loss Enhancement")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}")

    # Load baseline
    print("\n[1] Loading V9 baseline...")
    baseline = load_baseline()
    if baseline:
        print(f"  Baseline (config3_seed43):")
        for h in [1, 10, 50, 100, 200, 500]:
            h_str = str(h)
            if h_str in baseline.get('evaluation', {}):
                nmae = baseline['evaluation'][h_str]['nmae_mean']
                print(f"    H={h}: NMAE={nmae:.4f}")
    else:
        print("  WARNING: Baseline not found, will train fresh baseline")

    # Physics configurations to test
    configs = {
        # Moderate physics constraints
        'physics_moderate': PhysicsODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_physics_ey=0.1, lambda_physics_theta=0.05, lambda_physics_delta=0.05,
            lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
            seed=43, n_epochs=200,
        ),
        # Strong physics constraints
        'physics_strong': PhysicsODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_physics_ey=0.3, lambda_physics_theta=0.15, lambda_physics_delta=0.15,
            lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
            seed=43, n_epochs=200,
        ),
        # Weak physics constraints
        'physics_weak': PhysicsODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_physics_ey=0.03, lambda_physics_theta=0.01, lambda_physics_delta=0.01,
            lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
            seed=43, n_epochs=200,
        ),
        # Only e_y constraint (most relevant)
        'physics_ey_only': PhysicsODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_physics_ey=0.2, lambda_physics_theta=0.0, lambda_physics_delta=0.0,
            lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
            seed=43, n_epochs=200,
        ),
        # Physics + longer curriculum
        'physics_curriculum': PhysicsODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_physics_ey=0.1, lambda_physics_theta=0.05, lambda_physics_delta=0.05,
            lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),
    }

    # Run experiments
    print("\n[2] Running physics-informed experiments...")
    results = {}

    # Add baseline reference
    if baseline:
        results['baseline_v9'] = {
            'config': baseline.get('config', {}),
            'evaluation': baseline.get('evaluation', {}),
        }

    for name, config in configs.items():
        try:
            results[name] = run_physics_experiment(data, segments, state_std, name, config)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

        # Incremental save
        save_json(os.path.join(RESULTS_PATH, 'V10_PHYSICS.json'), results)

    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 70)

    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<25} ERROR: {res['error'][:40]}")
            continue
        eval_data = res.get('evaluation', {})
        vals = []
        for h in [10, 50, 100, 200, 500]:
            h_str = str(h)
            if h_str in eval_data:
                vals.append(f"{eval_data[h_str]['nmae_mean']:.4f}")
            else:
                vals.append("N/A")
        print(f"{name:<25} {'  '.join(vals)}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min")
    print(f"Results saved to: {os.path.join(RESULTS_PATH, 'V10_PHYSICS.json')}")


if __name__ == '__main__':
    main()

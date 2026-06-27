"""V10 Experiment 3: Feedback Correction Neural ODE.

Tests learned feedback correction to stabilize long-horizon prediction.
Variants: linear, mlp, state_dependent, low_rank."""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from canonical_node.config_v9 import *
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_feedback import FeedbackNeuralODE, FeedbackODEConfig
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
    print(f"\n  Training {name}...", flush=True)
    t0 = time.time()
    model = FeedbackNeuralODE(config)
    model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )
    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0

    ckpt_path = os.path.join(V10_PATH, 'checkpoints', f'FEEDBACK_{name}.pt')
    model.save(ckpt_path)

    result = {
        'config': {
            'hidden': config.hidden, 'depth': config.depth,
            'activation': config.activation, 'lr': config.lr,
            'feedback_type': config.feedback_type,
            'feedback_gain_init': config.feedback_gain_init,
            'feedback_rank': config.feedback_rank,
            'lambda_feedback': config.lambda_feedback,
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
    print("V10 Experiment 3: Feedback Correction Neural ODE")
    print("=" * 70)
    t_start = time.time()

    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}")

    # V9 baseline for comparison
    baseline_nmae = {'10': 0.0628, '50': 0.4557, '100': 0.5064, '200': 0.4737, '500': 0.5529}

    configs = {
        # 1. Linear feedback (simplest)
        'feedback_linear_m0.1': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.1,
            feedback_rank=0, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 2. Stronger linear damping
        'feedback_linear_m0.3': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.3,
            feedback_rank=0, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 3. Weak linear damping
        'feedback_linear_m0.03': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.03,
            feedback_rank=0, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 4. Low-rank feedback (fewer parameters)
        'feedback_rank3': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.1,
            feedback_rank=3, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 5. MLP feedback
        'feedback_mlp': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='mlp', feedback_gain_init=-0.1,
            feedback_rank=0, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 6. State-dependent feedback
        'feedback_state_dep': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='state_dependent', feedback_gain_init=-0.1,
            feedback_rank=0, lambda_feedback=0.01,
            seed=43, n_epochs=200,
        ),
        # 7. Linear + longer curriculum
        'feedback_linear_curriculum': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.1,
            feedback_rank=0, lambda_feedback=0.01,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),
        # 8. No feedback regularization (let feedback learn freely)
        'feedback_linear_noreg': FeedbackODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            feedback_type='linear', feedback_gain_init=-0.1,
            feedback_rank=0, lambda_feedback=0.0,
            seed=43, n_epochs=200,
        ),
    }

    print(f"\n[1] Running {len(configs)} feedback experiments...")
    results = {'baseline_v9': baseline_nmae}

    for name, config in configs.items():
        try:
            results[name] = run_experiment(data, segments, state_std, name, config)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}
        save_json(os.path.join(RESULTS_PATH, 'V10_FEEDBACK.json'), results)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 70)
    for name, res in results.items():
        if isinstance(res, dict) and '10' in res:
            # Baseline format
            vals = [f"{res.get(str(h), 'N/A')}" for h in [10, 50, 100, 200, 500]]
            print(f"{name:<35} {'  '.join(vals)}")
        elif 'error' in res:
            print(f"{name:<35} ERROR: {res['error'][:40]}")
        else:
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

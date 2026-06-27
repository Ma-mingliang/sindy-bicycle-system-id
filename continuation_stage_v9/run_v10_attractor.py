"""V10 Experiment 4: Attractor-Based Stabilization Neural ODE.

Tests learned attractor dynamics to stabilize long-horizon prediction.
Unlike linear feedback (which fights base dynamics), the attractor:
1. Learns the stable manifold from data
2. Only activates when states drift out-of-distribution
3. Uses per-state correction weights (focus on e_y, e_psi)
4. Has a consistency loss to prevent changing behavior on training data

V10 Feedback results (for reference):
    baseline:         H=10: 0.0628  H=100: 0.5064  H=500: 0.5529
    linear_m0.1:      H=10: 0.0597  H=100: 0.6578  H=500: 0.6973 (WORSE)
    mlp:              H=10: 0.0865  H=100: 0.7499  H=500: 0.9070 (WORSE)
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

from canonical_node.config_v9 import *
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_attractor import AttractorNeuralODE, AttractorODEConfig
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
    model = AttractorNeuralODE(config)
    model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )
    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0

    ckpt_path = os.path.join(V10_PATH, 'checkpoints', f'ATTRACTOR_{name}.pt')
    model.save(ckpt_path)

    result = {
        'config': {
            'hidden': config.hidden, 'depth': config.depth,
            'activation': config.activation, 'lr': config.lr,
            'attractor_type': config.attractor_type,
            'attractor_hidden': config.attractor_hidden,
            'correct_mask': config.correct_mask,
            'gate_type': config.gate_type,
            'lambda_consistency_attractor': config.lambda_consistency_attractor,
            'lambda_attractor_reg': config.lambda_attractor_reg,
            'lambda_manifold_id': config.lambda_manifold_id,
            'rollout_curriculum': config.rollout_curriculum,
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
    print("V10 Experiment 4: Attractor-Based Stabilization Neural ODE")
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
        # ================================================================
        # Group 1: Basic attractor variants (correcting only e_y, e_psi)
        # ================================================================

        # 1. Residual attractor: correct only e_y and e_psi
        'residual_ey_epsi': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 2. Full attractor: correct all states
        'full_all_states': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,1,1,1,1,1',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 3. Residual with e_y, e_psi, theta_dot (the three most error-prone)
        'residual_ey_epsi_thetadot': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,1,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # ================================================================
        # Group 2: Gate type ablation
        # ================================================================

        # 4. Learned gate (MLP decides when to activate)
        'residual_learned_gate': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='learned',
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 5. Sigmoid gate (simple magnitude-based)
        'residual_sigmoid_gate': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='sigmoid',
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # ================================================================
        # Group 3: Hyperparameter sweep
        # ================================================================

        # 6. Stronger consistency (force attractor to be quiet on training data)
        'residual_strong_consistency': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=2.0,  # 4x stronger
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 7. Weaker consistency (let attractor learn more freely)
        'residual_weak_consistency': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.1,  # 5x weaker
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 8. Larger attractor network
        'residual_large_attractor': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=64, attractor_depth=3,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # ================================================================
        # Group 4: Ensemble attractor
        # ================================================================

        # 9. Ensemble of 3 attractors
        'ensemble_3_ey_epsi': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='ensemble', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # ================================================================
        # Group 5: Extended curriculum
        # ================================================================

        # 10. Residual with longer curriculum
        'residual_long_curriculum': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50,100',
            seed=43, n_epochs=300,
        ),

        # 11. Lower gate threshold (activate attractor more aggressively)
        'residual_low_threshold': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.2,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.1,
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),

        # 12. No manifold identity reg (let manifold learn freely)
        'residual_no_manifold_reg': AttractorODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            attractor_type='residual', attractor_hidden=32, attractor_depth=2,
            correct_mask='1,1,0,0,0,0,0',
            gate_type='distance', gate_threshold=0.5,
            lambda_consistency_attractor=0.5,
            lambda_attractor_reg=0.01,
            lambda_manifold_id=0.0,  # No manifold identity constraint
            rollout_curriculum='1,5,10,20,50',
            seed=43, n_epochs=250,
        ),
    }

    print(f"\n[1] Running {len(configs)} attractor experiments...")
    results = {'baseline_v9': baseline_nmae}

    for name, config in configs.items():
        try:
            results[name] = run_experiment(data, segments, state_std, name, config)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}
        save_json(os.path.join(RESULTS_PATH, 'V10_ATTRACTOR.json'), results)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("ATTRACTOR EXPERIMENT SUMMARY")
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

    # Best performer
    print("\n" + "=" * 70)
    best_name = None
    best_score = float('inf')
    for name, res in results.items():
        if 'error' in res:
            continue
        eval_data = res.get('evaluation', {})
        if '100' in eval_data and '500' in eval_data:
            score = eval_data['100']['nmae_mean'] + eval_data['500']['nmae_mean']
            if score < best_score:
                best_score = score
                best_name = name

    if best_name:
        print(f"Best attractor: {best_name} (H=100+H=500 score: {best_score:.4f})")
        print(f"Baseline score: {baseline_nmae['100'] + baseline_nmae['500']:.4f}")
        improvement = (baseline_nmae['100'] + baseline_nmae['500']) - best_score
        print(f"Improvement: {improvement:.4f} ({improvement/(baseline_nmae['100'] + baseline_nmae['500'])*100:.1f}%)")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min")


if __name__ == '__main__':
    main()

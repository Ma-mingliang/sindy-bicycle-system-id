"""V11 Experiment: Contractivity-Promoting Regularization for Neural ODE.

Tests contractivity regularization to improve long-horizon prediction stability.
Unlike feedback correction (V10), this does NOT change the architecture -- it only
adds a loss term during training that penalizes positive eigenvalues of the
Jacobian's symmetric part.

References:
- Zakwan et al. 2025: "Robust Convolution Neural ODEs via Contractivity"
- Guglielmi et al. 2025: Contractivity analysis for Neural ODEs

Key difference from V10 feedback:
- V10 feedback: dx/dt = f(x,u) + K*x  (adds correction term, fights base dynamics)
- V11 contractive: dx/dt = f(x,u) + loss_regularization  (guides base dynamics)
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
from canonical_node.neural_ode_contractive import NeuralODEContractive, ContractiveODEConfig
from canonical_node.neural_ode_v9 import NeuralODEV9, NeuralODEConfig
from canonical_node.evaluation_v9 import multi_step_evaluate

V11_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V11_PATH, 'raw_results')
HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500]
N_SEGMENTS = 5
SEG_LEN = 1100


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {path}")


def run_experiment(data, segments, state_std, name, config):
    """Train and evaluate a contractive Neural ODE."""
    print(f"\n  Training {name}...", flush=True)
    t0 = time.time()

    model = NeuralODEContractive(config)
    model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_path = os.path.join(V11_PATH, 'checkpoints', f'CONTRACTIVE_{name}.pt')
    model.save(ckpt_path)

    # Build result dict
    result = {
        'config': {
            'hidden': config.hidden,
            'depth': config.depth,
            'activation': config.activation,
            'lr': config.lr,
            'lambda_contractive': config.lambda_contractive,
            'contractive_method': config.contractive_method,
            'contractive_warmup_epochs': config.contractive_warmup_epochs,
            'contractive_ramp_epochs': config.contractive_ramp_epochs,
            'lambda_jacobian': config.lambda_jacobian,
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
        'training_log_summary': {
            'final_loss': model._training_log[-1]['loss'] if model._training_log else None,
            'final_max_eig': model._training_log[-1]['max_eigenvalue_mean'] if model._training_log else None,
            'final_frac_positive': model._training_log[-1]['frac_positive_eigenvalues'] if model._training_log else None,
            'final_eff_lambda': model._training_log[-1]['eff_lambda'] if model._training_log else None,
        },
    }

    # Print summary
    summary = [f"h={h}:{eval_result[h]['nmae_mean']:.4f}" for h in [1, 10, 50, 100, 200, 500]]
    print(f"  {name}: {elapsed:.0f}s [{', '.join(summary)}]")
    return result


def main():
    print("=" * 70)
    print("V11: Contractivity-Promoting Regularization for Neural ODE")
    print("=" * 70)
    t_start = time.time()

    # ================================================================
    # [0] Load data
    # ================================================================
    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']
    print(f"  Train: {len(data['train_states'])}, Segments: {len(segments)}")

    # V9 baseline for comparison
    baseline_nmae = {'10': 0.0628, '50': 0.4557, '100': 0.5064, '200': 0.4737, '500': 0.5529}

    # ================================================================
    # [1] Baseline: retrain V9 to confirm reproducibility
    # ================================================================
    print("\n[1] V9 Baseline retrain...")
    baseline_config = NeuralODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, seed=42,
    )
    print("  Training V9 baseline...", flush=True)
    t0 = time.time()
    baseline_model = NeuralODEV9(baseline_config)
    baseline_model.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )
    baseline_eval = multi_step_evaluate(baseline_model, segments, state_std, HORIZONS)
    elapsed = time.time() - t0
    baseline_model.save(os.path.join(V11_PATH, 'checkpoints', 'V9_BASELINE.pt'))
    print(f"  V9 baseline: {elapsed:.0f}s")
    for h in [1, 10, 50, 100, 200, 500]:
        print(f"    H={h}: NMAE={baseline_eval[h]['nmae_mean']:.4f}")

    results = {
        'baseline_v9_retrained': {
            'evaluation': {
                str(h): {
                    'nmae_mean': baseline_eval[h]['nmae_mean'],
                    'survival_rate': baseline_eval[h]['survival_rate'],
                }
                for h in HORIZONS
            }
        },
        'baseline_v9_published': baseline_nmae,
    }

    # ================================================================
    # [2] Contractivity grid search
    # ================================================================
    print("\n[2] Contractivity regularization grid search...")

    # Grid: sweep lambda, warmup, method
    configs = {}

    # --- Group A: Lambda sweep (fixed warmup=20, ramp=50) ---
    for lam in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]:
        configs[f'contractive_lam{lam}'] = ContractiveODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_contractive=lam,
            contractive_method='symmetric_eig',
            contractive_warmup_epochs=20,
            contractive_ramp_epochs=50,
            lambda_jacobian=0.01,
            n_epochs=200, seed=42,
        )

    # --- Group B: Warmup schedule sweep (fixed lambda=0.05) ---
    for warmup in [0, 10, 20, 40]:
        configs[f'contractive_warm{warmup}'] = ContractiveODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_contractive=0.05,
            contractive_method='symmetric_eig',
            contractive_warmup_epochs=warmup,
            contractive_ramp_epochs=50,
            lambda_jacobian=0.01,
            n_epochs=200, seed=42,
        )

    # --- Group C: With and without existing Jacobian loss ---
    configs['contractive_no_jac_norm'] = ContractiveODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        lambda_contractive=0.05,
        contractive_method='symmetric_eig',
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=50,
        lambda_jacobian=0.0,  # Disable Frobenius norm, rely only on eigenvalue
        n_epochs=200, seed=42,
    )

    configs['contractive_no_jac_norm_lam0.1'] = ContractiveODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        lambda_contractive=0.1,
        contractive_method='symmetric_eig',
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=50,
        lambda_jacobian=0.0,
        n_epochs=200, seed=42,
    )

    # --- Group D: Power iteration method (for comparison / validation) ---
    configs['contractive_power_iter'] = ContractiveODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        lambda_contractive=0.05,
        contractive_method='power_iter',
        contractive_power_steps=5,
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=50,
        lambda_jacobian=0.01,
        n_epochs=200, seed=42,
    )

    # --- Group E: Best lambda + longer training ---
    configs['contractive_best_long'] = ContractiveODEConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        lambda_contractive=0.05,
        contractive_method='symmetric_eig',
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=50,
        lambda_jacobian=0.01,
        rollout_curriculum='1,5,10,20,50',
        n_epochs=300, seed=42,
    )

    # --- Group F: Multi-seed for best config (determine variance) ---
    for seed in [42, 43, 44]:
        configs[f'contractive_best_seed{seed}'] = ContractiveODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            lambda_contractive=0.05,
            contractive_method='symmetric_eig',
            contractive_warmup_epochs=20,
            contractive_ramp_epochs=50,
            lambda_jacobian=0.01,
            n_epochs=200, seed=seed,
        )

    print(f"  Running {len(configs)} experiments...")

    for name, config in configs.items():
        try:
            results[name] = run_experiment(data, segments, state_std, name, config)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

        # Save after each experiment (crash recovery)
        save_json(os.path.join(RESULTS_PATH, 'V11_CONTRACTIVE.json'), results)

    # ================================================================
    # [3] Summary table
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Config':<40} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8}")
    print("-" * 80)

    # Baseline
    print(f"{'baseline_v9_published':<40} "
          f"{'0.0628':>8} {'0.4557':>8} {'0.5064':>8} {'0.4737':>8} {'0.5529':>8}")

    # Retrained baseline
    b = results.get('baseline_v9_retrained', {}).get('evaluation', {})
    if b:
        vals = [f"{b.get(str(h), {}).get('nmae_mean', 0):.4f}" for h in [10, 50, 100, 200, 500]]
        print(f"{'baseline_v9_retrained':<40} {'  '.join(vals)}")

    # Contractive variants
    for name, res in results.items():
        if name.startswith('baseline_') or 'error' in res:
            continue
        eval_data = res.get('evaluation', {})
        if eval_data:
            vals = []
            for h in [10, 50, 100, 200, 500]:
                h_str = str(h)
                if h_str in eval_data:
                    vals.append(f"{eval_data[h_str]['nmae_mean']:.4f}")
                else:
                    vals.append("N/A")
            print(f"{name:<40} {'  '.join(vals)}")

    # ================================================================
    # [4] Key analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("KEY ANALYSIS")
    print("=" * 70)

    # Find best at each horizon
    for h in [10, 50, 100, 200, 500]:
        best_name = None
        best_val = float('inf')
        for name, res in results.items():
            if name.startswith('baseline_') or 'error' in res:
                continue
            eval_data = res.get('evaluation', {})
            val = eval_data.get(str(h), {}).get('nmae_mean', float('inf'))
            if val < best_val:
                best_val = val
                best_name = name
        if best_name:
            bl = baseline_nmae[str(h)]
            improvement = (bl - best_val) / bl * 100
            print(f"  Best H={h}: {best_name} (NMAE={best_val:.4f}, "
                  f"{'+' if improvement > 0 else ''}{improvement:.1f}% vs baseline)")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time / 60:.1f} min")
    print("Done.")


if __name__ == '__main__':
    main()

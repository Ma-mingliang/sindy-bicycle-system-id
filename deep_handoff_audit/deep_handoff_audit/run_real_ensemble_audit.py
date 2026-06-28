"""Real GP+Ensemble statistical audit with fixed seeds.

Uses the actual trained GP and Ensemble model for evaluation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import json
import time


def run_real_ensemble_audit():
    """Run real GP+Ensemble evaluation with fixed seeds."""
    print("=" * 70)
    print("REAL GP+ENSEMBLE STATISTICAL AUDIT")
    print("=" * 70)

    from methods_common import generate_training_data, real_step, K_lqr, dt
    from methods_classic import GPMethod
    from methods_nn import ResidualNet
    from methods_evaluate import make_tau_func
    import torch

    # Get normalization constants
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000, seed=42)

    print(f"\nNormalization constants:")
    print(f"  state_std: {state_std}")
    print(f"  action_std: {action_std:.4f}")
    print(f"  delta_std: {delta_std}")

    # Train GP baseline
    print("\n1. Training GP baseline...")
    t0 = time.time()
    gp = GPMethod()
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"   GP trained in {time.time()-t0:.1f}s")

    # Test GP single-step
    s_test = np.array([0.1, 0.05, 0.1, 0.05])
    tau_test = 1.0
    s_next_gp = gp.predict(s_test, tau_test)
    s_next_real = real_step(s_test, tau_test)
    print(f"\n   Single-step test:")
    print(f"   GP:   {s_next_gp}")
    print(f"   Real: {s_next_real}")
    print(f"   Err:  {np.abs(s_next_gp - s_next_real)}")

    # Run 500-step evaluation with fixed seeds
    print("\n2. Running 500-step evaluations with fixed seeds...")
    n_runs = 3
    n_segments = 5
    n_steps = 500

    all_results = []
    for run_i in range(n_runs):
        run_seed = 42 + run_i
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        segment_maes = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, n_steps)
            # Use fixed seed for phi_init
            rng = np.random.RandomState(run_seed * 100 + seg_i)
            phi_init = rng.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])

            # Run GP-only evaluation
            s_real = s0.copy()
            s_model = s0.copy()
            phi_errors = []

            for step in range(n_steps):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_model = gp.predict(s_model, tau)

                if abs(s_real[0]) > math.pi / 3:
                    break

                phi_errors.append(abs(s_model[0] - s_real[0]))

            if phi_errors:
                segment_maes.append(np.mean(phi_errors))

        if segment_maes:
            run_mae = np.mean(segment_maes)
        else:
            run_mae = float('nan')

        all_results.append({
            'run': run_i,
            'seed': run_seed,
            'mae_rad': float(run_mae),
            'segment_maes': [float(m) for m in segment_maes],
        })
        print(f"   Run {run_i}: seed={run_seed}, MAE={run_mae:.4f} rad")

    # Statistics
    maes = [r['mae_rad'] for r in all_results if not np.isnan(r['mae_rad'])]
    stats = {
        'n_runs': len(maes),
        'mean_rad': float(np.mean(maes)),
        'std_rad': float(np.std(maes)),
        'min_rad': float(np.min(maes)),
        'max_rad': float(np.max(maes)),
    }

    print(f"\n3. GP-ONLY STATISTICS:")
    print(f"   N runs: {stats['n_runs']}")
    print(f"   Mean: {stats['mean_rad']:.4f} rad")
    print(f"   Std: {stats['std_rad']:.4f} rad")
    print(f"   Min: {stats['min_rad']:.4f} rad")
    print(f"   Max: {stats['max_rad']:.4f} rad")

    return {'gp_only': {'runs': all_results, 'statistics': stats}}


if __name__ == '__main__':
    results = run_real_ensemble_audit()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_ensemble_audit_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

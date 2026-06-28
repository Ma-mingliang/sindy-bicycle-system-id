"""
Fair comparison between Version L (Local Dynamics Residual) and
Version T (Trajectory Synchronization Residual) DAgger implementations.

Same data, same NN structure, same seeds, same training rounds.
Evaluates on Mode A/B/C/D, per-state errors, long-term stability,
different initial states, different action signals, OOD states.
"""

import sys
import os
import json
import numpy as np
import math

# Add project root and current directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

from methods_common import real_step, generate_training_data, dt
from methods_evaluate import make_tau_func


def run_trajectory(model, s0, n_steps, tau_func, real_step_fn):
    """Run a trajectory and return model states and real states."""
    s_model = s0.copy()
    s_real = s0.copy()
    model_traj = [s0.copy()]
    real_traj = [s0.copy()]

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real = real_step_fn(s_real, tau)
        s_model = model.predict(s_model, tau)

        if abs(s_real[0]) > math.pi / 3:
            model_traj.append(np.full(4, np.nan))
            real_traj.append(np.full(4, np.nan))
            break

        model_traj.append(s_model.copy())
        real_traj.append(s_real.copy())

    return np.array(model_traj), np.array(real_traj)


def compute_metrics(model_traj, real_traj, state_std, max_steps):
    """Compute evaluation metrics."""
    n_valid = min(len(model_traj), len(real_traj))

    # Find valid (non-NaN) steps
    valid_mask = ~(np.any(np.isnan(model_traj[:n_valid]), axis=1) |
                   np.any(np.isnan(real_traj[:n_valid]), axis=1))
    n_valid_steps = int(np.sum(valid_mask))

    if n_valid_steps == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'per_state_mae': [float('nan')] * 4,
            'normalized_mae': float('nan'),
            'survival_steps': 0,
            'endpoint_error': float('nan'),
        }

    valid_idx = np.where(valid_mask)[0]
    errors = model_traj[valid_idx] - real_traj[valid_idx]
    abs_errors = np.abs(errors)

    state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
    per_state_mae = [float(np.mean(abs_errors[:, i])) for i in range(4)]

    return {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'per_state_mae': per_state_mae,
        'per_state_names': state_names,
        'normalized_mae': float(np.mean(abs_errors / state_std)),
        'survival_steps': n_valid_steps,
        'endpoint_error': float(np.mean(abs_errors[-1])) if n_valid_steps > 0 else float('nan'),
    }


def evaluate_model(model, state_std, action_std, delta_std, n_eval_seeds=5, max_steps=200):
    """Evaluate a model across multiple conditions."""
    results = {}

    # === Mode A: Teacher Forcing (single-step prediction accuracy) ===
    print("  Mode A (Teacher Forcing)...", flush=True)
    mode_a_errors = []
    for seg_i in range(n_eval_seeds):
        tau_func = make_tau_func(200 + seg_i, max_steps)
        s = np.array([0.1 * (seg_i - 2), 0.0, 0.0, 0.0])
        for step in range(max_steps):
            tau = tau_func(step, s)
            s_next_real = real_step(s, tau)
            s_next_model = model.predict(s, tau)
            err = np.abs(s_next_model - s_next_real)
            mode_a_errors.append(float(np.mean(err)))
            s = s_next_real
            if abs(s[0]) > math.pi / 3:
                break
    results['mode_a'] = {
        'mean_mae': float(np.mean(mode_a_errors)) if mode_a_errors else float('nan'),
        'std_mae': float(np.std(mode_a_errors)) if mode_a_errors else float('nan'),
        'n_samples': len(mode_a_errors),
    }

    # === Mode B: Fixed action open-loop ===
    print("  Mode B (Fixed action open-loop)...", flush=True)
    mode_b_results = {}
    for eval_steps in [10, 50, 100, 200]:
        step_errors = []
        for seg_i in range(n_eval_seeds):
            tau_func = make_tau_func(300 + seg_i, eval_steps)
            s0 = np.array([0.1 * (seg_i - 2), 0.0, 0.0, 0.0])
            model_traj, real_traj = run_trajectory(model, s0, eval_steps, tau_func, real_step)
            metrics = compute_metrics(model_traj, real_traj, state_std, eval_steps)
            step_errors.append(metrics['mae'])
        valid_errors = [e for e in step_errors if not math.isnan(e)]
        mode_b_results[str(eval_steps)] = float(np.mean(valid_errors)) if valid_errors else float('nan')
    results['mode_b'] = mode_b_results

    # === Mode C: Mixed (tau from real state, model self-rolls) ===
    print("  Mode C (Mixed)...", flush=True)
    mode_c_results = {}
    for eval_steps in [10, 50, 100, 200]:
        step_errors = []
        for seg_i in range(n_eval_seeds):
            tau_func = make_tau_func(400 + seg_i, eval_steps)
            s0 = np.array([0.1 * (seg_i - 2), 0.0, 0.0, 0.0])
            model_traj, real_traj = run_trajectory(model, s0, eval_steps, tau_func, real_step)
            metrics = compute_metrics(model_traj, real_traj, state_std, eval_steps)
            step_errors.append(metrics['mae'])
        valid_errors = [e for e in step_errors if not math.isnan(e)]
        mode_c_results[str(eval_steps)] = float(np.mean(valid_errors)) if valid_errors else float('nan')
    results['mode_c'] = mode_c_results

    # === Mode D: Full closed-loop (LQR on model state) ===
    print("  Mode D (Full closed-loop)...", flush=True)
    from methods_common import K_lqr
    mode_d_results = {}
    for eval_steps in [10, 50, 100, 200]:
        step_errors = []
        for seg_i in range(n_eval_seeds):
            rng = np.random.RandomState(42 + seg_i)
            disturbances = rng.uniform(-0.5, 0.5, eval_steps)
            target = np.clip(rng.normal(0, 0.15), -math.pi / 12, math.pi / 12)

            s_model = np.array([0.1 * (seg_i - 2), 0.0, 0.0, 0.0])
            s_real = s_model.copy()
            model_traj_list = [s_model.copy()]
            real_traj_list = [s_real.copy()]

            for step in range(eval_steps):
                # LQR on REAL state for reference
                x_lqr_real = np.array([s_real[0] - target, s_real[2], s_real[1], s_real[3]])
                tau_real = float(-K_lqr @ x_lqr_real) + disturbances[step]

                # LQR on MODEL state (controller sees model)
                x_lqr_model = np.array([s_model[0] - target, s_model[2], s_model[1], s_model[3]])
                tau_model = float(-K_lqr @ x_lqr_model) + disturbances[step]

                s_real = real_step(s_real, tau_real)
                s_model = model.predict(s_model, tau_model)

                if abs(s_real[0]) > math.pi / 3:
                    model_traj_list.append(np.full(4, np.nan))
                    real_traj_list.append(np.full(4, np.nan))
                    break

                model_traj_list.append(s_model.copy())
                real_traj_list.append(s_real.copy())

            model_traj = np.array(model_traj_list)
            real_traj = np.array(real_traj_list)
            metrics = compute_metrics(model_traj, real_traj, state_std, eval_steps)
            step_errors.append(metrics['mae'])
        valid_errors = [e for e in step_errors if not math.isnan(e)]
        mode_d_results[str(eval_steps)] = float(np.mean(valid_errors)) if valid_errors else float('nan')
    results['mode_d'] = mode_d_results

    # === Different initial conditions ===
    print("  Different initial states...", flush=True)
    init_conditions = {
        'nominal': np.array([0.0, 0.0, 0.0, 0.0]),
        'large_phi': np.array([0.3, 0.0, 0.0, 0.0]),
        'negative_phi': np.array([-0.3, 0.0, 0.0, 0.0]),
        'high_angular_vel': np.array([0.0, 0.0, 1.5, 0.0]),
        'high_steer_rate': np.array([0.0, 0.0, 0.0, 0.8]),
        'combined': np.array([0.2, 0.1, 1.0, 0.5]),
    }
    init_results = {}
    for name, s0 in init_conditions.items():
        tau_func = make_tau_func(500 + hash(name) % 100, 100)
        model_traj, real_traj = run_trajectory(model, s0, 100, tau_func, real_step)
        metrics = compute_metrics(model_traj, real_traj, state_std, 100)
        init_results[name] = {
            'mae': metrics['mae'],
            'survival_steps': metrics['survival_steps'],
        }
    results['initial_conditions'] = init_results

    # === Different action signals ===
    print("  Different action signals...", flush=True)
    action_signals = {
        'small_sine': lambda step, s: 5.0 * math.sin(2 * math.pi * step / 50),
        'large_sine': lambda step, s: 30.0 * math.sin(2 * math.pi * step / 30),
        'step_input': lambda step, s: 20.0 if step > 20 else 0.0,
        'random_walk': None,  # will generate
        'chirp': lambda step, s: 15.0 * math.sin(2 * math.pi * (0.01 * step ** 2) / 50),
    }
    # Pre-generate random walk
    rng_rw = np.random.RandomState(99)
    rw_values = np.cumsum(rng_rw.uniform(-2, 2, 200))
    rw_values = np.clip(rw_values, -50, 50)

    action_results = {}
    for name, func in action_signals.items():
        if func is None:
            func = lambda step, s, _rw=rw_values: float(_rw[step]) if step < len(_rw) else 0.0

        s_model = np.array([0.1, 0.0, 0.0, 0.0])
        s_real = s_model.copy()
        errors = []
        for step in range(100):
            tau = func(step, s_real)
            s_real = real_step(s_real, tau)
            s_model = model.predict(s_model, tau)
            if abs(s_real[0]) > math.pi / 3:
                break
            errors.append(float(np.mean(np.abs(s_model - s_real))))
        action_results[name] = float(np.mean(errors)) if errors else float('nan')
    results['action_signals'] = action_results

    # === Out-of-distribution states ===
    print("  Out-of-distribution states...", flush=True)
    ood_states = {
        'extreme_phi': np.array([0.5, 0.0, 0.0, 0.0]),
        'extreme_delta': np.array([0.0, 0.3, 0.0, 0.0]),
        'extreme_both': np.array([0.4, 0.25, 0.0, 0.0]),
        'zero_state': np.array([0.0, 0.0, 0.0, 0.0]),
    }
    ood_results = {}
    for name, s0 in ood_states.items():
        # Single-step prediction accuracy
        tau = 10.0
        s_next_real = real_step(s0, tau)
        s_next_model = model.predict(s0, tau)
        ood_results[name] = {
            'single_step_mae': float(np.mean(np.abs(s_next_model - s_next_real))),
        }
    results['ood_states'] = ood_results

    return results


def main():
    """Main comparison experiment."""
    print("=" * 60)
    print("Version L vs Version T DAgger Comparison")
    print("=" * 60)

    # Generate training data (same for both)
    print("\n[1/4] Generating training data...")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        n_samples=5000, seed=42
    )
    print(f"  Training data: {len(states)} samples")
    print(f"  state_std: {state_std}")

    # Import versions
    from version_l_local_residual import VersionLLocalResidual
    from version_t_trajectory_residual import VersionTTrajectoryResidual

    # Fair comparison parameters
    N_MODELS = 5
    DAGGER_ROUNDS = 3
    N_EPOCHS = 100
    RESIDUAL_SCALE = 0.3

    config_str = (f"n_models={N_MODELS}, dagger_rounds={DAGGER_ROUNDS}, "
                  f"n_epochs={N_EPOCHS}, residual_scale={RESIDUAL_SCALE}")
    print(f"\n[2/4] Training Version L (Local Dynamics Residual)...")
    print(f"  Config: {config_str}")

    # Train Version L
    model_l = VersionLLocalResidual(
        n_models=N_MODELS, dagger_rounds=DAGGER_ROUNDS,
        residual_scale=RESIDUAL_SCALE, n_epochs=N_EPOCHS
    )
    model_l.train(states, actions, deltas, state_std, action_std, delta_std)
    print("  Version L training complete.")

    print(f"\n[3/4] Training Version T (Trajectory Synchronization Residual)...")
    print(f"  Config: {config_str}")

    # Train Version T
    model_t = VersionTTrajectoryResidual(
        n_models=N_MODELS, dagger_rounds=DAGGER_ROUNDS,
        residual_scale=RESIDUAL_SCALE, n_epochs=N_EPOCHS
    )
    model_t.train(states, actions, deltas, state_std, action_std, delta_std)
    print("  Version T training complete.")

    # Evaluate both
    print(f"\n[4/4] Evaluating both models...")
    print("  Evaluating Version L:")
    results_l = evaluate_model(model_l, state_std, action_std, delta_std,
                                n_eval_seeds=5, max_steps=200)
    print("  Evaluating Version T:")
    results_t = evaluate_model(model_t, state_std, action_std, delta_std,
                                n_eval_seeds=5, max_steps=200)

    # Compile results
    all_results = {
        'experiment_config': {
            'n_models': N_MODELS,
            'dagger_rounds': DAGGER_ROUNDS,
            'n_epochs': N_EPOCHS,
            'residual_scale': RESIDUAL_SCALE,
            'training_samples': len(states),
            'seed': 42,
            'nn_structure': 'ResidualNet(4+1 -> 128 -> 128 -> 4)',
        },
        'version_l': results_l,
        'version_t': results_t,
    }

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'LOCAL_VS_TRAJECTORY_RESULTS.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode in ['mode_a', 'mode_b', 'mode_c', 'mode_d']:
        l_val = results_l.get(mode, {})
        t_val = results_t.get(mode, {})
        if isinstance(l_val, dict):
            # Mode A
            l_mae = l_val.get('mean_mae', float('nan'))
            t_mae = t_val.get('mean_mae', float('nan'))
        else:
            # Mode B/C/D: use 100-step result
            l_mae = l_val.get('100', float('nan'))
            t_mae = t_val.get('100', float('nan'))
        winner = 'L' if (not math.isnan(l_mae) and l_mae < t_mae) else 'T'
        print(f"  {mode.upper()}: L={l_mae:.6f}, T={t_mae:.6f} -> Winner: {winner}")

    return all_results


if __name__ == '__main__':
    main()

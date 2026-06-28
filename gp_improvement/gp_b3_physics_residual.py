"""GP-B3: Physics Residual Gaussian Process.

Uses the physics model (Meijaard dynamics) as a prior and trains the GP
on the residuals between physics predictions and true dynamics.

Key idea:
- Physics model provides good prior for most of the state space
- GP learns only the residual (unmodeled dynamics)
- Better generalization due to physics prior
- More data-efficient

Usage:
    python gp_b3_physics_residual.py

Expected output:
    - Physics residual GP MAE at 500 steps
    - Comparison with GP-B0 baseline
    - Residual statistics
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

import numpy as np
import math
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_evaluate import make_tau_func


def physics_predict(s, tau, dt=0.01):
    """Simple physics prediction using linearized Meijaard dynamics.

    This is a simplified version - in practice, use the full Meijaard model.
    """
    # Linearized dynamics around equilibrium
    # s = [phi, delta, phi_dot, delta_dot]
    phi, delta, phi_dot, delta_dot = s

    # Simplified physics: roll dynamics
    # phi_ddot ≈ g/h * phi + v/h * delta_dot
    g, h, v = 9.81, 0.97, 3.5
    phi_ddot = (g/h) * phi + (v/h) * delta_dot

    # Steer dynamics
    # delta_ddot ≈ tau / I_steer
    I_steer = 0.1  # approximate
    delta_ddot = tau / I_steer

    # Euler integration
    phi_next = phi + phi_dot * dt
    delta_next = delta + delta_dot * dt
    phi_dot_next = phi_dot + phi_ddot * dt
    delta_dot_next = delta_dot + delta_ddot * dt

    return np.array([phi_next, delta_next, phi_dot_next, delta_dot_next])


class PhysicsResidualGPBaseline:
    """GP baseline that learns residuals from physics model.

    The GP learns: residual = true_next_state - physics_next_state
    Final prediction: physics_next_state + residual
    """

    def __init__(self, max_samples=5000, seed=42):
        self.max_samples = max_samples
        self.seed = seed
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train GP on physics residuals."""
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        rng = np.random.RandomState(self.seed)
        n = len(states)
        if n > self.max_samples:
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        # Compute physics predictions
        physics_residuals = np.empty((len(idx), 4))
        for i, j in enumerate(idx):
            s_next_physics = physics_predict(states[j], actions[j])
            s_next_true = states[j] + deltas[j]
            physics_residuals[i] = (s_next_true - s_next_physics) / delta_std

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = physics_residuals

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self._gps = []

        for col in range(4):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gps.append(gp)
            print(f"    Physics residual GP output {col} trained")

    def predict(self, s, tau):
        """Predict next state using physics + GP residual."""
        # Physics prediction
        s_next_physics = physics_predict(s, tau)

        # GP residual prediction
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        residual_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        residual = residual_norm * self._delta_std

        return s_next_physics + residual


def evaluate_physics_residual_gp(baseline, state_std, action_std, delta_std,
                                 n_segments=5, eval_steps=None, seed_offset=0):
    """Evaluate physics residual GP baseline."""
    if eval_steps is None:
        eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    all_errors = {step: [] for step in eval_steps}

    for seg_i in range(n_segments):
        rng = np.random.RandomState(42 + seed_offset + seg_i)
        tau_func = make_tau_func(seg_i, 500)
        phi_init = rng.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_real = s0.copy()
        s_model = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_model = baseline.predict(s_model, tau)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_model[0] - s_real[0]))

    results = {}
    for step in eval_steps:
        if all_errors[step]:
            results[step] = {
                'mean': np.mean(all_errors[step]),
                'std': np.std(all_errors[step]),
                'n': len(all_errors[step])
            }
        else:
            results[step] = {'mean': float('nan'), 'std': float('nan'), 'n': 0}

    return results


def run_gp_b3(n_runs=3, n_segments=5):
    """Run GP-B3 physics residual baseline."""
    print("=" * 60)
    print("GP-B3: Physics Residual Gaussian Process")
    print("=" * 60)

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    all_run_results = []

    for run_i in range(n_runs):
        print(f"\n--- Run {run_i + 1}/{n_runs} ---")
        start_time = time.time()

        states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

        baseline = PhysicsResidualGPBaseline(max_samples=5000, seed=42)
        baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        results = evaluate_physics_residual_gp(
            baseline, state_std, action_std, delta_std,
            n_segments=n_segments, eval_steps=eval_steps, seed_offset=run_i * 10
        )

        all_run_results.append(results)
        elapsed = time.time() - start_time
        print(f"  500-step MAE: {results[500]['mean']:.5f} rad")
        print(f"  Time: {elapsed:.1f}s")

    # Aggregate across runs
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)

    aggregated = {}
    for step in eval_steps:
        means = [r[step]['mean'] for r in all_run_results if not np.isnan(r[step]['mean'])]
        if means:
            aggregated[step] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means),
                'n_runs': len(means)
            }
        else:
            aggregated[step] = {'mean': float('nan'), 'std': float('nan'),
                               'min': float('nan'), 'max': float('nan'), 'n_runs': 0}

    print(f"\n{'Step':>6} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print(f"{'-'*55}")
    for step in eval_steps:
        r = aggregated[step]
        print(f"{step:6d} | {r['mean']:10.5f} | {r['std']:10.5f} | {r['min']:10.5f} | {r['max']:10.5f}")

    return {
        'aggregated': aggregated,
        'all_runs': all_run_results,
        'config': {
            'n_runs': n_runs,
            'n_segments': n_segments,
            'n_train': 30000,
            'max_samples': 5000,
            'physics_model': 'simplified Meijaard linearized'
        }
    }


if __name__ == '__main__':
    results = run_gp_b3(n_runs=3, n_segments=5)

    import json
    output_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/gp_b3_results.json'

    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

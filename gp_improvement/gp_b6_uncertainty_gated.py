"""GP-B6: Uncertainty-Gated Residual GP.

Uses GP uncertainty to dynamically gate the residual contribution.
When the GP is uncertain, it reduces the residual contribution to avoid
amplifying errors.

Key ideas:
- GP provides both prediction and uncertainty
- Residual contribution is scaled by inverse uncertainty
- High uncertainty → low residual contribution
- Low uncertainty → full residual contribution

Usage:
    python gp_b6_uncertainty_gated.py

Expected output:
    - Uncertainty-gated GP MAE at 500 steps
    - Gating statistics
    - Comparison with GP-B0 baseline
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


class UncertaintyGatedGPBaseline:
    """GP baseline with uncertainty-gated residual.

    Uses GP's built-in uncertainty to gate residual contribution.
    """

    def __init__(self, max_samples=5000, seed=42, gate_threshold=0.1):
        self.max_samples = max_samples
        self.seed = seed
        self.gate_threshold = gate_threshold
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train GP baseline."""
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        rng = np.random.RandomState(self.seed)
        n = len(states)
        if n > self.max_samples:
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gps.append(gp)

    def predict(self, s, tau):
        """Predict next state."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        delta_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std

    def predict_with_uncertainty(self, s, tau):
        """Predict with uncertainty estimates."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        predictions = []
        uncertainties = []
        for gp in self._gps:
            pred, std = gp.predict(x, return_std=True)
            predictions.append(pred[0])
            uncertainties.append(std[0])

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)

        # Compute gating factor
        # High uncertainty → low gate value
        mean_uncertainty = np.mean(uncertainties)
        gate = 1.0 / (1.0 + mean_uncertainty / self.gate_threshold)

        next_state = s + predictions * self._delta_std * gate
        return next_state, uncertainties, gate


class UncertaintyGatedResidualGP:
    """GP that uses uncertainty to gate residual contribution.

    This combines a physics prior with a GP residual,
    where the residual contribution is gated by uncertainty.
    """

    def __init__(self, max_samples=5000, seed=42, gate_threshold=0.1):
        self.max_samples = max_samples
        self.seed = seed
        self.gate_threshold = gate_threshold
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train GP on residuals."""
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        rng = np.random.RandomState(self.seed)
        n = len(states)
        if n > self.max_samples:
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gps.append(gp)

    def predict(self, s, tau):
        """Predict with uncertainty gating."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        predictions = []
        uncertainties = []
        for gp in self._gps:
            pred, std = gp.predict(x, return_std=True)
            predictions.append(pred[0])
            uncertainties.append(std[0])

        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)

        # Compute gating factor
        mean_uncertainty = np.mean(uncertainties)
        gate = 1.0 / (1.0 + mean_uncertainty / self.gate_threshold)

        # Apply gated residual
        delta = predictions * self._delta_std * gate
        return s + delta, uncertainties, gate


def evaluate_uncertainty_gated(baseline, state_std, action_std, delta_std,
                               n_segments=5, eval_steps=None, seed_offset=0):
    """Evaluate uncertainty-gated GP."""
    if eval_steps is None:
        eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    all_errors = {step: [] for step in eval_steps}
    all_gates = {step: [] for step in eval_steps}

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
            s_model, uncertainty, gate = baseline.predict(s_model, tau)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_model[0] - s_real[0]))
                all_gates[step + 1].append(gate)

    results = {}
    for step in eval_steps:
        if all_errors[step]:
            results[step] = {
                'mean': np.mean(all_errors[step]),
                'std': np.std(all_errors[step]),
                'gate': np.mean(all_gates[step]),
                'n': len(all_errors[step])
            }
        else:
            results[step] = {'mean': float('nan'), 'std': float('nan'),
                            'gate': float('nan'), 'n': 0}

    return results


def run_gp_b6(n_runs=3, n_segments=5, gate_threshold=0.1):
    """Run GP-B6 uncertainty-gated residual."""
    print("=" * 60)
    print("GP-B6: Uncertainty-Gated Residual GP")
    print("=" * 60)

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    all_run_results = []

    for run_i in range(n_runs):
        print(f"\n--- Run {run_i + 1}/{n_runs} ---")
        start_time = time.time()

        states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

        baseline = UncertaintyGatedResidualGP(
            max_samples=5000, seed=42, gate_threshold=gate_threshold
        )
        baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        results = evaluate_uncertainty_gated(
            baseline, state_std, action_std, delta_std,
            n_segments=n_segments, eval_steps=eval_steps, seed_offset=run_i * 10
        )

        all_run_results.append(results)
        elapsed = time.time() - start_time
        print(f"  500-step MAE: {results[500]['mean']:.5f} rad")
        print(f"  500-step gate: {results[500]['gate']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Aggregate across runs
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)

    aggregated = {}
    for step in eval_steps:
        means = [r[step]['mean'] for r in all_run_results if not np.isnan(r[step]['mean'])]
        gates = [r[step]['gate'] for r in all_run_results if not np.isnan(r[step]['gate'])]
        if means:
            aggregated[step] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means),
                'gate': np.mean(gates),
                'n_runs': len(means)
            }
        else:
            aggregated[step] = {'mean': float('nan'), 'std': float('nan'),
                               'min': float('nan'), 'max': float('nan'),
                               'gate': float('nan'), 'n_runs': 0}

    print(f"\n{'Step':>6} | {'Mean':>10} | {'Std':>10} | {'Gate':>8}")
    print(f"{'-'*40}")
    for step in eval_steps:
        r = aggregated[step]
        print(f"{step:6d} | {r['mean']:10.5f} | {r['std']:10.5f} | {r['gate']:8.4f}")

    return {
        'aggregated': aggregated,
        'all_runs': all_run_results,
        'config': {
            'n_runs': n_runs,
            'n_segments': n_segments,
            'n_train': 30000,
            'max_samples': 5000,
            'gate_threshold': gate_threshold
        }
    }


if __name__ == '__main__':
    results = run_gp_b6(n_runs=3, n_segments=5, gate_threshold=0.1)

    import json
    output_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/gp_b6_results.json'

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

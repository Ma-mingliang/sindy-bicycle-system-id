"""GP-B5: Bootstrap GP Ensemble.

Uses bootstrap sampling to create multiple GP models and combines
their predictions for better uncertainty estimation.

Key ideas:
- Bootstrap sampling creates diverse training sets
- Multiple GPs capture epistemic uncertainty
- Average prediction reduces variance
- Can provide uncertainty estimates

Usage:
    python gp_b5_bootstrap_ensemble.py

Expected output:
    - Bootstrap ensemble MAE at 500 steps
    - Uncertainty estimates
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


class BootstrapGPEnsemble:
    """Bootstrap GP Ensemble.

    Creates multiple GPs using bootstrap sampling and combines predictions.
    """

    def __init__(self, max_samples=5000, n_models=5, seed=42):
        self.max_samples = max_samples
        self.n_models = n_models
        self.seed = seed
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train bootstrap GP ensemble."""
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        rng = np.random.RandomState(self.seed)
        n = len(states)

        self._ensemble = []

        for model_i in range(self.n_models):
            # Bootstrap sample
            if n > self.max_samples:
                idx = rng.choice(n, self.max_samples, replace=True)  # with replacement
            else:
                idx = rng.choice(n, n, replace=True)

            X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
            Y = deltas[idx] / delta_std

            gps = []
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            for col in range(4):
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
                gp.fit(X, Y[:, col])
                gps.append(gp)

            self._ensemble.append(gps)
            print(f"    Bootstrap GP model {model_i + 1}/{self.n_models} trained")

    def predict(self, s, tau):
        """Predict next state using ensemble average."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        predictions = []
        for gps in self._ensemble:
            delta_norm = np.array([gp.predict(x)[0] for gp in gps])
            predictions.append(delta_norm)

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)

        return s + mean_prediction * self._delta_std

    def predict_with_uncertainty(self, s, tau):
        """Predict with uncertainty estimates."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        predictions = []
        for gps in self._ensemble:
            delta_norm = np.array([gp.predict(x)[0] for gp in gps])
            predictions.append(delta_norm)

        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        next_state = s + mean_prediction * self._delta_std
        uncertainty = std_prediction * self._delta_std

        return next_state, uncertainty


def evaluate_bootstrap_ensemble(baseline, state_std, action_std, delta_std,
                                n_segments=5, eval_steps=None, seed_offset=0):
    """Evaluate bootstrap GP ensemble."""
    if eval_steps is None:
        eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    all_errors = {step: [] for step in eval_steps}
    all_uncertainties = {step: [] for step in eval_steps}

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
            s_model, uncertainty = baseline.predict_with_uncertainty(s_model, tau)

            if abs(s_real[0]) > math.pi / 3:
                break

            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_model[0] - s_real[0]))
                all_uncertainties[step + 1].append(np.mean(uncertainty))

    results = {}
    for step in eval_steps:
        if all_errors[step]:
            results[step] = {
                'mean': np.mean(all_errors[step]),
                'std': np.std(all_errors[step]),
                'uncertainty': np.mean(all_uncertainties[step]),
                'n': len(all_errors[step])
            }
        else:
            results[step] = {'mean': float('nan'), 'std': float('nan'),
                            'uncertainty': float('nan'), 'n': 0}

    return results


def run_gp_b5(n_runs=3, n_segments=5, n_models=5):
    """Run GP-B5 bootstrap ensemble."""
    print("=" * 60)
    print("GP-B5: Bootstrap GP Ensemble")
    print("=" * 60)

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    all_run_results = []

    for run_i in range(n_runs):
        print(f"\n--- Run {run_i + 1}/{n_runs} ---")
        start_time = time.time()

        states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

        baseline = BootstrapGPEnsemble(max_samples=5000, n_models=n_models, seed=42)
        baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        results = evaluate_bootstrap_ensemble(
            baseline, state_std, action_std, delta_std,
            n_segments=n_segments, eval_steps=eval_steps, seed_offset=run_i * 10
        )

        all_run_results.append(results)
        elapsed = time.time() - start_time
        print(f"  500-step MAE: {results[500]['mean']:.5f} rad")
        print(f"  500-step uncertainty: {results[500]['uncertainty']:.5f} rad")
        print(f"  Time: {elapsed:.1f}s")

    # Aggregate across runs
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)

    aggregated = {}
    for step in eval_steps:
        means = [r[step]['mean'] for r in all_run_results if not np.isnan(r[step]['mean'])]
        uncertainties = [r[step]['uncertainty'] for r in all_run_results
                        if not np.isnan(r[step]['uncertainty'])]
        if means:
            aggregated[step] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means),
                'uncertainty': np.mean(uncertainties),
                'n_runs': len(means)
            }
        else:
            aggregated[step] = {'mean': float('nan'), 'std': float('nan'),
                               'min': float('nan'), 'max': float('nan'),
                               'uncertainty': float('nan'), 'n_runs': 0}

    print(f"\n{'Step':>6} | {'Mean':>10} | {'Std':>10} | {'Uncertainty':>12}")
    print(f"{'-'*45}")
    for step in eval_steps:
        r = aggregated[step]
        print(f"{step:6d} | {r['mean']:10.5f} | {r['std']:10.5f} | {r['uncertainty']:12.5f}")

    return {
        'aggregated': aggregated,
        'all_runs': all_run_results,
        'config': {
            'n_runs': n_runs,
            'n_segments': n_segments,
            'n_train': 30000,
            'max_samples': 5000,
            'n_models': n_models
        }
    }


if __name__ == '__main__':
    results = run_gp_b5(n_runs=3, n_segments=5, n_models=5)

    import json
    output_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/gp_b5_results.json'

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

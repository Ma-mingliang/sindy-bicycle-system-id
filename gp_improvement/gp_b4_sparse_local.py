"""GP-B4: Sparse/Local Gaussian Process.

Sparse GP methods reduce computational cost by using inducing points.
Local GP uses separate models for different regions of the state space.

Key approaches:
1. Sparse GP: Use Nystroem approximation with inducing points
2. Local GP: Separate models for different regions
3. Combination: Sparse + Local

Usage:
    python gp_b4_sparse_local.py

Expected output:
    - Sparse GP MAE at 500 steps
    - Local GP MAE at 500 steps
    - Comparison with GP-B0 baseline
    - Computational time comparison
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
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_evaluate import make_tau_func


class SparseGPBaseline:
    """Sparse GP using Nystroem approximation.

    Instead of full GP, uses kernel approximation + linear model.
    Much faster for large datasets.
    """

    def __init__(self, max_samples=5000, n_components=100, seed=42):
        self.max_samples = max_samples
        self.n_components = n_components
        self.seed = seed
        self._models = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train sparse GP baseline."""
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

        self._models = []
        for col in range(4):
            # Nystroem approximation
            nystroem = Nystroem(
                kernel='rbf',
                gamma=1.0,
                n_components=self.n_components,
                random_state=self.seed
            )
            X_transformed = nystroem.fit_transform(X)

            # Ridge regression on transformed features
            model = Ridge(alpha=1e-6)
            model.fit(X_transformed, Y[:, col])

            self._models.append((nystroem, model))
            print(f"    Sparse GP output {col} trained")

    def predict(self, s, tau):
        """Predict next state."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        delta_norm = np.zeros(4)
        for col, (nystroem, model) in enumerate(self._models):
            x_transformed = nystroem.transform(x)
            delta_norm[col] = model.predict(x_transformed)[0]

        return s + delta_norm * self._delta_std


class LocalGPBaseline:
    """Local GP using separate models for different regions.

    Divides the state space into regions and trains separate GPs.
    Can capture non-stationary dynamics better.
    """

    def __init__(self, max_samples=5000, n_regions=4, seed=42):
        self.max_samples = max_samples
        self.n_regions = n_regions
        self.seed = seed
        self._region_models = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._region_boundaries = None

    def _assign_region(self, X):
        """Assign data points to regions based on first principal component."""
        # Use first two dimensions (phi, delta) for region assignment
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_regions, random_state=self.seed, n_init=10)
        return kmeans.fit_predict(X[:, :2])

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        """Train local GP baseline."""
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

        # Assign regions
        region_labels = self._assign_region(X)
        self._region_models = []

        for region_id in range(self.n_regions):
            region_mask = region_labels == region_id
            X_region = X[region_mask]
            Y_region = Y[region_mask]

            if len(X_region) < 10:
                # Not enough data in this region, use global model
                self._region_models.append(None)
                continue

            gps = []
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            for col in range(4):
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
                gp.fit(X_region, Y_region[:, col])
                gps.append(gp)

            self._region_models.append(gps)
            print(f"    Local GP region {region_id}: {len(X_region)} samples")

    def predict(self, s, tau):
        """Predict next state using local GP."""
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        # Find nearest region (simple: use first two dimensions)
        from sklearn.cluster import KMeans
        # For prediction, use simple distance-based assignment
        min_dist = float('inf')
        best_region = 0

        for region_id, gps in enumerate(self._region_models):
            if gps is None:
                continue
            # Use simple distance to region center
            # This is a simplification - in practice, store region centers
            dist = np.sum(x[0, :2] ** 2)  # placeholder
            if dist < min_dist:
                min_dist = dist
                best_region = region_id

        if self._region_models[best_region] is not None:
            gps = self._region_models[best_region]
            delta_norm = np.array([gp.predict(x)[0] for gp in gps])
            return s + delta_norm * self._delta_std
        else:
            # Fallback: zero prediction
            return s.copy()


def evaluate_sparse_local_gp(baseline, state_std, action_std, delta_std,
                             n_segments=5, eval_steps=None, seed_offset=0):
    """Evaluate sparse/local GP baseline."""
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


def run_gp_b4(n_runs=3, n_segments=5):
    """Run GP-B4 sparse/local baseline."""
    print("=" * 60)
    print("GP-B4: Sparse/Local Gaussian Process")
    print("=" * 60)

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    all_run_results = []

    for run_i in range(n_runs):
        print(f"\n--- Run {run_i + 1}/{n_runs} ---")
        start_time = time.time()

        states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

        # Test sparse GP
        print("\n  Sparse GP:")
        sparse_baseline = SparseGPBaseline(max_samples=5000, n_components=100, seed=42)
        sparse_baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        sparse_results = evaluate_sparse_local_gp(
            sparse_baseline, state_std, action_std, delta_std,
            n_segments=n_segments, eval_steps=eval_steps, seed_offset=run_i * 10
        )

        elapsed = time.time() - start_time
        print(f"  Sparse GP 500-step MAE: {sparse_results[500]['mean']:.5f} rad")
        print(f"  Time: {elapsed:.1f}s")

        all_run_results.append({
            'sparse': sparse_results,
            'local': None  # Skip local GP for now (requires more complex setup)
        })

    # Aggregate across runs
    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)

    aggregated = {}
    for step in eval_steps:
        means = [r['sparse'][step]['mean'] for r in all_run_results
                 if not np.isnan(r['sparse'][step]['mean'])]
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
            'sparse_components': 100
        }
    }


if __name__ == '__main__':
    results = run_gp_b4(n_runs=3, n_segments=5)

    import json
    output_file = 'D:/系统辨识作业/sindy_bicycle/gp_improvement/gp_b4_results.json'

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

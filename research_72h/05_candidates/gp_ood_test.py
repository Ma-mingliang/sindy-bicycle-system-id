"""GP OOD (Out-of-Distribution) Test.

Comprehensive evaluation of GP model generalization to out-of-distribution data.
Tests GP on scenarios with different speeds, steering, and initial conditions
that deviate from the training distribution.

Compares GP vs Neural ODE robustness under distribution shift.

Usage:
    python gp_ood_test.py

Expected output:
    - OOD detection precision/recall
    - GP performance on ID vs OOD segments
    - GP vs Neural ODE comparison on OOD scenarios
    - Uncertainty calibration analysis
    - Generalization capability report
"""

import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

# ============================================================
# Constants
# ============================================================
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
DATA_PATH = 'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz'
DT = 1.0 / 30.0

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Neural ODE model path
NODE_MODEL_PATH = 'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/v9_seed42.pt'


# ============================================================
# Utility Functions
# ============================================================
def check_survival(state):
    """Check if state is within physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(seed=42):
    """Load and split dataset into train/test episodes."""
    data = np.load(DATA_PATH, allow_pickle=True)
    obs = data['obs'][:, IDX_7D_FROM_8D]
    next_obs = data['next_obs'][:, IDX_7D_FROM_8D]
    deltas = next_obs - obs
    done = data['done']
    action = data['action']

    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    episodes = []
    for ep_idx in range(n_episodes):
        start = episode_starts[ep_idx]
        end = episode_ends[ep_idx] + 1
        episodes.append({
            'obs': obs[start:end],
            'action': action[start:end],
            'deltas': deltas[start:end],
            'length': end - start,
        })

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    state_std = np.std(train_obs, axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(train_action)
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(train_deltas, axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    return {
        'episodes': episodes,
        'train_eps': train_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'train_obs': train_obs,
        'train_action': train_action,
        'train_deltas': train_deltas,
    }


# ============================================================
# GP Model
# ============================================================
class GPModel:
    """Gaussian Process model for bicycle dynamics prediction."""

    def __init__(self, n_samples=1000, kernel=None, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        self.kernel = kernel or (ConstantKernel(1.0) * RBF(length_scale=1.0))
        self.gps = []
        self.state_std = None
        self.action_std = None
        self.delta_std = None
        self.train_X = None

    def train(self, data):
        """Train GP model on data dict from load_data()."""
        np.random.seed(self.seed)
        self.state_std = data['state_std']
        self.action_std = data['action_std']
        self.delta_std = data['delta_std']

        n_total = len(data['train_obs'])
        indices = np.random.choice(n_total, min(self.n_samples, n_total), replace=False)

        X = np.hstack([
            data['train_obs'][indices] / self.state_std,
            data['train_action'][indices].reshape(-1, 1) / self.action_std
        ])
        Y = data['train_deltas'][indices] / self.delta_std
        self.train_X = X.copy()

        self.gps = []
        for i in range(STATE_DIM):
            print(f"  Training GP for {STATE_NAMES_7D[i]}...", end="", flush=True)
            t0 = time.time()
            gp = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=0,
                random_state=self.seed,
                alpha=1e-6
            )
            gp.fit(X, Y[:, i])
            elapsed = time.time() - t0
            print(f" done ({elapsed:.1f}s)")
            self.gps.append(gp)

        return self

    def predict(self, s, tau):
        """Predict next state."""
        x = np.hstack([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        delta_pred = np.array([gp.predict(x)[0] for gp in self.gps])
        return s + delta_pred * self.delta_std * DT

    def predict_with_uncertainty(self, s, tau):
        """Predict next state with uncertainty (std dev)."""
        x = np.hstack([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        means = []
        stds = []
        for gp in self.gps:
            mean, std = gp.predict(x, return_std=True)
            means.append(mean[0])
            stds.append(std[0])
        delta_mean = np.array(means) * self.delta_std * DT
        delta_std = np.array(stds) * self.delta_std * DT
        return s + delta_mean, delta_std

    def predict_variance(self, s, tau):
        """Return prediction variance (scalar, sum across dims)."""
        _, std = self.predict_with_uncertainty(s, tau)
        return float(np.sum(std ** 2))


# ============================================================
# Neural ODE Model (optional)
# ============================================================
class NeuralODEWrapper:
    """Wrapper to load and use trained Neural ODE v9 model."""

    def __init__(self, model_path=NODE_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.loaded = False

    def try_load(self):
        """Attempt to load the Neural ODE model."""
        try:
            import torch
            sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/canonical_node')
            from neural_ode_v9 import NeuralODEV9
            self.model = NeuralODEV9()
            self.model.load(self.model_path)
            self.loaded = True
            print("  Neural ODE v9 loaded successfully")
            return True
        except Exception as e:
            print(f"  Neural ODE load failed: {e}")
            print("  Skipping Neural ODE comparison")
            return False

    def predict(self, s, tau):
        if not self.loaded:
            raise RuntimeError("Neural ODE not loaded")
        return self.model.predict(s, tau)


# ============================================================
# OOD Detector
# ============================================================
class OODDetector:
    """Z-score based OOD detection on state space."""

    def __init__(self, train_obs, threshold=3.0):
        self.mean = np.mean(train_obs, axis=0)
        self.std = np.std(train_obs, axis=0) + 1e-8
        self.threshold = threshold

    def is_ood(self, x):
        """Check if single state is OOD."""
        z = np.abs(x - self.mean) / self.std
        return bool(np.max(z) > self.threshold)

    def ood_score(self, x):
        """Return max z-score (higher = more OOD)."""
        z = np.abs(x - self.mean) / self.std
        return float(np.max(z))

    def is_batch_ood(self, X):
        """Check batch of states."""
        z = np.abs(X - self.mean) / self.std
        return np.max(z, axis=1) > self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold


# ============================================================
# OOD Scenario Generators
# ============================================================
class OODScenarios:
    """Generate OOD test scenarios from existing test episodes."""

    def __init__(self, episodes, test_eps, train_obs, state_std):
        self.episodes = episodes
        self.test_eps = test_eps
        self.train_obs = train_obs
        self.state_std = state_std

        # Compute training distribution statistics
        self.train_mean = np.mean(train_obs, axis=0)
        self.train_std = np.std(train_obs, axis=0)
        self.train_min = np.min(train_obs, axis=0)
        self.train_max = np.max(train_obs, axis=0)

    def identify_ood_segments(self, ood_detector, min_length=200, n_sigma=2.5):
        """Identify segments from test episodes that are OOD.

        Strategy: Find contiguous regions within test episodes where
        the state is far from training distribution.
        """
        ood_segments = []
        id_segments = []

        for ep_idx in self.test_eps:
            ep = self.episodes[ep_idx]
            obs = ep['obs']
            action = ep['action']
            n = ep['length']

            if n < min_length:
                continue

            # Score each timestep
            scores = np.array([ood_detector.ood_score(obs[t]) for t in range(n)])

            # Find OOD regions (sliding window)
            window = 50
            for start in range(0, n - min_length, window // 2):
                end = start + min_length
                window_scores = scores[start:end]
                mean_score = np.mean(window_scores)
                max_score = np.max(window_scores)

                seg = {
                    'ep_idx': ep_idx,
                    'start': start,
                    'end': end,
                    'obs': obs[start:end],
                    'action': action[start:end],
                    'mean_ood_score': mean_score,
                    'max_ood_score': max_score,
                    'length': end - start,
                }

                # Classify as OOD or ID based on mean score
                if mean_score > n_sigma:
                    ood_segments.append(seg)
                elif mean_score < n_sigma * 0.5:
                    id_segments.append(seg)

        # Sort by OOD score descending, take top N
        ood_segments.sort(key=lambda s: s['mean_ood_score'], reverse=True)
        id_segments.sort(key=lambda s: s['mean_ood_score'])

        return ood_segments, id_segments

    def generate_synthetic_ood(self, test_segments, perturbation_type, magnitude):
        """Generate synthetic OOD by perturbing test segments.

        Args:
            test_segments: list of segments from test episodes
            perturbation_type: 'speed', 'lateral', 'heading', 'action'
            magnitude: perturbation multiplier (1.0 = no change)

        Returns:
            list of perturbed segments
        """
        perturbed = []
        for seg in test_segments[:5]:  # Limit to 5 segments
            obs = seg['obs'].copy()
            action = seg['action'].copy()

            if perturbation_type == 'speed':
                # Scale velocity (index 2) by magnitude
                obs[:, 2] = obs[:, 2] * magnitude
            elif perturbation_type == 'lateral':
                # Add lateral offset to e_y (index 0)
                obs[:, 0] = obs[:, 0] + magnitude
            elif perturbation_type == 'heading':
                # Add heading offset to e_psi (index 1)
                obs[:, 1] = obs[:, 1] + magnitude
            elif perturbation_type == 'action':
                # Scale action magnitude
                action = action * magnitude
            elif perturbation_type == 'lean':
                # Add lean angle to theta (index 3)
                obs[:, 3] = obs[:, 3] + magnitude
            elif perturbation_type == 'combined':
                # Multi-dimensional perturbation
                obs[:, 0] = obs[:, 0] + magnitude * 0.3  # e_y
                obs[:, 1] = obs[:, 1] + magnitude * 0.2  # e_psi
                obs[:, 2] = obs[:, 2] * (1.0 + magnitude * 0.1)  # v
                action = action * (1.0 + magnitude * 0.5)  # action

            perturbed.append({
                'ep_idx': seg['ep_idx'],
                'start': seg['start'],
                'end': seg['end'],
                'obs': obs,
                'action': action,
                'mean_ood_score': seg['mean_ood_score'],
                'max_ood_score': seg['max_ood_score'],
                'length': seg['length'],
                'perturbation': perturbation_type,
                'magnitude': magnitude,
            })

        return perturbed


# ============================================================
# Evaluation
# ============================================================
def evaluate_model_on_segments(model, segments, state_std, horizons, model_name="GP"):
    """Evaluate model on list of segments at multiple horizons.

    Args:
        model: object with predict(s, tau) method
        segments: list of segment dicts with 'obs' and 'action'
        state_std: state standard deviations for normalization
        horizons: list of horizon values
        model_name: name for logging

    Returns:
        dict of results per horizon
    """
    results = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}
        step_errors_all = []

        for seg in segments:
            obs = seg['obs']
            actions = seg['action'].flatten()
            n = min(h, len(actions))

            s_cur = obs[0].copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions[step])

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next):
                        survived = False
                        break

                    if step + 1 < len(obs):
                        err = np.abs(s_next - obs[step + 1]) / state_std
                        step_errors.append(err)

                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors)
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                step_errors_all.append(errors)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_median': float(np.nanmedian(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
            'n_segments': len(segments),
            'n_valid': len(valid_nmae),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                    'std': float(np.nanstd(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


def evaluate_gp_uncertainty_calibration(gp, segments, state_std, n_segments=20):
    """Evaluate how well GP uncertainty correlates with actual error.

    For well-calibrated uncertainty:
    - High uncertainty should correlate with high error
    - Low uncertainty should correlate with low error
    """
    uncertainty_error_pairs = []

    for seg in segments[:n_segments]:
        obs = seg['obs']
        actions = seg['action'].flatten()
        n = min(100, len(actions))

        s_cur = obs[0].copy()

        for step in range(n):
            try:
                s_next_pred, pred_std = gp.predict_with_uncertainty(s_cur, actions[step])

                if step + 1 < len(obs):
                    actual_error = np.abs(s_next_pred - obs[step + 1]) / state_std
                    uncertainty = pred_std / (state_std * DT + 1e-10)

                    uncertainty_error_pairs.append({
                        'uncertainty': float(np.mean(uncertainty)),
                        'error': float(np.mean(actual_error)),
                        'per_dim_uncertainty': uncertainty.tolist(),
                        'per_dim_error': actual_error.tolist(),
                    })

                # Use ground truth for next step to avoid compounding
                if step + 1 < len(obs):
                    s_cur = obs[step + 1].copy()
                else:
                    break
            except Exception:
                break

    if not uncertainty_error_pairs:
        return {'correlation': float('nan'), 'calibration_error': float('nan')}

    uncertainties = np.array([p['uncertainty'] for p in uncertainty_error_pairs])
    errors = np.array([p['error'] for p in uncertainty_error_pairs])

    # Pearson correlation
    if np.std(uncertainties) > 1e-10 and np.std(errors) > 1e-10:
        correlation = float(np.corrcoef(uncertainties, errors)[0, 1])
    else:
        correlation = 0.0

    # Calibration: ratio of actual error to predicted uncertainty
    mean_uncertainty = np.mean(uncertainties)
    mean_error = np.mean(errors)
    calibration_ratio = mean_error / (mean_uncertainty + 1e-10)

    # Binned calibration - bin by error magnitude for more meaningful grouping
    n_bins = 5
    unc_range = np.max(uncertainties) - np.min(uncertainties)
    if unc_range > 1e-8:
        # Use uncertainty-based binning when there's sufficient variance
        bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        n_bins = len(bin_edges) - 1
    else:
        # Fallback: bin by error magnitude when uncertainty is nearly constant
        bin_edges = np.percentile(errors, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        n_bins = len(bin_edges) - 1

    binned_calibration = []
    for i in range(n_bins):
        if unc_range > 1e-8:
            mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        else:
            mask = (errors >= bin_edges[i]) & (errors < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_mean_unc = np.mean(uncertainties[mask])
            bin_mean_err = np.mean(errors[mask])
            binned_calibration.append({
                'bin': i,
                'uncertainty_range': [float(bin_edges[i]), float(bin_edges[i + 1])],
                'mean_uncertainty': float(bin_mean_unc),
                'mean_error': float(bin_mean_err),
                'ratio': float(bin_mean_err / (bin_mean_unc + 1e-10)),
                'count': int(np.sum(mask)),
            })

    return {
        'correlation': correlation,
        'calibration_ratio': calibration_ratio,
        'mean_uncertainty': float(mean_uncertainty),
        'mean_error': float(mean_error),
        'binned_calibration': binned_calibration,
        'n_pairs': len(uncertainty_error_pairs),
    }


def compute_ood_detection_metrics(ood_detector, train_obs, test_id_obs, test_ood_obs):
    """Compute precision/recall for OOD detection.

    Args:
        ood_detector: OODDetector instance
        train_obs: training observations (for reference)
        test_id_obs: known in-distribution test observations
        test_ood_obs: known out-of-distribution test observations

    Returns:
        dict of detection metrics
    """
    # ID samples should NOT be flagged
    id_predictions = ood_detector.is_batch_ood(test_id_obs)
    # OOD samples SHOULD be flagged
    ood_predictions = ood_detector.is_batch_ood(test_ood_obs)

    # True negatives: ID correctly not flagged
    tn = int(np.sum(~id_predictions))
    # False positives: ID incorrectly flagged as OOD
    fp = int(np.sum(id_predictions))
    # True positives: OOD correctly flagged
    tp = int(np.sum(ood_predictions))
    # False negatives: OOD not flagged
    fn = int(np.sum(~ood_predictions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'n_id_samples': len(test_id_obs),
        'n_ood_samples': len(test_ood_obs),
    }


def sweep_ood_threshold(ood_detector, test_id_obs, test_ood_obs):
    """Sweep OOD detection threshold to find optimal operating point."""
    # Compute scores for all samples
    id_scores = np.array([ood_detector.ood_score(x) for x in test_id_obs])
    ood_scores = np.array([ood_detector.ood_score(x) for x in test_ood_obs])

    thresholds = np.linspace(0.5, 5.0, 20)
    results = []

    for t in thresholds:
        ood_detector.set_threshold(t)
        tp = int(np.sum(ood_scores > t))
        fn = int(np.sum(ood_scores <= t))
        fp = int(np.sum(id_scores > t))
        tn = int(np.sum(id_scores <= t))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'threshold': float(t),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        })

    # Reset threshold
    ood_detector.set_threshold(3.0)

    # Find best F1 threshold
    best = max(results, key=lambda r: r['f1'])
    return results, best


# ============================================================
# Main Experiment
# ============================================================
def main():
    print("=" * 70)
    print("GP OOD (Out-of-Distribution) Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {}

    # --------------------------------------------------------
    # 1. Load Data
    # --------------------------------------------------------
    print("1. Loading data...")
    data = load_data(seed=42)
    train_obs = data['train_obs']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    print(f"   Train episodes: {len(data['train_eps'])}, Test episodes: {len(data['test_eps'])}")
    print(f"   Train samples: {len(train_obs)}")
    print(f"   State std: {state_std}")

    # --------------------------------------------------------
    # 2. Train GP Model
    # --------------------------------------------------------
    print("\n2. Training GP model (1000 samples, RBF kernel)...")
    gp = GPModel(n_samples=1000, seed=42)
    t0 = time.time()
    gp.train(data)
    gp_train_time = time.time() - t0
    print(f"   GP training time: {gp_train_time:.1f}s")

    # --------------------------------------------------------
    # 3. Build OOD Detector
    # --------------------------------------------------------
    print("\n3. Building OOD detector...")
    ood_detector = OODDetector(train_obs, threshold=3.0)

    # Create synthetic OOD test set by perturbing test data
    # Use extreme regions as "known OOD"
    n_ood = min(2000, len(train_obs))
    rng = np.random.RandomState(42)

    # Generate OOD samples: push states beyond 3-sigma
    ood_samples = []
    for _ in range(n_ood):
        idx = rng.randint(len(train_obs))
        sample = train_obs[idx].copy()
        # Perturb 2-3 random dimensions beyond 3-sigma
        dims = rng.choice(STATE_DIM, size=rng.randint(2, 4), replace=False)
        for d in dims:
            sign = rng.choice([-1, 1])
            sample[d] = train_obs[idx, d] + sign * rng.uniform(3.5, 5.0) * state_std[d]
        ood_samples.append(sample)
    ood_samples = np.array(ood_samples)

    # ID test samples from held-out test episodes
    id_samples = []
    for ep_idx in data['test_eps']:
        ep = data['episodes'][ep_idx]
        n = min(500, ep['length'])
        id_samples.append(ep['obs'][:n])
    id_samples = np.concatenate(id_samples, axis=0)[:n_ood]

    print(f"   ID test samples: {len(id_samples)}")
    print(f"   OOD test samples: {len(ood_samples)}")

    # --------------------------------------------------------
    # 4. OOD Detection Metrics
    # --------------------------------------------------------
    print("\n4. Computing OOD detection metrics...")
    detection_metrics = compute_ood_detection_metrics(
        ood_detector, train_obs, id_samples, ood_samples
    )
    print(f"   Precision: {detection_metrics['precision']:.4f}")
    print(f"   Recall: {detection_metrics['recall']:.4f}")
    print(f"   F1 Score: {detection_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {detection_metrics['accuracy']:.4f}")
    all_results['ood_detection'] = detection_metrics

    # Threshold sweep
    print("\n5. OOD threshold sweep...")
    threshold_results, best_threshold = sweep_ood_threshold(
        ood_detector, id_samples, ood_samples
    )
    print(f"   Best threshold: {best_threshold['threshold']:.2f} "
          f"(F1={best_threshold['f1']:.4f}, P={best_threshold['precision']:.4f}, "
          f"R={best_threshold['recall']:.4f})")
    all_results['threshold_sweep'] = {
        'results': threshold_results,
        'best': best_threshold,
    }

    # --------------------------------------------------------
    # 6. Identify Natural OOD Segments in Test Data
    # --------------------------------------------------------
    print("\n6. Identifying OOD segments in test episodes...")
    scenario_gen = OODScenarios(
        data['episodes'], data['test_eps'], train_obs, state_std
    )
    ood_segments, id_segments = scenario_gen.identify_ood_segments(
        ood_detector, min_length=200, n_sigma=2.5
    )
    print(f"   Natural OOD segments: {len(ood_segments)}")
    print(f"   Natural ID segments: {len(id_segments)}")

    if ood_segments:
        ood_scores = [s['mean_ood_score'] for s in ood_segments[:5]]
        print(f"   Top OOD scores: {[f'{s:.2f}' for s in ood_scores]}")

    all_results['natural_segments'] = {
        'n_ood': len(ood_segments),
        'n_id': len(id_segments),
        'top_ood_scores': [s['mean_ood_score'] for s in ood_segments[:10]],
    }

    # --------------------------------------------------------
    # 7. GP Performance: ID vs OOD
    # --------------------------------------------------------
    horizons = [1, 10, 50, 100, 200, 500]
    n_eval_segments = min(5, len(ood_segments), len(id_segments))

    if n_eval_segments > 0:
        print(f"\n7. Evaluating GP on ID vs OOD segments ({n_eval_segments} each)...")
        id_eval = id_segments[:n_eval_segments]
        ood_eval = ood_segments[:n_eval_segments]

        print("   Evaluating on ID segments...")
        gp_id_results = evaluate_model_on_segments(
            gp, id_eval, state_std, horizons, "GP-ID"
        )
        print("   Evaluating on OOD segments...")
        gp_ood_results = evaluate_model_on_segments(
            gp, ood_eval, state_std, horizons, "GP-OOD"
        )

        print(f"\n   {'Horizon':<10} {'GP-ID NMAE':<14} {'GP-OOD NMAE':<14} {'ID Survival':<14} {'OOD Survival':<14}")
        print("   " + "-" * 66)
        for h in horizons:
            id_nmae = gp_id_results[h]['nmae_mean']
            ood_nmae = gp_ood_results[h]['nmae_mean']
            id_surv = gp_id_results[h]['survival_rate']
            ood_surv = gp_ood_results[h]['survival_rate']
            id_str = f"{id_nmae:.4f}" if not np.isnan(id_nmae) else "N/A"
            ood_str = f"{ood_nmae:.4f}" if not np.isnan(ood_nmae) else "N/A"
            print(f"   H={h:<7} {id_str:<14} {ood_str:<14} {id_surv:<14.2%} {ood_surv:<14.2%}")

        all_results['gp_id_vs_ood'] = {
            'id': {str(h): gp_id_results[h] for h in horizons},
            'ood': {str(h): gp_ood_results[h] for h in horizons},
        }

        # Compute degradation ratio
        degradation = {}
        for h in horizons:
            id_val = gp_id_results[h]['nmae_mean']
            ood_val = gp_ood_results[h]['nmae_mean']
            if not np.isnan(id_val) and not np.isnan(ood_val) and id_val > 0:
                degradation[h] = (ood_val - id_val) / id_val
            else:
                degradation[h] = float('nan')

        print(f"\n   OOD degradation ratio (higher = worse):")
        for h in horizons:
            d = degradation[h]
            d_str = f"{d:.2%}" if not np.isnan(d) else "N/A"
            print(f"   H={h}: {d_str}")

        all_results['gp_id_vs_ood']['degradation'] = {str(h): v for h, v in degradation.items()}
    else:
        print("\n7. Not enough OOD/ID segments for comparison. Using synthetic perturbations.")
        # Fallback: use test episodes as ID, synthetic perturbations as OOD
        fallback_segments = []
        for ep_idx in data['test_eps']:
            ep = data['episodes'][ep_idx]
            if ep['length'] >= 200:
                fallback_segments.append({
                    'obs': ep['obs'][:200],
                    'action': ep['action'][:200],
                    'length': 200,
                })
        fallback_segments = fallback_segments[:5]

        if fallback_segments:
            gp_id_results = evaluate_model_on_segments(
                gp, fallback_segments, state_std, horizons, "GP-ID"
            )
            all_results['gp_id_vs_ood'] = {
                'id': {str(h): gp_id_results[h] for h in horizons},
                'ood': {},
                'degradation': {},
                'note': 'No natural OOD segments found; ID results only',
            }
            ood_segments = []
            id_segments = fallback_segments
        else:
            print("   ERROR: No suitable test segments found!")
            ood_segments = []
            id_segments = []

    # --------------------------------------------------------
    # 8. Synthetic OOD Perturbation Analysis
    # --------------------------------------------------------
    print("\n8. Synthetic OOD perturbation analysis...")
    perturbation_types = ['speed', 'lateral', 'heading', 'action', 'lean', 'combined']
    perturbation_magnitudes = [0.1, 0.5, 1.0, 2.0]

    # Use ID segments as base for perturbation
    base_segments = id_segments[:3] if id_segments else []
    if not base_segments:
        # Fallback to test episodes
        for ep_idx in data['test_eps']:
            ep = data['episodes'][ep_idx]
            if ep['length'] >= 200:
                base_segments.append({
                    'obs': ep['obs'][:200],
                    'action': ep['action'][:200],
                    'length': 200,
                    'ep_idx': ep_idx,
                    'start': 0,
                    'end': 200,
                    'mean_ood_score': 0.0,
                    'max_ood_score': 0.0,
                })
        base_segments = base_segments[:3]

    perturbation_results = {}
    if base_segments:
        for ptype in perturbation_types:
            perturbation_results[ptype] = {}
            for mag in perturbation_magnitudes:
                perturbed = scenario_gen.generate_synthetic_ood(
                    base_segments, ptype, mag
                )
                if perturbed:
                    # Use H=100 for perturbation analysis
                    result = evaluate_model_on_segments(
                        gp, perturbed, state_std, [100], f"GP-{ptype}-{mag}"
                    )
                    perturbation_results[ptype][str(mag)] = result[100]

        print(f"\n   {'Type':<12}", end="")
        for mag in perturbation_magnitudes:
            print(f"  mag={mag:<5}", end="")
        print("  NMAE")
        print("   " + "-" * 70)
        for ptype in perturbation_types:
            print(f"   {ptype:<12}", end="")
            for mag in perturbation_magnitudes:
                key = str(mag)
                if key in perturbation_results[ptype]:
                    nmae = perturbation_results[ptype][key]['nmae_mean']
                    nmae_str = f"{nmae:.4f}" if not np.isnan(nmae) else "  N/A "
                    print(f"  {nmae_str}", end="")
                else:
                    print(f"    N/A ", end="")
            print()

    all_results['perturbation_analysis'] = perturbation_results

    # --------------------------------------------------------
    # 9. GP Uncertainty Calibration
    # --------------------------------------------------------
    print("\n9. GP uncertainty calibration analysis...")
    calibration_segments = id_segments[:5] if id_segments else []
    if not calibration_segments:
        for ep_idx in data['test_eps']:
            ep = data['episodes'][ep_idx]
            if ep['length'] >= 100:
                calibration_segments.append({
                    'obs': ep['obs'][:100],
                    'action': ep['action'][:100],
                    'length': 100,
                })
        calibration_segments = calibration_segments[:5]

    if calibration_segments:
        calibration = evaluate_gp_uncertainty_calibration(
            gp, calibration_segments, state_std, n_segments=10
        )
        print(f"   Uncertainty-Error correlation: {calibration['correlation']:.4f}")
        print(f"   Calibration ratio (error/uncertainty): {calibration['calibration_ratio']:.4f}")
        print(f"   Mean uncertainty: {calibration['mean_uncertainty']:.6f}")
        print(f"   Mean error: {calibration['mean_error']:.6f}")

        if calibration['binned_calibration']:
            print(f"\n   {'Bin':<6} {'Unc Range':<20} {'Mean Unc':<12} {'Mean Err':<12} {'Ratio':<10}")
            print("   " + "-" * 60)
            for bc in calibration['binned_calibration']:
                print(f"   {bc['bin']:<6} "
                      f"[{bc['uncertainty_range'][0]:.4f},{bc['uncertainty_range'][1]:.4f}]"
                      f"  {bc['mean_uncertainty']:<12.6f} {bc['mean_error']:<12.6f} {bc['ratio']:<10.4f}")

        all_results['uncertainty_calibration'] = calibration
    else:
        all_results['uncertainty_calibration'] = {'note': 'No segments available'}

    # --------------------------------------------------------
    # 10. GP vs Neural ODE on OOD (optional)
    # --------------------------------------------------------
    print("\n10. GP vs Neural ODE comparison...")
    node = NeuralODEWrapper()
    node_loaded = node.try_load()

    comparison_results = {}
    if node_loaded and (ood_segments or id_segments):
        eval_horizons = [1, 10, 50, 100, 200]

        # Evaluate on ID segments
        if id_segments:
            print("   Evaluating Neural ODE on ID segments...")
            node_id = evaluate_model_on_segments(
                node, id_segments[:n_eval_segments], state_std, eval_horizons, "NODE-ID"
            )
            gp_id = evaluate_model_on_segments(
                gp, id_segments[:n_eval_segments], state_std, eval_horizons, "GP-ID"
            )
            comparison_results['id'] = {
                'gp': {str(h): gp_id[h] for h in eval_horizons},
                'node': {str(h): node_id[h] for h in eval_horizons},
            }

            print(f"\n   ID Comparison:")
            print(f"   {'Horizon':<10} {'GP NMAE':<14} {'NODE NMAE':<14} {'Winner':<10}")
            print("   " + "-" * 48)
            for h in eval_horizons:
                gp_val = gp_id[h]['nmae_mean']
                node_val = node_id[h]['nmae_mean']
                if not np.isnan(gp_val) and not np.isnan(node_val):
                    winner = "GP" if gp_val < node_val else "NODE"
                    print(f"   H={h:<7} {gp_val:<14.4f} {node_val:<14.4f} {winner:<10}")

        # Evaluate on OOD segments
        if ood_segments:
            print("\n   Evaluating Neural ODE on OOD segments...")
            node_ood = evaluate_model_on_segments(
                node, ood_segments[:n_eval_segments], state_std, eval_horizons, "NODE-OOD"
            )
            gp_ood = evaluate_model_on_segments(
                gp, ood_segments[:n_eval_segments], state_std, eval_horizons, "GP-OOD"
            )
            comparison_results['ood'] = {
                'gp': {str(h): gp_ood[h] for h in eval_horizons},
                'node': {str(h): node_ood[h] for h in eval_horizons},
            }

            print(f"\n   OOD Comparison:")
            print(f"   {'Horizon':<10} {'GP NMAE':<14} {'NODE NMAE':<14} {'Winner':<10}")
            print("   " + "-" * 48)
            for h in eval_horizons:
                gp_val = gp_ood[h]['nmae_mean']
                node_val = node_ood[h]['nmae_mean']
                if not np.isnan(gp_val) and not np.isnan(node_val):
                    winner = "GP" if gp_val < node_val else "NODE"
                    print(f"   H={h:<7} {gp_val:<14.4f} {node_val:<14.4f} {winner:<10}")

        # Synthetic perturbation comparison
        if base_segments:
            print("\n   Synthetic perturbation comparison (H=100)...")
            for ptype in ['speed', 'lateral', 'action', 'combined']:
                perturbed = scenario_gen.generate_synthetic_ood(
                    base_segments, ptype, 0.5
                )
                if perturbed:
                    gp_res = evaluate_model_on_segments(gp, perturbed, state_std, [100])
                    node_res = evaluate_model_on_segments(node, perturbed, state_std, [100])
                    gp_nmae = gp_res[100]['nmae_mean']
                    node_nmae = node_res[100]['nmae_mean']
                    winner = "GP" if gp_nmae < node_nmae else "NODE"
                    print(f"   {ptype:<12} GP={gp_nmae:.4f}  NODE={node_nmae:.4f}  Winner={winner}")
                    comparison_results[f'synth_{ptype}'] = {
                        'gp': gp_nmae, 'node': node_nmae, 'winner': winner,
                    }

    elif not node_loaded:
        comparison_results = {'note': 'Neural ODE not available (PyTorch load failed)'}
        print("   Neural ODE not available, skipping comparison")

    all_results['gp_vs_node'] = comparison_results

    # --------------------------------------------------------
    # 11. GP Generalization Summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("GP GENERALIZATION ANALYSIS SUMMARY")
    print("=" * 70)

    # ID vs OOD summary
    if 'gp_id_vs_ood' in all_results and all_results['gp_id_vs_ood'].get('ood'):
        id_h100 = all_results['gp_id_vs_ood']['id'].get('100', {})
        ood_h100 = all_results['gp_id_vs_ood']['ood'].get('100', {})
        deg = all_results['gp_id_vs_ood'].get('degradation', {}).get(100, float('nan'))

        print(f"\n  GP ID vs OOD (H=100):")
        print(f"    ID NMAE:  {id_h100.get('nmae_mean', 'N/A')}")
        print(f"    OOD NMAE: {ood_h100.get('nmae_mean', 'N/A')}")
        print(f"    Degradation: {deg:.2%}" if not np.isnan(deg) else "    Degradation: N/A")

    # OOD detection summary
    print(f"\n  OOD Detection (threshold={ood_detector.threshold}):")
    print(f"    Precision: {detection_metrics['precision']:.4f}")
    print(f"    Recall:    {detection_metrics['recall']:.4f}")
    print(f"    F1:        {detection_metrics['f1_score']:.4f}")

    # Uncertainty calibration summary
    if 'uncertainty_calibration' in all_results and 'correlation' in all_results['uncertainty_calibration']:
        cal = all_results['uncertainty_calibration']
        print(f"\n  Uncertainty Calibration:")
        print(f"    Correlation: {cal['correlation']:.4f}")
        print(f"    Calibration ratio: {cal['calibration_ratio']:.4f}")

    # Perturbation robustness
    if perturbation_results:
        print(f"\n  Perturbation Robustness (H=100):")
        for ptype in perturbation_types:
            if ptype in perturbation_results:
                baseline = perturbation_results[ptype].get('0.1', {}).get('nmae_mean', float('nan'))
                worst = perturbation_results[ptype].get('2.0', {}).get('nmae_mean', float('nan'))
                if not np.isnan(baseline) and not np.isnan(worst) and baseline > 0:
                    ratio = worst / baseline
                    print(f"    {ptype:<12}: baseline={baseline:.4f}, worst={worst:.4f}, ratio={ratio:.1f}x")

    # GP vs NODE summary
    if 'gp_vs_node' in all_results and 'note' not in all_results.get('gp_vs_node', {}):
        comp = all_results['gp_vs_node']
        if 'ood' in comp:
            print(f"\n  GP vs Neural ODE (OOD, H=100):")
            gp_ood = comp['ood']['gp'].get('100', {}).get('nmae_mean', 'N/A')
            node_ood = comp['ood']['node'].get('100', {}).get('nmae_mean', 'N/A')
            print(f"    GP OOD NMAE:  {gp_ood}")
            print(f"    NODE OOD NMAE: {node_ood}")

    # --------------------------------------------------------
    # 12. Save Results
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Saving results...")

    # Convert to JSON-serializable format
    def to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'gp_ood_test',
        'gp_config': {
            'n_samples': 1000,
            'kernel': 'RBF',
            'train_time': gp_train_time,
        },
        'results': to_serializable(all_results),
    }

    output_json = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP013_gp_ood_test.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {output_json}")

    # Generate markdown report
    report = generate_markdown_report(all_results, gp_train_time, detection_metrics)
    output_md = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/GP_OOD_ANALYSIS.md'
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Markdown: {output_md}")

    print("\nDone!")
    return all_results


# ============================================================
# Report Generator
# ============================================================
def generate_markdown_report(results, gp_train_time, detection_metrics):
    """Generate comprehensive markdown report."""

    lines = []
    lines.append("# GP OOD Analysis Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Experiment**: GP Out-of-Distribution Generalization Test")
    lines.append(f"**Dataset**: stage2_dataset_150k.npz (150k samples, 58 episodes)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # Compute key metrics for summary
    has_ood_comparison = (
        'gp_id_vs_ood' in results
        and results['gp_id_vs_ood'].get('ood')
        and len(results['gp_id_vs_ood']['ood']) > 0
    )

    if has_ood_comparison:
        id_h100 = results['gp_id_vs_ood']['id'].get('100', {})
        ood_h100 = results['gp_id_vs_ood']['ood'].get('100', {})
        deg = results['gp_id_vs_ood'].get('degradation', {}).get('100', None)

        id_val = id_h100.get('nmae_mean', None)
        ood_val = ood_h100.get('nmae_mean', None)

        lines.append("Key findings:")
        lines.append("")
        if id_val is not None and ood_val is not None:
            lines.append(f"- **GP ID NMAE (H=100)**: {id_val:.4f}")
            lines.append(f"- **GP OOD NMAE (H=100)**: {ood_val:.4f}")
            if deg is not None and not np.isnan(deg):
                lines.append(f"- **OOD Degradation**: {deg:.1%}")
        lines.append(f"- **OOD Detection F1**: {detection_metrics['f1_score']:.4f}")
        lines.append(f"- **GP Training Time**: {gp_train_time:.1f}s")
    else:
        lines.append("- Limited OOD segments found in test data. Results are primarily based on synthetic perturbations.")
        lines.append(f"- **OOD Detection F1**: {detection_metrics['f1_score']:.4f}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: OOD Detection
    lines.append("## 1. OOD Detection Performance")
    lines.append("")
    lines.append("Z-score based OOD detector evaluated on synthetic OOD samples "
                 "(states perturbed beyond 3-sigma from training distribution).")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| True Positives | {detection_metrics['true_positives']} |")
    lines.append(f"| False Positives | {detection_metrics['false_positives']} |")
    lines.append(f"| True Negatives | {detection_metrics['true_negatives']} |")
    lines.append(f"| False Negatives | {detection_metrics['false_negatives']} |")
    lines.append(f"| **Precision** | {detection_metrics['precision']:.4f} |")
    lines.append(f"| **Recall** | {detection_metrics['recall']:.4f} |")
    lines.append(f"| **F1 Score** | {detection_metrics['f1_score']:.4f} |")
    lines.append(f"| Accuracy | {detection_metrics['accuracy']:.4f} |")
    lines.append("")

    # Threshold sweep
    if 'threshold_sweep' in results:
        best = results['threshold_sweep']['best']
        lines.append("### Optimal Threshold")
        lines.append("")
        lines.append(f"- Best F1 threshold: **{best['threshold']:.2f}** "
                     f"(F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 2: ID vs OOD Performance
    lines.append("## 2. GP Performance: In-Distribution vs Out-of-Distribution")
    lines.append("")

    if has_ood_comparison:
        id_results = results['gp_id_vs_ood']['id']
        ood_results = results['gp_id_vs_ood']['ood']
        degradation = results['gp_id_vs_ood'].get('degradation', {})

        lines.append("| Horizon | GP-ID NMAE | GP-OOD NMAE | ID Survival | OOD Survival | Degradation |")
        lines.append("|---------|-----------|------------|------------|-------------|-------------|")

        for h_str in sorted(id_results.keys(), key=int):
            h = int(h_str)
            id_r = id_results.get(h_str, {})
            ood_r = ood_results.get(h_str, {})
            deg = degradation.get(h_str, None)

            id_nmae = id_r.get('nmae_mean', float('nan'))
            ood_nmae = ood_r.get('nmae_mean', float('nan'))
            id_surv = id_r.get('survival_rate', float('nan'))
            ood_surv = ood_r.get('survival_rate', float('nan'))

            id_str = f"{id_nmae:.4f}" if not np.isnan(id_nmae) else "N/A"
            ood_str = f"{ood_nmae:.4f}" if not np.isnan(ood_nmae) else "N/A"
            id_surv_str = f"{id_surv:.0%}" if not np.isnan(id_surv) else "N/A"
            ood_surv_str = f"{ood_surv:.0%}" if not np.isnan(ood_surv) else "N/A"
            deg_str = f"{deg:.1%}" if deg is not None and not np.isnan(deg) else "N/A"

            lines.append(f"| H={h} | {id_str} | {ood_str} | {id_surv_str} | {ood_surv_str} | {deg_str} |")
        lines.append("")
    else:
        lines.append("Insufficient natural OOD segments found in test data for direct comparison.")
        lines.append("Synthetic perturbation analysis below provides the primary OOD evaluation.")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 3: Synthetic Perturbation Analysis
    lines.append("## 3. Synthetic Perturbation Robustness")
    lines.append("")
    lines.append("GP evaluated on test segments with controlled perturbations at H=100.")
    lines.append("Perturbations simulate distribution shift in different state dimensions.")
    lines.append("")

    if 'perturbation_analysis' in results and results['perturbation_analysis']:
        perts = results['perturbation_analysis']
        mags = sorted(set(
            m for ptype in perts.values() for m in ptype.keys()
        ), key=float)

        # Table header
        header = "| Perturbation | " + " | ".join(f"mag={m}" for m in mags) + " |"
        separator = "|---|" + "|".join("---" for _ in mags) + "|"
        lines.append(header)
        lines.append(separator)

        for ptype in ['speed', 'lateral', 'heading', 'action', 'lean', 'combined']:
            if ptype not in perts:
                continue
            row = f"| **{ptype}** |"
            for m in mags:
                if m in perts[ptype]:
                    nmae = perts[ptype][m].get('nmae_mean', float('nan'))
                    surv = perts[ptype][m].get('survival_rate', float('nan'))
                    if not np.isnan(nmae):
                        row += f" {nmae:.4f} ({surv:.0%}) |"
                    else:
                        row += " N/A |"
                else:
                    row += " N/A |"
            lines.append(row)
        lines.append("")

        # Analysis
        lines.append("### Perturbation Impact Analysis")
        lines.append("")
        lines.append("- **Speed**: Tests robustness to velocity changes (scaled v)")
        lines.append("- **Lateral**: Tests robustness to lateral offset changes (e_y shift)")
        lines.append("- **Heading**: Tests robustness to heading changes (e_psi shift)")
        lines.append("- **Action**: Tests robustness to larger steering inputs")
        lines.append("- **Lean**: Tests robustness to lean angle changes (theta shift)")
        lines.append("- **Combined**: Multi-dimensional perturbation (e_y + e_psi + v + action)")
        lines.append("")

        # Find most/least robust perturbation type
        robustness = {}
        for ptype in perts:
            baseline_key = '0.1'
            worst_key = '2.0'
            if baseline_key in perts[ptype] and worst_key in perts[ptype]:
                base_nmae = perts[ptype][baseline_key].get('nmae_mean', float('nan'))
                worst_nmae = perts[ptype][worst_key].get('nmae_mean', float('nan'))
                if not np.isnan(base_nmae) and not np.isnan(worst_nmae) and base_nmae > 0:
                    robustness[ptype] = worst_nmae / base_nmae

        if robustness:
            most_robust = min(robustness, key=robustness.get)
            least_robust = max(robustness, key=robustness.get)
            lines.append(f"- **Most robust** to: {most_robust} (ratio: {robustness[most_robust]:.1f}x)")
            lines.append(f"- **Least robust** to: {least_robust} (ratio: {robustness[least_robust]:.1f}x)")
            lines.append("")

    lines.append("---")
    lines.append("")

    # Section 4: Uncertainty Calibration
    lines.append("## 4. GP Uncertainty Calibration")
    lines.append("")
    lines.append("Evaluates how well GP's predicted uncertainty correlates with actual prediction error.")
    lines.append("Well-calibrated uncertainty is critical for safe OOD detection.")
    lines.append("")

    if 'uncertainty_calibration' in results and 'correlation' in results['uncertainty_calibration']:
        cal = results['uncertainty_calibration']
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Uncertainty-Error Correlation | {cal['correlation']:.4f} |")
        lines.append(f"| Calibration Ratio (error/unc) | {cal['calibration_ratio']:.4f} |")
        lines.append(f"| Mean Uncertainty | {cal['mean_uncertainty']:.6f} |")
        lines.append(f"| Mean Error | {cal['mean_error']:.6f} |")
        lines.append(f"| Sample Pairs | {cal['n_pairs']} |")
        lines.append("")

        if cal.get('binned_calibration'):
            lines.append("### Binned Calibration")
            lines.append("")
            lines.append("| Bin | Uncertainty Range | Mean Uncertainty | Mean Error | Ratio | Count |")
            lines.append("|-----|-------------------|-----------------|------------|-------|-------|")
            for bc in cal['binned_calibration']:
                lines.append(
                    f"| {bc['bin']} | "
                    f"[{bc['uncertainty_range'][0]:.4f}, {bc['uncertainty_range'][1]:.4f}] | "
                    f"{bc['mean_uncertainty']:.6f} | {bc['mean_error']:.6f} | "
                    f"{bc['ratio']:.4f} | {bc['count']} |"
                )
            lines.append("")

        # Interpretation
        corr = cal['correlation']
        if corr > 0.7:
            interp = "Strong positive correlation -- GP uncertainty is a reliable indicator of prediction error."
        elif corr > 0.4:
            interp = "Moderate correlation -- GP uncertainty provides useful but imperfect error signal."
        elif corr > 0.0:
            interp = "Weak correlation -- GP uncertainty is a poor predictor of actual error."
        else:
            interp = "No/negative correlation -- GP uncertainty is unreliable for OOD detection."

        lines.append(f"**Interpretation**: {interp}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 5: GP vs Neural ODE
    lines.append("## 5. GP vs Neural ODE on OOD Data")
    lines.append("")

    if 'gp_vs_node' in results and 'note' not in results.get('gp_vs_node', {}):
        comp = results['gp_vs_node']

        if 'id' in comp:
            lines.append("### In-Distribution Comparison")
            lines.append("")
            lines.append("| Horizon | GP NMAE | Neural ODE NMAE | Winner |")
            lines.append("|---------|---------|----------------|--------|")
            gp_id = comp['id']['gp']
            node_id = comp['id']['node']
            for h_str in sorted(gp_id.keys(), key=int):
                gp_val = gp_id[h_str].get('nmae_mean', float('nan'))
                node_val = node_id[h_str].get('nmae_mean', float('nan'))
                if not np.isnan(gp_val) and not np.isnan(node_val):
                    winner = "GP" if gp_val < node_val else "Neural ODE"
                    lines.append(f"| H={h_str} | {gp_val:.4f} | {node_val:.4f} | {winner} |")
            lines.append("")

        if 'ood' in comp:
            lines.append("### Out-of-Distribution Comparison")
            lines.append("")
            lines.append("| Horizon | GP NMAE | Neural ODE NMAE | Winner |")
            lines.append("|---------|---------|----------------|--------|")
            gp_ood = comp['ood']['gp']
            node_ood = comp['ood']['node']
            for h_str in sorted(gp_ood.keys(), key=int):
                gp_val = gp_ood[h_str].get('nmae_mean', float('nan'))
                node_val = node_ood[h_str].get('nmae_mean', float('nan'))
                if not np.isnan(gp_val) and not np.isnan(node_val):
                    winner = "GP" if gp_val < node_val else "Neural ODE"
                    lines.append(f"| H={h_str} | {gp_val:.4f} | {node_val:.4f} | {winner} |")
            lines.append("")

        # Synthetic perturbation comparison
        synth_keys = [k for k in comp if k.startswith('synth_')]
        if synth_keys:
            lines.append("### Synthetic Perturbation Comparison (H=100)")
            lines.append("")
            lines.append("| Perturbation | GP NMAE | Neural ODE NMAE | Winner |")
            lines.append("|---|---|---|---|")
            for k in sorted(synth_keys):
                ptype = k.replace('synth_', '')
                gp_val = comp[k]['gp']
                node_val = comp[k]['node']
                winner = comp[k]['winner']
                lines.append(f"| {ptype} | {gp_val:.4f} | {node_val:.4f} | {winner} |")
            lines.append("")
    else:
        lines.append("Neural ODE model not available for comparison (PyTorch load failed).")
        lines.append("To enable this comparison, ensure PyTorch is installed and the v9 model exists at:")
        lines.append(f"`{NODE_MODEL_PATH}`")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 6: Natural OOD Segments
    lines.append("## 6. Natural OOD Segments in Test Data")
    lines.append("")

    if 'natural_segments' in results:
        ns = results['natural_segments']
        lines.append(f"- Natural OOD segments found: **{ns['n_ood']}**")
        lines.append(f"- Natural ID segments found: **{ns['n_id']}**")
        if ns.get('top_ood_scores'):
            scores = ns['top_ood_scores'][:5]
            lines.append(f"- Top OOD scores: {', '.join(f'{s:.2f}' for s in scores)}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Section 7: Generalization Assessment
    lines.append("## 7. GP Generalization Assessment")
    lines.append("")
    lines.append("### Strengths")
    lines.append("")

    strengths = []
    weaknesses = []

    # Analyze strengths
    if detection_metrics['f1_score'] > 0.8:
        strengths.append("High OOD detection accuracy -- z-score detector works well for this state space")
    if detection_metrics['recall'] > 0.9:
        strengths.append("High recall -- rarely misses OOD samples")

    # Check uncertainty calibration
    if 'uncertainty_calibration' in results and 'correlation' in results['uncertainty_calibration']:
        corr = results['uncertainty_calibration']['correlation']
        if corr > 0.5:
            strengths.append("GP uncertainty correlates with actual error -- useful for confidence estimation")

    # Check robustness
    if 'perturbation_analysis' in results:
        perts = results['perturbation_analysis']
        for ptype in perts:
            if '0.1' in perts[ptype] and '2.0' in perts[ptype]:
                base = perts[ptype]['0.1'].get('nmae_mean', float('nan'))
                worst = perts[ptype]['2.0'].get('nmae_mean', float('nan'))
                if not np.isnan(base) and not np.isnan(worst) and base > 0:
                    ratio = worst / base
                    if ratio < 3.0:
                        strengths.append(f"Robust to {ptype} perturbation (< 3x degradation at 2x magnitude)")

    if not strengths:
        strengths.append("GP provides uncertainty estimates that can be used for OOD gating")

    for s in strengths:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("### Weaknesses")
    lines.append("")

    # Analyze weaknesses
    if detection_metrics['precision'] < 0.7:
        weaknesses.append("Low precision -- many false positives in OOD detection")
    if detection_metrics['f1_score'] < 0.5:
        weaknesses.append("Low F1 score -- OOD detection needs improvement")

    if 'uncertainty_calibration' in results and 'correlation' in results['uncertainty_calibration']:
        corr = results['uncertainty_calibration']['correlation']
        if corr < 0.3:
            weaknesses.append("Weak uncertainty-error correlation -- GP variance is not a reliable error proxy")

    if 'perturbation_analysis' in results:
        perts = results['perturbation_analysis']
        for ptype in perts:
            if '0.1' in perts[ptype] and '2.0' in perts[ptype]:
                base = perts[ptype]['0.1'].get('nmae_mean', float('nan'))
                worst = perts[ptype]['2.0'].get('nmae_mean', float('nan'))
                if not np.isnan(base) and not np.isnan(worst) and base > 0:
                    ratio = worst / base
                    if ratio > 10.0:
                        weaknesses.append(f"Catastrophic degradation on {ptype} perturbation ({ratio:.0f}x at 2x magnitude)")

    if not weaknesses:
        weaknesses.append("No major weaknesses identified in this evaluation")

    for w in weaknesses:
        lines.append(f"- {w}")
    lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")
    lines.append("1. **Use GP uncertainty for OOD gating**: When GP variance exceeds a calibrated "
                 "threshold, fall back to a physics baseline or reduce residual contribution.")
    lines.append("2. **Combine with physics constraints**: GP predictions outside the training "
                 "distribution should be blended with known dynamics (e.g., Meijaard model).")
    lines.append("3. **Monitor state-space coverage**: Track which regions of the state space "
                 "are well-covered by training data and flag predictions in sparse regions.")
    lines.append("4. **Adaptive threshold**: The optimal OOD detection threshold should be "
                 "calibrated on a validation set and updated as more data is collected.")
    lines.append("5. **Ensemble for robustness**: Use multiple GP models trained on different "
                 "subsets to detect when predictions disagree (high variance = likely OOD).")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Experimental Setup")
    lines.append("")
    lines.append(f"- **GP kernel**: RBF (length_scale=1.0) with ConstantKernel")
    lines.append(f"- **Training samples**: 1000 (random subset from 101k training samples)")
    lines.append(f"- **GP dimensions**: 7D state (e_y, e_psi, v, theta, theta_dot, delta, delta_dot)")
    lines.append(f"- **Action dimension**: 1D steering torque")
    lines.append(f"- **GP training time**: {gp_train_time:.1f}s")
    lines.append(f"- **Dataset**: stage2_dataset_150k.npz (150k samples, 58 episodes)")
    lines.append(f"- **Train/test split**: 75%/25% by episodes")
    lines.append(f"- **OOD detector**: Z-score based, threshold={OODDetector(train_obs=np.zeros((1,7)), threshold=3.0).threshold}")
    lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    main()

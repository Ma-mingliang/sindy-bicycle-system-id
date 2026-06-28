"""V6 Agent F: 7D Route Baseline -- GP on real 7D bicycle data.

Loads real 8D route data from stage2_dataset_150k.npz, drops kappa (dim 5)
to form the 7D state [e_y, e_psi, v, theta, theta_dot, delta, delta_dot],
and trains a Gaussian Process baseline. Evaluates single-step and multi-step
prediction at horizons [1, 5, 10, 20, 50, 100].

Output: continuation_stage_v6/subagents/AGENT_F_RESULTS.json
"""
import sys
import os
import json
import time
import warnings
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ---- Path setup ----
ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ---- Suppress warnings ----
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*fvalue.*")
warnings.filterwarnings("ignore", message=".*Optimization.*")
warnings.filterwarnings("ignore", message=".*GaussianProcessRegressor.*")
try:
    import sklearn.exceptions
    warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
except ImportError:
    pass

# =====================================================================
# Constants
# =====================================================================
DATA_PATH = Path(ROOT) / "data" / "stage2_dataset_150k.npz"
OUTPUT_DIR = Path(__file__).parent
RESULTS_FILE = OUTPUT_DIR / "AGENT_F_RESULTS.json"

# 7D state names (after dropping kappa)
STATE_NAMES_7D = ["e_y", "e_psi", "v", "theta", "theta_dot", "delta", "delta_dot"]

# Original 8D indices to keep (drop index 5 = kappa)
KEEP_INDICES = [0, 1, 2, 3, 4, 6, 7]

# Evaluation horizons
EVAL_HORIZONS = [1, 5, 10, 20, 50, 100]

# GP config
GP_MAX_SAMPLES = 1000
GP_N_RESTARTS = 1


# =====================================================================
# Data loading and preparation
# =====================================================================
def load_real_data():
    """Load 8D route data and extract 7D subset."""
    print(f"Loading data from {DATA_PATH}")
    raw = np.load(str(DATA_PATH), allow_pickle=True)
    obs_8d = raw["obs"]        # (N, 8) normalized
    next_obs_8d = raw["next_obs"]
    actions = raw["action"]     # (N, 1) normalized
    done = raw["done"]          # (N,) bool
    n_episodes = int(raw["n_episodes"])
    n_steps = int(raw["n_steps"])

    print(f"  Raw 8D data: {obs_8d.shape[0]} samples, {n_episodes} episodes")

    # Extract 7D (drop kappa = dim 5)
    obs_7d = obs_8d[:, KEEP_INDICES]
    next_obs_7d = next_obs_8d[:, KEEP_INDICES]

    return obs_7d, next_obs_7d, actions, done, n_episodes


def compute_data_statistics(obs_7d, actions, done):
    """Compute per-state statistics."""
    stats = {}
    for i, name in enumerate(STATE_NAMES_7D):
        col = obs_7d[:, i]
        stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }
    stats["action"] = {
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "std": float(actions.std()),
    }

    # Episode info
    done_idx = np.where(done)[0]
    ep_lengths = np.diff(np.concatenate([[-1], done_idx]))
    stats["episodes"] = {
        "n_episodes": len(done_idx),
        "total_samples": len(obs_7d),
        "min_ep_length": int(ep_lengths.min()),
        "max_ep_length": int(ep_lengths.max()),
        "mean_ep_length": float(ep_lengths.mean()),
    }

    return stats


def split_train_test(obs_7d, next_obs_7d, actions, done, train_ratio=0.8):
    """Split by episode (no leakage across episode boundaries)."""
    done_idx = np.where(done)[0]
    n_episodes = len(done_idx)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_episodes)
    n_train = int(n_episodes * train_ratio)

    train_episodes = perm[:n_train]
    test_episodes = perm[n_train:]

    def gather(ep_indices):
        X_list, Y_list, A_list = [], [], []
        for ep_i in ep_indices:
            start = done_idx[ep_i - 1] + 1 if ep_i > 0 else 0
            end = done_idx[ep_i] + 1
            X_list.append(obs_7d[start:end])
            Y_list.append(next_obs_7d[start:end])
            A_list.append(actions[start:end])
        return np.vstack(X_list), np.vstack(Y_list), np.vstack(A_list)

    X_train, Y_train, A_train = gather(train_episodes)
    X_test, Y_test, A_test = gather(test_episodes)

    return X_train, Y_train, A_train, X_test, Y_test, A_test


# =====================================================================
# GP model (7D)
# =====================================================================
class GP7D:
    """One GP per state dimension, predicting next_state from (state, action)."""

    def __init__(self, max_samples=GP_MAX_SAMPLES, n_restarts=GP_N_RESTARTS):
        self.max_samples = max_samples
        self.n_restarts = n_restarts
        self._gps = None
        self._input_std = None
        self._delta_std = None

    def train(self, states, next_states, actions):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        # Compute deltas
        deltas = next_states - states

        # Normalize inputs
        X = np.column_stack([states, actions.reshape(-1, 1)])
        self._input_std = np.std(X, axis=0)
        self._input_std[self._input_std < 1e-10] = 1.0  # avoid div by zero

        self._delta_std = np.std(deltas, axis=0)
        self._delta_std[self._delta_std < 1e-10] = 1.0

        X_norm = X / self._input_std
        Y_norm = deltas / self._delta_std

        # Subsample if needed
        n = len(X_norm)
        actual_max = self.max_samples
        # Kernel matrix is n x n float64; check available memory
        mem_needed_mb = actual_max * actual_max * 8 / 1e6
        if mem_needed_mb > 100:
            actual_max = min(actual_max, int(np.sqrt(100e6 / 8)))
            print(f"    (capped to {actual_max} samples due to memory: "
                  f"kernel would need {mem_needed_mb:.0f} MB)")
        if n > actual_max:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, actual_max, replace=False)
        else:
            idx = np.arange(n)

        X_train = X_norm[idx]
        Y_train = Y_norm[idx]

        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self._gps = []
        t0 = time.time()
        for col in range(7):
            t_col = time.time()
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=self.n_restarts,
                    alpha=1e-6,
                )
                gp.fit(X_train, Y_train[:, col])
                self._gps.append(gp)
                print(f"    GP dim {col} ({STATE_NAMES_7D[col]}): trained "
                      f"({len(X_train)} samples, {time.time()-t_col:.1f}s)")
            except MemoryError:
                print(f"    GP dim {col} ({STATE_NAMES_7D[col]}): MEMORY ERROR, "
                      f"retrying with fewer samples")
                sub_n = min(len(X_train), 500)
                rng2 = np.random.RandomState(42)
                sub_idx = rng2.choice(len(X_train), sub_n, replace=False)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=max(1, self.n_restarts - 1),
                    alpha=1e-5,
                )
                gp.fit(X_train[sub_idx], Y_train[sub_idx, col])
                self._gps.append(gp)
                print(f"    GP dim {col} ({STATE_NAMES_7D[col]}): trained "
                      f"(fallback {sub_n} samples, {time.time()-t_col:.1f}s)")
            except Exception as e:
                print(f"    GP dim {col} ({STATE_NAMES_7D[col]}): ERROR: {e}")
                self._gps.append(None)

        elapsed = time.time() - t0
        print(f"  GP training done in {elapsed:.1f}s")
        return elapsed

    def predict_next_state(self, state, action):
        """Single-step prediction: returns next_state."""
        x = np.concatenate([state, [action]]).reshape(1, -1)
        x_norm = x / self._input_std
        delta_norm = np.array([gp.predict(x_norm)[0] if gp is not None else 0.0
                               for gp in self._gps])
        delta = delta_norm * self._delta_std
        return state + delta.flatten()

    def predict_trajectory(self, s0, actions_seq):
        """Multi-step rollout starting from s0."""
        trajectory = [s0.copy()]
        s = s0.copy()
        for a in actions_seq:
            s = self.predict_next_state(s, a)
            trajectory.append(s.copy())
        return np.array(trajectory)


# =====================================================================
# Evaluation
# =====================================================================
def evaluate_single_step(model, states, next_states, actions):
    """Compute single-step prediction error."""
    preds = np.array([
        model.predict_next_state(states[i], actions[i, 0])
        for i in range(len(states))
    ])
    errors = np.abs(preds - next_states)
    per_state_mae = errors.mean(axis=0)
    overall_mae = errors.mean()
    return {
        "overall_mae": float(overall_mae),
        "per_state_mae": {name: float(mae) for name, mae in zip(STATE_NAMES_7D, per_state_mae)},
        "per_state_rmse": {name: float(np.sqrt(((preds[:, i] - next_states[:, i])**2).mean()))
                           for i, name in enumerate(STATE_NAMES_7D)},
    }


def evaluate_multistep(model, test_episodes_data, horizons):
    """Multi-step rollout evaluation on test episodes."""
    results = {}
    for h in horizons:
        all_errors = []
        n_valid = 0
        for ep_states, ep_actions in test_episodes_data:
            if len(ep_states) < h + 1:
                continue
            s0 = ep_states[0]
            actions_seq = ep_actions[:h, 0]
            true_traj = ep_states[:h + 1]

            # Rollout
            pred_traj = [s0.copy()]
            s = s0.copy()
            diverged = False
            for t in range(h):
                s = model.predict_next_state(s, ep_actions[t, 0])
                if np.any(np.isnan(s)) or np.any(np.abs(s) > 10):
                    diverged = True
                    break
                pred_traj.append(s.copy())

            if diverged or len(pred_traj) < h + 1:
                continue

            pred_traj = np.array(pred_traj)
            errors = np.abs(pred_traj - true_traj)
            all_errors.append(errors)
            n_valid += 1

        if n_valid > 0:
            all_errors = np.array(all_errors)
            mean_errors = all_errors.mean(axis=0)
            results[str(h)] = {
                "n_valid_episodes": n_valid,
                "per_step_mae": {name: float(mean_errors[:, i].mean())
                                 for i, name in enumerate(STATE_NAMES_7D)},
                "per_step_rmse": {name: float(np.sqrt((mean_errors[:, i]**2).mean()))
                                  for i, name in enumerate(STATE_NAMES_7D)},
                "endpoint_mae": {name: float(mean_errors[-1, i])
                                 for i, name in enumerate(STATE_NAMES_7D)},
                "overall_endpoint_mae": float(mean_errors[-1].mean()),
            }
        else:
            results[str(h)] = {"n_valid_episodes": 0, "error": "no valid episodes"}

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("V6 AGENT F: 7D Route Baseline -- GP on Real Data")
    print("=" * 70)
    t0 = time.time()

    # 1. Load data
    print("\n--- Loading real 8D data ---")
    obs_7d, next_obs_7d, actions, done, n_episodes = load_real_data()

    # 2. Data statistics
    print("\n--- Data statistics (7D) ---")
    stats = compute_data_statistics(obs_7d, actions, done)
    for name in STATE_NAMES_7D:
        s = stats[name]
        print(f"  {name:12s}: [{s['min']:+.4f}, {s['max']:+.4f}], "
              f"mean={s['mean']:+.4f}, std={s['std']:.4f}")
    print(f"  action:     [{stats['action']['min']:+.4f}, {stats['action']['max']:+.4f}], "
          f"mean={stats['action']['mean']:+.4f}, std={stats['action']['std']:.4f}")
    ep = stats["episodes"]
    print(f"  episodes: {ep['n_episodes']}, total_samples: {ep['total_samples']}, "
          f"ep_length: [{ep['min_ep_length']}, {ep['max_ep_length']}]")

    # 3. Split train/test by episode
    print("\n--- Splitting train/test (80/20 by episode) ---")
    X_train, Y_train, A_train, X_test, Y_test, A_test = split_train_test(
        obs_7d, next_obs_7d, actions, done
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # 4. Train GP
    print("\n--- Training GP baseline ---")
    gp = GP7D(max_samples=GP_MAX_SAMPLES, n_restarts=GP_N_RESTARTS)
    train_time = gp.train(X_train, Y_train, A_train)

    # 5. Single-step evaluation
    print("\n--- Single-step evaluation ---")
    single_step = evaluate_single_step(gp, X_test, Y_test, A_test)
    print(f"  Overall MAE: {single_step['overall_mae']:.6f}")
    for name in STATE_NAMES_7D:
        print(f"  {name:12s}: MAE={single_step['per_state_mae'][name]:.6f}, "
              f"RMSE={single_step['per_state_rmse'][name]:.6f}")

    # 6. Multi-step evaluation
    print("\n--- Multi-step evaluation ---")
    # Reconstruct test episodes
    done_idx = np.where(done)[0]
    test_ep_indices = []
    perm = np.random.RandomState(42).permutation(len(done_idx))
    n_train = int(len(done_idx) * 0.8)
    test_eps = perm[n_train:]

    test_episodes_data = []
    for ep_i in test_eps:
        start = done_idx[ep_i - 1] + 1 if ep_i > 0 else 0
        end = done_idx[ep_i] + 1
        ep_obs = obs_7d[start:end]
        ep_act = actions[start:end]
        test_episodes_data.append((ep_obs, ep_act))

    multistep = evaluate_multistep(gp, test_episodes_data, EVAL_HORIZONS)
    for h_str, m in multistep.items():
        if "error" in m:
            print(f"  Horizon {h_str:>4s}: {m['error']}")
        else:
            print(f"  Horizon {h_str:>4s}: endpoint_mae={m['overall_endpoint_mae']:.6f} "
                  f"({m['n_valid_episodes']} episodes)")

    # 7. Save results
    elapsed = time.time() - t0
    output = {
        "agent": "F",
        "task": "7D Route Baseline",
        "data_source": "real (stage2_dataset_150k.npz)",
        "state_vector": STATE_NAMES_7D,
        "state_dim": 7,
        "action_dim": 1,
        "kappa_dropped": True,
        "data_statistics": stats,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "gp_config": {
            "max_samples": GP_MAX_SAMPLES,
            "n_restarts": GP_N_RESTARTS,
            "train_time_s": train_time,
        },
        "single_step": single_step,
        "multi_step": multistep,
        "eval_horizons": EVAL_HORIZONS,
        "elapsed_s": elapsed,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
    return output


if __name__ == "__main__":
    main()

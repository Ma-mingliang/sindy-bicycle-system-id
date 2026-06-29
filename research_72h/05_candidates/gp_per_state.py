"""GP Per-State Model - Train separate GP for each state variable.

Core idea: Each state has its own GP with kernel chosen based on state dynamics:
- e_y: GP with Matern kernel (smooth lateral error)
- e_psi: GP with Matern kernel (heading error)
- v: GP with RBF kernel (smooth velocity)
- theta: GP with Matern kernel (heading angle)
- theta_dot: GP with Matern kernel (angular velocity)
- delta: GP with Matern kernel (steering angle)
- delta_dot: GP with Matern kernel (steering rate)

Evaluation: H=1,10,50,100,200,500,1000
Comparison: v9 Neural ODE baseline
"""
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

warnings.filterwarnings('ignore')

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# State definitions
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Kernel configuration per state
KERNEL_CONFIG = {
    'e_y': {'type': 'matern', 'nu': 2.5, 'n_samples': 500},
    'e_psi': {'type': 'matern', 'nu': 2.5, 'n_samples': 500},
    'v': {'type': 'rbf', 'n_samples': 500},
    'theta': {'type': 'matern', 'nu': 2.5, 'n_samples': 500},
    'theta_dot': {'type': 'matern', 'nu': 1.5, 'n_samples': 500},
    'delta': {'type': 'matern', 'nu': 2.5, 'n_samples': 500},
    'delta_dot': {'type': 'matern', 'nu': 1.5, 'n_samples': 500},
}

# v9 baseline results (seed=42)
V9_BASELINE = {
    1: 0.0255, 10: 0.0648, 50: 0.1854, 100: 0.4407, 200: 0.9967, 500: 1.3837,
    'primary': 0.9404,  # mean of H=100,200,500
}


def check_survival(state):
    """Check if state survives physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', seed=42):
    """Load 7D data from 8D dataset with episode structure preserved."""
    data = np.load(data_path, allow_pickle=True)
    obs_8d = data['obs']
    action = data['action']
    next_obs_8d = data['next_obs']
    done = data['done']

    # Convert to 7D
    obs = obs_8d[:, IDX_7D_FROM_8D]
    next_obs = next_obs_8d[:, IDX_7D_FROM_8D]
    deltas = next_obs - obs

    # Split by episodes
    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    # Train/test split (75/25 by episode)
    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    # Build episode list
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

    # Compute statistics from training episodes
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


def build_kernel(kernel_type, nu=2.5):
    """Build kernel for a specific state."""
    if kernel_type == 'matern':
        return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=nu)
    elif kernel_type == 'rbf':
        return ConstantKernel(1.0) * RBF(length_scale=1.0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def train_gp_per_state(data, n_restarts=1, seed=42):
    """Train separate GP for each state variable."""
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    # Prepare normalized input
    X_full = np.hstack([
        train_obs / state_std,
        train_action.reshape(-1, 1) / action_std
    ])

    gps = {}
    learned_kernels = {}
    train_times = {}

    for dim_idx, state_name in enumerate(STATE_NAMES_7D):
        config = KERNEL_CONFIG[state_name]
        n_samples = min(config['n_samples'], len(X_full))

        # Subsample for speed
        indices = np.random.choice(len(X_full), n_samples, replace=False)
        X = X_full[indices]
        y = train_deltas[indices, dim_idx] / delta_std[dim_idx]

        kernel = build_kernel(config['type'], config.get('nu', 2.5))

        print(f"  Training GP for {state_name} ({config['type']}, nu={config.get('nu', 'N/A')}, "
              f"n={n_samples})...", end='', flush=True)

        t0 = time.time()
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
            alpha=1e-6,
            normalize_y=True,
        )
        gp.fit(X, y)
        elapsed = time.time() - t0

        gps[dim_idx] = gp
        learned_kernels[state_name] = str(gp.kernel_)
        train_times[state_name] = elapsed

        print(f" {elapsed:.1f}s", flush=True)
        print(f"    Learned kernel: {gp.kernel_}", flush=True)

    return gps, learned_kernels, train_times


def evaluate_gp_model(gps, data, horizons, n_segments=5, seed=42):
    """Evaluate GP per-state model on test segments."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    np.random.seed(seed)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 1100:
            segments.append(ep)
    segments = segments[:n_segments]

    if not segments:
        print("Warning: No test segments long enough!")
        return {}

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}
        per_state_std = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['obs'][0].copy()
            actions_seg = seg['action'].flatten()
            real_states = seg['obs']
            n = min(h, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    # Prepare normalized input
                    x = np.hstack([
                        s_cur / state_std,
                        [actions_seg[step] / action_std]
                    ]).reshape(1, -1)

                    # Predict delta for each state
                    delta_pred = np.zeros(STATE_DIM)
                    for dim_idx in range(STATE_DIM):
                        delta_pred[dim_idx] = gps[dim_idx].predict(x)[0]

                    # Convert from normalized delta to physical delta
                    dsdt = delta_pred * delta_std * dt
                    s_next = s_cur + dsdt

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next):
                        survived = False
                        break

                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)

                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            if step_errors:
                n_valid = min(len(step_errors), n)
                errors = np.array(step_errors[:n_valid])
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))

            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
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


def analyze_prediction_quality(gps, data, seed=42):
    """Analyze per-state prediction quality on training data."""
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    # Subsample for analysis
    n = min(2000, len(train_obs))
    indices = np.random.choice(len(train_obs), n, replace=False)

    X = np.hstack([
        train_obs[indices] / state_std,
        train_action[indices].reshape(-1, 1) / action_std
    ])

    analysis = {}
    for dim_idx, state_name in enumerate(STATE_NAMES_7D):
        y_true = train_deltas[indices, dim_idx] / delta_std[dim_idx]
        y_pred = gps[dim_idx].predict(X)

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

        analysis[state_name] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'y_true_std': float(np.std(y_true)),
            'y_pred_std': float(np.std(y_pred)),
        }

    return analysis


def main():
    print("=" * 60, flush=True)
    print("GP Per-State Model", flush=True)
    print("Train separate GP for each state variable", flush=True)
    print("=" * 60, flush=True)

    # Load data
    print("\n1. Loading data...", flush=True)
    data = load_data(seed=42)
    print(f"   State std: {data['state_std']}", flush=True)
    print(f"   Action std: {data['action_std']}", flush=True)
    print(f"   Delta std: {data['delta_std']}", flush=True)
    print(f"   Train episodes: {len(data['train_eps'])}", flush=True)
    print(f"   Test episodes: {len(data['test_eps'])}", flush=True)
    print(f"   Total train samples: {len(data['train_obs'])}", flush=True)

    # Train GP per state
    print("\n2. Training GP per state...", flush=True)
    start_time = time.time()
    gps, learned_kernels, train_times = train_gp_per_state(data, n_restarts=1, seed=42)
    total_train_time = time.time() - start_time
    print(f"\n   Total training time: {total_train_time:.1f}s", flush=True)

    # Analyze prediction quality
    print("\n3. Analyzing prediction quality...", flush=True)
    analysis = analyze_prediction_quality(gps, data, seed=42)
    print(f"\n   {'State':<15} {'R2':<10} {'MSE':<12} {'MAE':<12} {'True Std':<12} {'Pred Std':<12}")
    print("   " + "-" * 73)
    for state_name in STATE_NAMES_7D:
        a = analysis[state_name]
        print(f"   {state_name:<15} {a['r2']:<10.4f} {a['mse']:<12.6f} {a['mae']:<12.6f} "
              f"{a['y_true_std']:<12.4f} {a['y_pred_std']:<12.4f}", flush=True)

    # Evaluate on test segments
    print("\n4. Evaluating on test segments...", flush=True)
    horizons = [1, 10, 50, 100, 200, 500, 1000]
    results = evaluate_gp_model(gps, data, horizons, n_segments=3, seed=42)

    # Print results
    print(f"\n   {'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("   " + "-" * 34)
    for h in horizons:
        r = results[h]
        print(f"   H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}", flush=True)

    # Per-state analysis at each horizon
    print("\n5. Per-state NMAE at each horizon...", flush=True)
    for h in horizons:
        r = results[h]
        print(f"\n   H={h}:")
        print(f"   {'State':<15} {'NMAE':<12} {'Std':<12}")
        print("   " + "-" * 39)
        for state_name in STATE_NAMES_7D:
            ps = r['per_state_nmae'][state_name]
            print(f"   {state_name:<15} {ps['mean']:<12.4f} {ps['std']:<12.4f}", flush=True)

    # Compute primary score
    primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
    print(f"\n6. PrimaryLongHorizonScore (mean of H=100,200,500): {primary:.4f}", flush=True)

    # Compare with v9 baseline
    print("\n7. Comparison with v9 baseline:", flush=True)
    print(f"   {'Horizon':<10} {'GP PerState':<15} {'v9 Baseline':<15} {'Improvement':<15}")
    print("   " + "-" * 55)
    for h in [1, 10, 50, 100, 200, 500]:
        gp_nmae = results[h]['nmae_mean']
        v9_nmae = V9_BASELINE[h]
        improvement = (v9_nmae - gp_nmae) / v9_nmae * 100
        marker = "***" if improvement > 0 else ""
        print(f"   H={h:<7} {gp_nmae:<15.4f} {v9_nmae:<15.4f} {improvement:>+10.1f}% {marker}", flush=True)

    v9_primary = V9_BASELINE['primary']
    improvement_primary = (v9_primary - primary) / v9_primary * 100
    print(f"\n   Primary Score: GP={primary:.4f}, v9={v9_primary:.4f}, Improvement={improvement_primary:+.1f}%", flush=True)

    # Learned kernels summary
    print("\n8. Learned Kernels:", flush=True)
    for state_name in STATE_NAMES_7D:
        print(f"   {state_name:<15}: {learned_kernels[state_name]}", flush=True)

    # Training time breakdown
    print("\n9. Training Time Breakdown:", flush=True)
    for state_name in STATE_NAMES_7D:
        print(f"   {state_name:<15}: {train_times[state_name]:.1f}s", flush=True)
    print(f"   {'Total':<15}: {total_train_time:.1f}s", flush=True)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_gp_per_state',
        'experiment': 'gp_per_state',
        'description': 'Train separate GP for each state variable with optimized kernels',
        'kernel_config': KERNEL_CONFIG,
        'learned_kernels': learned_kernels,
        'train_times': {k: float(v) for k, v in train_times.items()},
        'total_train_time': total_train_time,
        'prediction_quality': analysis,
        'horizons': horizons,
        'results': {},
        'v9_comparison': {},
        'primary_score': primary,
        'v9_primary_score': v9_primary,
        'improvement_primary': improvement_primary,
    }

    for h in horizons:
        r = results[h]
        output['results'][str(h)] = {
            'nmae_mean': r['nmae_mean'],
            'nmae_std': r['nmae_std'],
            'survival_rate': r['survival_rate'],
            'n_valid': r['n_valid'],
            'per_state_nmae': r['per_state_nmae'],
        }

        if h in [1, 10, 50, 100, 200, 500]:
            v9_nmae = V9_BASELINE[h]
            gp_nmae = r['nmae_mean']
            improvement = (v9_nmae - gp_nmae) / v9_nmae * 100
            output['v9_comparison'][str(h)] = {
                'gp_nmae': gp_nmae,
                'v9_nmae': v9_nmae,
                'improvement_pct': improvement,
            }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP047_gp_per_state.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == '__main__':
    main()

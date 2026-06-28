"""Optimize GP Hyperparameters.

Tests different GP configurations to find the best long-horizon prediction.
"""
import sys
import json
import time
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}


def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(seed=42):
    data = np.load('D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', allow_pickle=True)
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


def train_gp_model(data, kernel_config, n_samples=5000, seed=42):
    """Train GP model with specified kernel configuration."""
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Sample training data
    n_total = len(data['train_obs'])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    X = np.hstack([
        data['train_obs'][indices] / state_std,
        data['train_action'][indices].reshape(-1, 1) / action_std
    ])
    Y = data['train_deltas'][indices] / delta_std

    # Train separate GP for each state dimension
    gps = []
    for i in range(STATE_DIM):
        print(f"  Training GP for {STATE_NAMES_7D[i]}...")
        gp = GaussianProcessRegressor(
            kernel=kernel_config,
            n_restarts_optimizer=3,
            random_state=seed,
            alpha=1e-6
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)

    return gps


def evaluate_gp(gps, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate GP model."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']

    np.random.seed(seed)
    segments = []
    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= 1100:
            segments.append(ep)
    segments = segments[:n_segments]

    results = {}
    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

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
                    x = np.hstack([s_cur / state_std, [actions_seg[step] / action_std]]).reshape(1, -1)

                    # Predict with each GP
                    delta_pred = np.array([gp.predict(x)[0] for gp in gps])
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
                except:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors[:min(len(step_errors), n)])
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
            'per_state_nmae': {
                name: {'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan')}
                for name in STATE_NAMES_7D
            },
        }

    return results


def main():
    print("=" * 60)
    print("Optimize GP Hyperparameters")
    print("=" * 60)

    # Define kernel configurations to test
    kernel_configs = {
        'rbf_default': ConstantKernel(1.0) * RBF(length_scale=1.0),
        'rbf_short': ConstantKernel(1.0) * RBF(length_scale=0.1),
        'rbf_long': ConstantKernel(1.0) * RBF(length_scale=10.0),
        'matern_15': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
        'matern_25': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
        'rbf_noise': ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01),
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for name, kernel in kernel_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing kernel: {name}")
        print(f"{'='*60}")

        start_time = time.time()
        gps = train_gp_model(data, kernel, n_samples=5000, seed=42)
        train_time = time.time() - start_time

        print(f"\nEvaluating {name}...")
        results = evaluate_gp(gps, data, data['state_std'], data['action_std'], data['delta_std'], horizons)

        print(f"\nResults for {name}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        # Compute primary score
        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")
        print(f"Training time: {train_time:.1f}s")

        all_results[name] = {
            'results': results,
            'primary': primary,
            'train_time': train_time,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: GP Kernel Comparison")
    print("=" * 60)

    print(f"\n{'Kernel':<20} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 70)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{name:<20} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Find best kernel
    best_kernel = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    print(f"\nBest kernel: {best_kernel} (Primary = {all_results[best_kernel]['primary']:.4f})")

    # Compare with v9 baseline
    print(f"\nComparison with v9 baseline:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best GP':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    best_results = all_results[best_kernel]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        gp_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(gp_val):
            improvement = (v9_val - gp_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {gp_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'gp_optimization',
        'kernel_configs': list(kernel_configs.keys()),
        'results': {k: {'primary': v['primary'], 'train_time': v['train_time']} for k, v in all_results.items()},
        'best_kernel': best_kernel,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP010_gp_optimization.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

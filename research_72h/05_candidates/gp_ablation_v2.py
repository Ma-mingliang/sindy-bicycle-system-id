"""GP Ablation Study v2 - With hyperparameter optimization.

Tests different GP configurations with hyperparameter optimization enabled.
Uses small sample sizes for speed but with n_restarts_optimizer=1.
"""
import sys
import json
import time
import warnings
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

warnings.filterwarnings('ignore')  # Suppress convergence warnings for cleaner output

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

# Only test 2 key dimensions for speed
TEST_DIMS = [0, 4]  # e_y, theta_dot
STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
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


def train_gp_model(data, kernel, n_samples=500, seed=42, n_restarts=1):
    """Train GP model with hyperparameter optimization."""
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    n_total = len(data['train_obs'])
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    X = np.hstack([
        data['train_obs'][indices] / state_std,
        data['train_action'][indices].reshape(-1, 1) / action_std
    ])

    gps = []
    learned_kernels = []
    for dim_idx in TEST_DIMS:
        Y = data['train_deltas'][indices, dim_idx] / delta_std[dim_idx]
        t0 = time.time()
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
            alpha=1e-6
        )
        gp.fit(X, Y)
        gps.append(gp)
        learned_kernels.append(str(gp.kernel_))
        elapsed = time.time() - t0
        print(f'  {STATE_NAMES_7D[dim_idx]}: {elapsed:.1f}s, kernel={gp.kernel_}', flush=True)

    return gps, learned_kernels


def evaluate_gp(gps, data, state_std, action_std, delta_std, horizons, n_segments=2, seed=42):
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
                    delta_pred_full = np.zeros(7)
                    for j, dim_idx in enumerate(TEST_DIMS):
                        delta_pred_full[dim_idx] = gps[j].predict(x)[0]
                    dsdt = delta_pred_full * delta_std * dt
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
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and len(step_errors) >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
        }

    return results


def main():
    print("=" * 60, flush=True)
    print("GP Ablation Study v2 (With Hyperparameter Optimization)", flush=True)
    print("=" * 60, flush=True)

    # Define ablation configurations with n_restarts_optimizer=1
    ablation_configs = {
        'full': {
            'kernel': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'n_samples': 500,
            'n_restarts': 1,
            'description': 'Full GP (baseline, 500 samples)',
        },
        'small_data': {
            'kernel': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'n_samples': 200,
            'n_restarts': 1,
            'description': 'Small dataset (200 samples)',
        },
        'large_data': {
            'kernel': ConstantKernel(1.0) * RBF(length_scale=1.0),
            'n_samples': 1000,
            'n_restarts': 1,
            'description': 'Large dataset (1000 samples)',
        },
        'matern': {
            'kernel': ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
            'n_samples': 500,
            'n_restarts': 1,
            'description': 'Matern kernel (nu=2.5)',
        },
        'rbf_short': {
            'kernel': ConstantKernel(1.0) * RBF(length_scale=0.1),
            'n_samples': 500,
            'n_restarts': 1,
            'description': 'RBF with short length scale (0.1)',
        },
        'rbf_long': {
            'kernel': ConstantKernel(1.0) * RBF(length_scale=10.0),
            'n_samples': 500,
            'n_restarts': 1,
            'description': 'RBF with long length scale (10.0)',
        },
    }

    print("\n1. Loading data...", flush=True)
    data = load_data(seed=42)
    print("   Data loaded.", flush=True)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for name, config in ablation_configs.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Ablation: {name} - {config['description']}", flush=True)
        print(f"{'='*60}", flush=True)

        start_time = time.time()
        gps, learned_kernels = train_gp_model(
            data, config['kernel'], n_samples=config['n_samples'],
            seed=42, n_restarts=config['n_restarts']
        )
        train_time = time.time() - start_time

        print(f"\nEvaluating {name}...", flush=True)
        results = evaluate_gp(gps, data, data['state_std'], data['action_std'], data['delta_std'], horizons, n_segments=2)

        print(f"\nResults:", flush=True)
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}", flush=True)
        print("-" * 34, flush=True)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}", flush=True)

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}", flush=True)
        print(f"Training time: {train_time:.1f}s", flush=True)

        all_results[name] = {
            'results': results,
            'primary': primary,
            'train_time': train_time,
            'description': config['description'],
            'learned_kernels': learned_kernels,
        }

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY: GP Ablation Results", flush=True)
    print("=" * 60, flush=True)

    print(f"\n{'Config':<20} {'H=1':<10} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}", flush=True)
    print("-" * 80, flush=True)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{name:<20} {r[1]['nmae_mean']:<10.4f} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}", flush=True)

    # Find best configuration
    best_config = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    print(f"\nBest configuration: {best_config} (Primary = {all_results[best_config]['primary']:.4f})", flush=True)

    # Analysis sections
    print("\n" + "=" * 60, flush=True)
    print("1. DATA QUANTITY ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    for cfg in ['small_data', 'full', 'large_data']:
        if cfg in all_results:
            print(f"  {all_results[cfg]['description']:<35} Primary={all_results[cfg]['primary']:.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("2. KERNEL ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    for cfg in ['full', 'matern']:
        if cfg in all_results:
            print(f"  {all_results[cfg]['description']:<35} Primary={all_results[cfg]['primary']:.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("3. LENGTH SCALE ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    for cfg in ['rbf_short', 'full', 'rbf_long']:
        if cfg in all_results:
            print(f"  {all_results[cfg]['description']:<35} Primary={all_results[cfg]['primary']:.4f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("4. LEARNED KERNELS", flush=True)
    print("=" * 60, flush=True)
    for name, data_dict in all_results.items():
        print(f"\n  {name}:", flush=True)
        for i, k in enumerate(data_dict.get('learned_kernels', [])):
            print(f"    {STATE_NAMES_7D[TEST_DIMS[i]]}: {k}", flush=True)

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'gp_ablation',
        'configs': list(ablation_configs.keys()),
        'results': {},
        'best_config': best_config,
    }

    for name, data_dict in all_results.items():
        output['results'][name] = {
            'primary': data_dict['primary'],
            'train_time': data_dict['train_time'],
            'description': data_dict['description'],
            'learned_kernels': data_dict.get('learned_kernels', []),
            'horizons': {}
        }
        for h in horizons:
            r = data_dict['results'][h]
            output['results'][name]['horizons'][str(h)] = {
                'nmae_mean': r['nmae_mean'],
                'nmae_std': r['nmae_std'],
                'survival_rate': r['survival_rate'],
            }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP012_gp_ablation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == '__main__':
    main()

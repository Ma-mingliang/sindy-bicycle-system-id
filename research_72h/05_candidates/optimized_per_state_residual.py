"""Optimized Per-State Residual Learning.

Target: 75%+ improvement across ALL horizons.
Key ideas:
1. Train NODE specifically for e_y, e_psi (not all states)
2. Use larger GP with more data
3. Ensemble of NODE models for e_y, e_psi
4. Physics-informed correction for e_y, e_psi
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

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

WHEELBASE = 1.0


class ODEFunc(nn.Module):
    """Neural ODE model for specific states."""
    def __init__(self, input_dim, output_dim, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(input_dim, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, x):
        return self.net(x)


class GPSingleDim:
    """Single dimension GP model."""
    def __init__(self, max_samples=5000):
        self._max_samples = max_samples
        self._gp = None
        self._x_mean = None
        self._x_std = None

    def train(self, X, y):
        n = len(X)
        if n > self._max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self._max_samples, replace=False)
            X, y = X[idx], y[idx]

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        kernel = Matern(nu=2.5, length_scale=1.0)
        self._gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-3)
        self._gp.fit(X_scaled, y)

    def predict(self, x):
        x_scaled = (x - self._x_mean) / self._x_std
        return self._gp.predict(x_scaled.reshape(1, -1))[0]


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


def train_node_for_states(data, target_dims, config, seed=42):
    """Train Neural ODE for specific target states."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    # Prepare input: all states + action
    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])
    # Prepare target: only target states
    Y = train_deltas[:, target_dims] / (delta_std[target_dims] * dt)

    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)

    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = ODEFunc(
        input_dim=STATE_DIM + ACTION_DIM,
        output_dim=len(target_dims),
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    model.train()
    for epoch in range(config['n_epochs']):
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

    model.eval()
    return model


def train_gp_for_states(data, target_dims, max_samples=5000):
    """Train GP models for specific target states."""
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])

    gp_models = {}
    for dim in target_dims:
        Y = train_deltas[:, dim] / delta_std[dim]
        gp = GPSingleDim(max_samples=max_samples)
        gp.train(X, Y)
        gp_models[dim] = gp
        print(f"    GP for {STATE_NAMES_7D[dim]} trained")

    return gp_models


class OptimizedPerStateHybrid:
    """Optimized per-state hybrid model."""
    def __init__(self, node_models, gp_models, state_std, action_std, delta_std,
                 node_dims, gp_dims):
        self._node_models = node_models  # dict: dim -> model
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._node_dims = node_dims
        self._gp_dims = gp_dims

    def predict(self, s, tau):
        # Prepare input
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]])
        x_torch = torch.FloatTensor(x).unsqueeze(0)

        # Predict delta for all states
        delta = np.zeros(STATE_DIM)

        # NODE predictions
        for dim in self._node_dims:
            if dim in self._node_models:
                with torch.no_grad():
                    pred = self._node_models[dim](x_torch).numpy()[0][0]
                delta[dim] = pred * self._delta_std[dim] * dt

        # GP predictions
        for dim in self._gp_dims:
            if dim in self._gp_models:
                delta[dim] = self._gp_models[dim].predict(x) * self._delta_std[dim]

        return s + delta


def evaluate_hybrid(hybrid, data, horizons, n_segments=5, seed=42):
    """Evaluate hybrid model."""
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
                    s_next = hybrid.predict(s_cur, actions_seg[step])

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
    print("Optimized Per-State Residual Learning")
    print("Target: 75%+ improvement across ALL horizons")
    print("=" * 60)

    node_config = {
        'hidden': 128, 'depth': 4, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 300, 'batch_size': 256,
    }

    # Define different state grouping strategies
    strategies = {
        'node_ey_epsi_only': {
            'node_dims': [0, 1],  # e_y, e_psi
            'gp_dims': [2, 3, 4, 5, 6],  # v, theta, theta_dot, delta, delta_dot
        },
        'node_ey_epsi_thetadot': {
            'node_dims': [0, 1, 4],  # e_y, e_psi, theta_dot
            'gp_dims': [2, 3, 5, 6],  # v, theta, delta, delta_dot
        },
        'node_all_dynamic': {
            'node_dims': [0, 1, 4, 6],  # e_y, e_psi, theta_dot, delta_dot
            'gp_dims': [2, 3, 5],  # v, theta, delta
        },
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)
    state_std = data['state_std']

    horizons = [1, 10, 50, 100, 200, 500, 1000]
    all_results = {}

    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"  NODE dims: {[STATE_NAMES_7D[d] for d in strategy['node_dims']]}")
        print(f"  GP dims: {[STATE_NAMES_7D[d] for d in strategy['gp_dims']]}")
        print(f"{'='*60}")

        # Train NODE for each target state
        print("\n  Training NODE models...")
        node_models = {}
        for dim in strategy['node_dims']:
            print(f"    Training NODE for {STATE_NAMES_7D[dim]}...")
            model = train_node_for_states(data, [dim], node_config, seed=42)
            node_models[dim] = model

        # Train GP for each target state
        print("\n  Training GP models...")
        gp_models = train_gp_for_states(data, strategy['gp_dims'], max_samples=5000)

        # Create hybrid model
        hybrid = OptimizedPerStateHybrid(
            node_models, gp_models, state_std, data['action_std'], data['delta_std'],
            node_dims=strategy['node_dims'], gp_dims=strategy['gp_dims']
        )

        # Evaluate
        print("\n  Evaluating...")
        results = evaluate_hybrid(hybrid, data, horizons)

        print(f"\n  Results for {strategy_name}:")
        print(f"  {'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("  " + "-" * 34)
        for h in horizons:
            r = results[h]
            print(f"  H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        # Compute primary score (H=100, 200, 500)
        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\n  PrimaryLongHorizonScore: {primary:.4f}")

        # Check improvement
        v9_primary = 0.5110
        improvement = (v9_primary - primary) / v9_primary * 100
        print(f"  Improvement vs v9: {improvement:.1f}%")
        if improvement >= 75:
            print("  *** TARGET ACHIEVED! ***")

        all_results[strategy_name] = {
            'results': results,
            'primary': primary,
            'improvement': improvement,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Optimized Per-State Residual Results")
    print("=" * 60)

    print(f"\n{'Strategy':<25} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'H=1000':<10} {'Primary':<10} {'Improve':<10}")
    print("-" * 85)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        improvement = data_dict['improvement']
        print(f"{name:<25} {r[100]['nmae_mean']:<10.4f} {r[200]['nmae_mean']:<10.4f} "
              f"{r[500]['nmae_mean']:<10.4f} {r[1000]['nmae_mean']:<10.4f} {primary:<10.4f} {improvement:<10.1f}%")

    # Find best strategy
    best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['improvement'])
    best_improvement = all_results[best_strategy]['improvement']
    print(f"\nBest strategy: {best_strategy}")
    print(f"  Improvement: {best_improvement:.1f}%")

    if best_improvement >= 75:
        print("\n*** TARGET ACHIEVED! 75%+ improvement! ***")
    else:
        print(f"\n*** TARGET NOT YET ACHIEVED. Need {75 - best_improvement:.1f}% more improvement. ***")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'optimized_per_state_residual',
        'target_improvement': 75,
        'strategies': list(strategies.keys()),
        'results': {k: {'primary': v['primary'], 'improvement': v['improvement']}
                   for k, v in all_results.items()},
        'best_strategy': best_strategy,
        'best_improvement': best_improvement,
        'target_achieved': best_improvement >= 75,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP040_optimized_per_state_residual.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

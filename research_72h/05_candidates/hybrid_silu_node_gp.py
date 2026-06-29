"""Hybrid SiLU NODE + GP Model.

First principles approach:
- Complex states (e_y, e_psi, theta_dot, delta_dot): Wide SiLU NODE
- Smooth states (v, theta, delta): GP

Rationale:
- GP works well for smooth states (R²=0.94-1.0)
- Wide SiLU NODE captures complex couplings
- Joint prediction for complex states preserves dependencies
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


class WideSiLUODEFunc(nn.Module):
    """Wide Neural ODE with SiLU activation."""
    def __init__(self, input_dim, output_dim, hidden=256, depth=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, x):
        return self.net(x)


class GPSingleDim:
    """Single dimension GP model."""
    def __init__(self, max_samples=3000):
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


def train_silu_node_for_complex_states(data, target_dims, config, seed=42):
    """Train Wide SiLU NODE for complex states."""
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

    model = WideSiLUODEFunc(
        input_dim=STATE_DIM + ACTION_DIM,
        output_dim=len(target_dims),
        hidden=config['hidden'],
        depth=config['depth']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"  Training Wide SiLU NODE for {[STATE_NAMES_7D[d] for d in target_dims]}...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s")

    return model


def train_gp_for_smooth_states(data, target_dims, max_samples=3000):
    """Train GP models for smooth states."""
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


class HybridSiLUNodeGP:
    """Hybrid model: Wide SiLU NODE for complex states, GP for smooth states."""
    def __init__(self, silu_model, gp_models, state_std, action_std, delta_std,
                 silu_dims, gp_dims):
        self._silu_model = silu_model
        self._gp_models = gp_models
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._silu_dims = silu_dims
        self._gp_dims = gp_dims

    def predict(self, s, tau):
        # Prepare input
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]])
        x_torch = torch.FloatTensor(x).unsqueeze(0)

        # Predict delta for all states
        delta = np.zeros(STATE_DIM)

        # Wide SiLU NODE predictions for complex states
        with torch.no_grad():
            pred = self._silu_model(x_torch).numpy()[0]
        for i, dim in enumerate(self._silu_dims):
            delta[dim] = pred[i] * self._delta_std[dim] * dt

        # GP predictions for smooth states
        for dim in self._gp_dims:
            if dim in self._gp_models:
                delta[dim] = self._gp_models[dim].predict(x) * self._delta_std[dim]

        return s + delta


def evaluate_hybrid(hybrid, data, horizons, n_segments=5, seed=42):
    """Evaluate hybrid model."""
    dt = 1.0 / 30.0
    state_std = data['state_std']
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
    print("Hybrid SiLU NODE + GP Model")
    print("=" * 60)

    silu_config = {
        'hidden': 256, 'depth': 5,
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    # State grouping based on analysis:
    # Complex states: e_y, e_psi, theta_dot, delta_dot (dynamic, hard to predict)
    # Smooth states: v, theta, delta (smooth, GP works well)
    silu_dims = [0, 1, 4, 6]  # e_y, e_psi, theta_dot, delta_dot
    gp_dims = [2, 3, 5]  # v, theta, delta

    print(f"\nState grouping:")
    print(f"  SiLU NODE: {[STATE_NAMES_7D[d] for d in silu_dims]}")
    print(f"  GP: {[STATE_NAMES_7D[d] for d in gp_dims]}")

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training Wide SiLU NODE for complex states...")
    silu_model = train_silu_node_for_complex_states(data, silu_dims, silu_config, seed=42)

    print("\n3. Training GP for smooth states...")
    gp_models = train_gp_for_smooth_states(data, gp_dims, max_samples=3000)

    print("\n4. Creating hybrid model...")
    hybrid = HybridSiLUNodeGP(
        silu_model, gp_models, data['state_std'], data['action_std'], data['delta_std'],
        silu_dims=silu_dims, gp_dims=gp_dims
    )

    print("\n5. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500, 1000]
    results = evaluate_hybrid(hybrid, data, horizons)

    print("\n" + "=" * 60)
    print("RESULTS: Hybrid SiLU NODE + GP")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Compare with baselines
    print(f"\nComparison with baselines:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Hybrid':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443}
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        h_val = results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(h_val):
            improvement = (v9_val - h_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {h_val:<12.4f} {improvement:<12.1f}%")

    # Compute primary score
    primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
    v9_primary = 0.5110
    improvement = (v9_primary - primary) / v9_primary * 100
    print(f"\nPrimaryLongHorizonScore: {primary:.4f}")
    print(f"Improvement vs v9: {improvement:.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'hybrid_silu_node_gp',
        'silu_dims': [STATE_NAMES_7D[d] for d in silu_dims],
        'gp_dims': [STATE_NAMES_7D[d] for d in gp_dims],
        'results': {str(h): {'nmae_mean': r['nmae_mean'], 'survival_rate': r['survival_rate']}
                   for h, r in results.items()},
        'primary_score': primary,
        'improvement_vs_v9': improvement,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP048_hybrid_silu_node_gp.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

"""Coupling Network for Per-State Prediction.

Core idea: Use a shared encoder to capture state coupling,
then use state-specific decoders to predict each state's delta.

Architecture:
- Shared Encoder: Takes all states → outputs coupling vector z
- State Decoders: Each takes (z, own_state) → predicts delta for that state

This preserves coupling while allowing state-specific learning.
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

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


class CouplingEncoder(nn.Module):
    """Shared encoder that captures state coupling."""
    def __init__(self, input_dim, coupling_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, coupling_dim),
        )

    def forward(self, x):
        """Input: all states + action → Output: coupling vector z."""
        return self.net(x)


class StateDecoder(nn.Module):
    """State-specific decoder that predicts delta for one state."""
    def __init__(self, coupling_dim, hidden=32):
        super().__init__()
        # Takes coupling vector + own state → predicts delta
        self.net = nn.Sequential(
            nn.Linear(coupling_dim + 1, hidden),  # +1 for own state
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Initialize small for residual learning
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, z, own_state):
        """Input: coupling vector z + own state → Output: delta for this state."""
        x = torch.cat([z, own_state.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


class CouplingNetwork(nn.Module):
    """Coupling Network for Per-State Prediction.

    Architecture:
    - Shared Encoder: All states → coupling vector z
    - 7 State Decoders: (z, own_state) → delta for each state
    """
    def __init__(self, state_dim=7, action_dim=1, coupling_dim=32, hidden=64):
        super().__init__()

        # Shared encoder: all states + action → coupling vector
        self.encoder = CouplingEncoder(
            input_dim=state_dim + action_dim,
            coupling_dim=coupling_dim,
            hidden=hidden
        )

        # State-specific decoders
        self.decoders = nn.ModuleList([
            StateDecoder(coupling_dim=coupling_dim, hidden=32)
            for _ in range(state_dim)
        ])

    def forward(self, s, a):
        """Predict delta for all states.

        Args:
            s: (batch, state_dim) - normalized states
            a: (batch, 1) - normalized actions

        Returns:
            delta: (batch, state_dim) - predicted deltas
        """
        # Shared encoder: capture coupling
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)

        # State-specific decoders
        deltas = []
        for i, decoder in enumerate(self.decoders):
            delta_i = decoder(z, s[:, i])
            deltas.append(delta_i)

        return torch.stack(deltas, dim=-1)


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


def train_coupling_network(data, config, seed=42):
    """Train coupling network."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_deltas = data['train_deltas']

    # Prepare data
    X = np.hstack([train_obs / state_std, train_action.reshape(-1, 1) / action_std])
    Y = train_deltas / (delta_std * dt)

    X_t = torch.FloatTensor(X)
    Y_t = torch.FloatTensor(Y)

    ds = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = CouplingNetwork(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        coupling_dim=config['coupling_dim'],
        hidden=config['hidden']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"Training Coupling Network (seed={seed})...")
    print(f"  Coupling dim: {config['coupling_dim']}")
    print(f"  Hidden: {config['hidden']}")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            s = xb[:, :STATE_DIM]
            a = xb[:, STATE_DIM:]

            pred = model(s, a)
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
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


def evaluate_coupling(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate coupling network."""
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
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                    a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)

                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).numpy()[0]

                    dsdt = dsdt_norm * delta_std * dt
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
    print("Coupling Network for Per-State Prediction")
    print("=" * 60)

    # Test different coupling dimensions
    configs = [
        {'name': 'small_coupling', 'coupling_dim': 16, 'hidden': 32, 'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
        {'name': 'medium_coupling', 'coupling_dim': 32, 'hidden': 64, 'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
        {'name': 'large_coupling', 'coupling_dim': 64, 'hidden': 128, 'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256},
    ]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    horizons = [1, 10, 50, 100, 200, 500, 1000]
    all_results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"{'='*60}")

        model, state_std, action_std, delta_std = train_coupling_network(data, config, seed=42)

        print(f"\nEvaluating...")
        results = evaluate_coupling(model, data, state_std, action_std, delta_std, horizons)

        print(f"\nResults for {config['name']}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

        all_results[config['name']] = {
            'results': results,
            'primary': primary,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Coupling Network Results")
    print("=" * 60)

    print(f"\n{'Config':<20} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 70)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{name:<20} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Compare with baselines
    print(f"\nComparison with baselines:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best Coupling':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443}
    best_name = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_results = all_results[best_name]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        c_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(c_val):
            improvement = (v9_val - c_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {c_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'coupling_network',
        'configs': [c['name'] for c in configs],
        'results': {k: {'primary': v['primary']} for k, v in all_results.items()},
        'best_config': best_name,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP050_coupling_network.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

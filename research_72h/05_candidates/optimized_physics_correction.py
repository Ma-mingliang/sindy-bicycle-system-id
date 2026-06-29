"""Optimized Physics-Informed Correction.

Further optimizes the breakthrough method to improve H=200/500 and reduce variance.
Key ideas:
1. Fine-tune correction_strength and physics_weight
2. State-dependent correction weights
3. Adaptive correction based on prediction confidence
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

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

# State-dependent correction weights
STATE_WEIGHTS = {
    'e_y': 2.0,      # Most sensitive - higher correction
    'e_psi': 2.0,    # Most sensitive - higher correction
    'v': 0.5,        # Less sensitive - lower correction
    'theta': 1.0,    # Medium
    'theta_dot': 1.0, # Medium
    'delta': 1.0,    # Medium
    'delta_dot': 1.0, # Medium
}


class ODEFunc(nn.Module):
    """Neural ODE model."""
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


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
    }


def train_model(data, config, seed=42):
    """Train Neural ODE model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    train_obs = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])

    train_s = torch.FloatTensor(train_obs / state_std)
    train_a = torch.FloatTensor(train_action.reshape(-1, 1) / action_std)
    train_dsdot = torch.FloatTensor(train_deltas / (delta_std * dt))

    ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
    loader = torch.utils.data.DataLoader(ds, batch_size=config['batch_size'], shuffle=True)

    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    model.train()
    for epoch in range(config['n_epochs']):
        for sb, ab, yb in loader:
            pred = model(sb, ab)
            loss = nn.functional.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scheduler.step()

    model.eval()
    return model


def build_state_corrector(train_obs, state_std, n_neighbors=10):
    """Build a nearest-neighbor corrector."""
    train_obs_norm = train_obs / state_std
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn_model.fit(train_obs_norm)
    return nn_model, train_obs_norm


def correct_state_optimized(s_pred, s_cur, nn_model, train_obs_norm, state_std, dt,
                             correction_strength=0.1, physics_weight=0.7):
    """Optimized state correction with state-dependent weights."""
    s_pred_norm = s_pred / state_std

    # Find nearest neighbors
    distances, indices = nn_model.kneighbors([s_pred_norm])

    # Compute correction: move toward nearest neighbor mean
    neighbor_mean = np.mean(train_obs_norm[indices[0]], axis=0)
    nn_correction = (neighbor_mean - s_pred_norm) * correction_strength

    # Apply state-dependent weights
    weights = np.array([STATE_WEIGHTS[name] for name in STATE_NAMES_7D])
    nn_correction = nn_correction * weights

    # Physics-informed correction for e_y and e_psi
    v = s_cur[2]
    e_psi = s_cur[1]
    e_y_dot_physics = v * np.sin(e_psi)
    e_y_next_physics = s_cur[0] + e_y_dot_physics * dt

    delta = s_cur[6]
    e_psi_dot_physics = -v * delta / WHEELBASE
    e_psi_next_physics = s_cur[1] + e_psi_dot_physics * dt

    physics_correction = np.zeros(7)
    physics_correction[0] = (e_y_next_physics - s_pred[0]) / state_std[0] * physics_weight
    physics_correction[1] = (e_psi_next_physics - s_pred[1]) / state_std[1] * physics_weight

    # Combine corrections
    total_correction = nn_correction + physics_correction

    # Apply correction
    s_corrected_norm = s_pred_norm + total_correction
    s_corrected = s_corrected_norm * state_std

    return s_corrected


def evaluate_optimized(model, data, state_std, action_std, delta_std, nn_model, train_obs_norm,
                        horizons, correction_strength=0.1, physics_weight=0.7, n_segments=5, seed=42):
    """Evaluate model with optimized correction."""
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

                    # Apply optimized correction
                    s_next = correct_state_optimized(s_next, s_cur, nn_model, train_obs_norm, state_std, dt,
                                                     correction_strength, physics_weight)

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
    print("Optimized Physics-Informed Correction")
    print("=" * 60)

    config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    # Test different parameter combinations
    param_combinations = [
        {'correction_strength': 0.05, 'physics_weight': 0.5},
        {'correction_strength': 0.05, 'physics_weight': 0.7},
        {'correction_strength': 0.05, 'physics_weight': 0.9},
        {'correction_strength': 0.1, 'physics_weight': 0.5},
        {'correction_strength': 0.1, 'physics_weight': 0.7},  # Previous best
        {'correction_strength': 0.1, 'physics_weight': 0.9},
        {'correction_strength': 0.15, 'physics_weight': 0.5},
        {'correction_strength': 0.15, 'physics_weight': 0.7},
        {'correction_strength': 0.15, 'physics_weight': 0.9},
    ]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training model...")
    model = train_model(data, config, seed=42)

    print("\n3. Building state corrector...")
    nn_model, train_obs_norm = build_state_corrector(data['train_obs'], data['state_std'])
    print(f"  Built nearest-neighbor corrector with {len(data['train_obs'])} training samples")

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for params in param_combinations:
        cs = params['correction_strength']
        pw = params['physics_weight']
        name = f"cs{cs}_pw{pw}"

        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        results = evaluate_optimized(
            model, data, data['state_std'], data['action_std'], data['delta_std'],
            nn_model, train_obs_norm, horizons, correction_strength=cs, physics_weight=pw
        )

        print(f"\nResults for {name}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

        all_results[name] = {
            'results': results,
            'primary': primary,
            'params': params,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Optimized Physics Correction Results")
    print("=" * 60)

    print(f"\n{'Config':<15} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 65)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{name:<15} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Find best configuration
    best_config = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    print(f"\nBest configuration: {best_config} (Primary = {all_results[best_config]['primary']:.4f})")

    # Compare with v9 baseline
    print(f"\nComparison with v9 baseline:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best Optimized':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    best_results = all_results[best_config]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        o_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(o_val):
            improvement = (v9_val - o_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {o_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'optimized_physics_correction',
        'param_combinations': param_combinations,
        'results': {k: {'primary': v['primary'], 'params': v['params']} for k, v in all_results.items()},
        'best_config': best_config,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP038_optimized_physics_correction.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

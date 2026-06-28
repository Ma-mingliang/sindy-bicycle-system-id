"""Quick Screen: v9_fixed + RK4 Integration.

Tests if RK4 integration improves long-horizon prediction.
"""
import sys
import os
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


class ODEFunc(nn.Module):
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
    }


def predict_euler(model, s, a, state_std, action_std, delta_std, dt):
    """Euler integration: x_{t+1} = x_t + dt * f(x_t, u_t)"""
    s_norm = torch.FloatTensor(s / state_std).unsqueeze(0)
    a_norm = torch.FloatTensor([a / action_std]).unsqueeze(0)
    with torch.no_grad():
        dsdt_norm = model(s_norm, a_norm).numpy()[0]
    dsdt = dsdt_norm * delta_std * dt
    return s + dsdt


def predict_rk4(model, s, a, state_std, action_std, delta_std, dt):
    """RK4 integration."""
    s_tensor = torch.FloatTensor(s / state_std).unsqueeze(0)
    a_tensor = torch.FloatTensor([a / action_std]).unsqueeze(0)

    with torch.no_grad():
        # k1 = f(s, a) * dt
        k1_norm = model(s_tensor, a_tensor).numpy()[0]
        k1 = k1_norm * delta_std * dt

        # k2 = f(s + k1/2, a) * dt
        s_mid1 = (s + k1 / 2) / state_std
        k2_norm = model(torch.FloatTensor(s_mid1).unsqueeze(0), a_tensor).numpy()[0]
        k2 = k2_norm * delta_std * dt

        # k3 = f(s + k2/2, a) * dt
        s_mid2 = (s + k2 / 2) / state_std
        k3_norm = model(torch.FloatTensor(s_mid2).unsqueeze(0), a_tensor).numpy()[0]
        k3 = k3_norm * delta_std * dt

        # k4 = f(s + k3, a) * dt
        s_end = (s + k3) / state_std
        k4_norm = model(torch.FloatTensor(s_end).unsqueeze(0), a_tensor).numpy()[0]
        k4 = k4_norm * delta_std * dt

    return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def evaluate(model, data, state_std, action_std, delta_std, horizons, method='euler', n_segments=5, seed=42):
    """Evaluate model with specified integration method."""
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
                    if method == 'rk4':
                        s_next = predict_rk4(model, s_cur, actions_seg[step],
                                           state_std, action_std, delta_std, dt)
                    else:
                        s_next = predict_euler(model, s_cur, actions_seg[step],
                                             state_std, action_std, delta_std, dt)

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
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                } for name in STATE_NAMES_7D
            },
        }

    return results


def main():
    print("=" * 60)
    print("Quick Screen: Euler vs RK4 Integration")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data = load_data(seed=42)

    # Load v9_fixed model (if available)
    model_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/v9_fixed_seed43.pt'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Using random model for demonstration")
        model = ODEFunc(hidden=64, depth=3, activation='tanh')
        state_std = data['state_std']
        action_std = data['action_std']
        delta_std = data['delta_std']
    else:
        print(f"Loading model: {model_path}")
        ckpt = torch.load(model_path, weights_only=False)
        model = ODEFunc(**{k: v for k, v in ckpt['config'].items()
                          if k in ['hidden', 'depth', 'activation']})
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        state_std = ckpt['state_std']
        action_std = ckpt['action_std']
        delta_std = ckpt['delta_std']

    # Evaluate with Euler
    print("\n2. Evaluating with Euler integration...")
    horizons = [1, 10, 50, 100, 200, 500]
    euler_results = evaluate(model, data, state_std, action_std, delta_std,
                            horizons, method='euler', n_segments=5)

    # Evaluate with RK4
    print("\n3. Evaluating with RK4 integration...")
    rk4_results = evaluate(model, data, state_std, action_std, delta_std,
                          horizons, method='rk4', n_segments=5)

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON: Euler vs RK4")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'Euler NMAE':<15} {'RK4 NMAE':<15} {'Improvement':<15}")
    print("-" * 55)
    for h in horizons:
        e_nmae = euler_results[h]['nmae_mean']
        r_nmae = rk4_results[h]['nmae_mean']
        if e_nmae > 0 and not np.isnan(e_nmae) and not np.isnan(r_nmae):
            improvement = (e_nmae - r_nmae) / e_nmae * 100
            print(f"H={h:<7} {e_nmae:<15.4f} {r_nmae:<15.4f} {improvement:<15.1f}%")
        else:
            print(f"H={h:<7} {e_nmae:<15.4f} {r_nmae:<15.4f} {'N/A':<15}")

    # Per-state comparison for H=100
    print(f"\nPer-state NMAE (H=100):")
    print(f"{'State':<12} {'Euler':<12} {'RK4':<12} {'Improvement':<12}")
    print("-" * 48)
    for name in STATE_NAMES_7D:
        e_val = euler_results[100]['per_state_nmae'][name]['mean']
        r_val = rk4_results[100]['per_state_nmae'][name]['mean']
        if e_val > 0 and not np.isnan(e_val) and not np.isnan(r_val):
            improvement = (e_val - r_val) / e_val * 100
            print(f"{name:<12} {e_val:<12.4f} {r_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'euler_vs_rk4',
        'model': 'v9_fixed_seed43',
        'euler_results': euler_results,
        'rk4_results': rk4_results,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP002_euler_vs_rk4.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

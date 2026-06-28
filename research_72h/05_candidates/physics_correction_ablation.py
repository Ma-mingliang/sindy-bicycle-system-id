"""Physics-Informed Correction Ablation Study.

Tests the effect of different components:
1. NN correction only
2. Physics correction only
3. Combined (best configuration)
4. Different correction strengths
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


def correct_state_ablation(s_pred, s_cur, nn_model, train_obs_norm, state_std, dt,
                            correction_strength=0.1, physics_weight=0.7, use_nn=True, use_physics=True):
    """Correct predicted state with ablation options."""
    s_pred_norm = s_pred / state_std
    total_correction = np.zeros(7)

    # NN correction
    if use_nn:
        distances, indices = nn_model.kneighbors([s_pred_norm])
        neighbor_mean = np.mean(train_obs_norm[indices[0]], axis=0)
        nn_correction = (neighbor_mean - s_pred_norm) * correction_strength
        total_correction += nn_correction

    # Physics correction
    if use_physics:
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
        total_correction += physics_correction

    # Apply correction
    s_corrected_norm = s_pred_norm + total_correction
    s_corrected = s_corrected_norm * state_std

    return s_corrected


def evaluate_with_correction(model, data, state_std, action_std, delta_std, nn_model, train_obs_norm,
                              horizons, correction_strength=0.1, physics_weight=0.7,
                              use_nn=True, use_physics=True, n_segments=5, seed=42):
    """Evaluate model with ablation options."""
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

                    # Apply correction with ablation options
                    s_next = correct_state_ablation(s_next, s_cur, nn_model, train_obs_norm, state_std, dt,
                                                    correction_strength, physics_weight, use_nn, use_physics)

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
    print("Physics-Informed Correction Ablation Study")
    print("=" * 60)

    config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    # Define ablation configurations
    ablation_configs = {
        'no_correction': {
            'correction_strength': 0.0,
            'physics_weight': 0.0,
            'use_nn': False,
            'use_physics': False,
        },
        'nn_only': {
            'correction_strength': 0.1,
            'physics_weight': 0.0,
            'use_nn': True,
            'use_physics': False,
        },
        'physics_only': {
            'correction_strength': 0.0,
            'physics_weight': 0.7,
            'use_nn': False,
            'use_physics': True,
        },
        'combined': {
            'correction_strength': 0.1,
            'physics_weight': 0.7,
            'use_nn': True,
            'use_physics': True,
        },
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training model...")
    model = train_model(data, config, seed=42)

    print("\n3. Building state corrector...")
    nn_model, train_obs_norm = build_state_corrector(data['train_obs'], data['state_std'])
    print(f"  Built nearest-neighbor corrector with {len(data['train_obs'])} training samples")

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for name, ablation_config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {name}")
        print(f"{'='*60}")

        results = evaluate_with_correction(
            model, data, data['state_std'], data['action_std'], data['delta_std'],
            nn_model, train_obs_norm, horizons, **ablation_config
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
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Ablation Results")
    print("=" * 60)

    print(f"\n{'Config':<20} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 70)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{name:<20} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Find best configuration
    best_config = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    print(f"\nBest configuration: {best_config} (Primary = {all_results[best_config]['primary']:.4f})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'physics_correction_ablation',
        'configs': list(ablation_configs.keys()),
        'results': {k: {'primary': v['primary']} for k, v in all_results.items()},
        'best_config': best_config,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP034_physics_correction_ablation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

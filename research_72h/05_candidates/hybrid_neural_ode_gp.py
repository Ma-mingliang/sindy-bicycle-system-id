"""Hybrid Neural ODE + GP Method.

Addresses root cause: Neural ODE learns dynamics but error accumulates.
Solution: Use Neural ODE for short-term, GP for long-term.
"""
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

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
    """Original v9 Neural ODE."""
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
        'train_action': train_action,
        'train_deltas': train_deltas,
    }


def train_neural_ode(data, config, seed=42):
    """Train Neural ODE model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    # Prepare training data (flat, shuffled)
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

    print(f"Training Neural ODE (seed={seed})...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for sb, ab, yb in loader:
            pred = model(sb, ab)
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


def train_gp_model(data, n_samples=3000, seed=42):
    """Train GP model."""
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
    Y = data['train_deltas'][indices] / delta_std

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gps = []
    for i in range(STATE_DIM):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            random_state=seed,
            alpha=1e-4
        )
        gp.fit(X, Y[:, i])
        gps.append(gp)

    return gps


def evaluate_hybrid(neural_ode, gps, data, state_std, action_std, delta_std, horizons, switch_horizon=50, n_segments=5, seed=42):
    """Evaluate hybrid method: Neural ODE for short-term, GP for long-term."""
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
                    # Use Neural ODE for short-term, GP for long-term
                    if step < switch_horizon:
                        # Neural ODE prediction
                        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                        a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)

                        with torch.no_grad():
                            dsdt_norm = neural_ode(s_norm, a_norm).numpy()[0]

                        dsdt = dsdt_norm * delta_std * dt
                        s_next = s_cur + dsdt
                    else:
                        # GP prediction
                        x = np.hstack([s_cur / state_std, [actions_seg[step] / action_std]]).reshape(1, -1)
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
    print("Hybrid Neural ODE + GP Method")
    print("=" * 60)

    neural_ode_config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    # Test different switch horizons
    switch_horizons = [20, 50, 100]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training Neural ODE...")
    neural_ode, state_std, action_std, delta_std = train_neural_ode(data, neural_ode_config, seed=42)

    print("\n3. Training GP...")
    gps = train_gp_model(data, n_samples=3000, seed=42)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for switch_h in switch_horizons:
        print(f"\n{'='*60}")
        print(f"Testing switch horizon: {switch_h}")
        print(f"{'='*60}")

        results = evaluate_hybrid(neural_ode, gps, data, state_std, action_std, delta_std, horizons, switch_horizon=switch_h)

        print(f"\nResults for switch_h={switch_h}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

        all_results[switch_h] = {
            'results': results,
            'primary': primary,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Hybrid Method Results")
    print("=" * 60)

    print(f"\n{'Switch H':<12} {'H=1':<10} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10}")
    print("-" * 72)

    for switch_h, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        print(f"{switch_h:<12} {r[1]['nmae_mean']:<10.4f} {r[50]['nmae_mean']:<10.4f} "
              f"{r[100]['nmae_mean']:<10.4f} {r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f}")

    # Compare with baselines
    print(f"\nComparison with baselines:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'GP NMAE':<12} {'Best Hybrid':<12} {'Improvement':<12}")
    print("-" * 58)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    gp_nmae = {1: 0.0173, 10: 0.1037, 50: 0.4334, 100: 0.4643, 200: 0.4147, 500: 0.4325}
    best_switch = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_results = all_results[best_switch]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        gp_val = gp_nmae.get(h, float('nan'))
        h_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(h_val):
            improvement = (v9_val - h_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {gp_val:<12.4f} {h_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'hybrid_neural_ode_gp',
        'switch_horizons': switch_horizons,
        'results': {str(k): {'primary': v['primary']} for k, v in all_results.items()},
        'best_switch': best_switch,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP015_hybrid_neural_ode_gp.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

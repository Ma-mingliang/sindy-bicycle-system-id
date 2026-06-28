"""Physics Residual Model.

Addresses root cause: Neural ODE learns dynamics but error accumulates.
Solution: Use known physics as base, Neural ODE learns residual.
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

# Physical parameters (simplified bicycle model)
WHEELBASE = 1.0  # L
GRAVITY = 9.81
HEIGHT_CG = 0.8  # h


class PhysicsResidualFunc(nn.Module):
    """Physics + Neural ODE residual model.

    dx/dt = f_physics(x, u) + f_residual(x, u)

    where f_physics is known bicycle dynamics, f_residual is learned.
    """
    def __init__(self, hidden=64, depth=3, activation='tanh', residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale

        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Initialize last layer small for residual
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def physics_derivatives(self, s, a):
        """Compute known physics derivatives.

        State: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]

        Known relationships:
        - e_y_dot = v * sin(e_psi)  (exact kinematics)
        - e_psi_dot = -v * delta / L  (exact kinematics)
        - v_dot ≈ 0  (velocity nearly constant)
        - theta_dot = theta_dot  (definition)
        - theta_ddot ≈ (g/h) * theta - (v^2/(h*L)) * delta  (simplified Whipple)
        - delta_dot = delta_dot  (definition)
        - delta_ddot ≈ -omega_n^2 * delta - 2*zeta*omega_n * delta_dot + u  (simplified)
        """
        # Unpack state
        e_y = s[:, 0]
        e_psi = s[:, 1]
        v = s[:, 2]
        theta = s[:, 3]
        theta_dot = s[:, 4]
        delta = s[:, 5]
        delta_dot = s[:, 6]
        u = a[:, 0]

        # Physical derivatives
        e_y_dot = v * torch.sin(e_psi)
        e_psi_dot = -v * delta / WHEELBASE
        v_dot = torch.zeros_like(v)  # Velocity nearly constant
        theta_dot_phys = theta_dot  # Definition
        theta_ddot = (GRAVITY / HEIGHT_CG) * theta - (v ** 2 / (HEIGHT_CG * WHEELBASE)) * delta
        delta_dot_phys = delta_dot  # Definition
        delta_ddot = -25.0 * delta - 5.0 * delta_dot + u  # Simplified steering dynamics

        # Stack derivatives
        dsdt_physics = torch.stack([
            e_y_dot,
            e_psi_dot,
            v_dot,
            theta_dot_phys,
            theta_ddot,
            delta_dot_phys,
            delta_ddot
        ], dim=-1)

        return dsdt_physics

    def forward(self, s, a):
        """Compute total derivatives: physics + residual."""
        # Physics derivatives
        dsdt_physics = self.physics_derivatives(s, a)

        # Neural residual
        x = torch.cat([s, a], dim=-1)
        dsdt_residual = self.net(x) * self.residual_scale

        # Total
        dsdt_total = dsdt_physics + dsdt_residual

        return dsdt_total


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


def train_physics_residual(data, config, seed=43):
    """Train physics residual model."""
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

    model = PhysicsResidualFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation'],
        residual_scale=config.get('residual_scale', 0.1)
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    print(f"Training Physics Residual model (seed={seed})...")
    print(f"  residual_scale: {config.get('residual_scale', 0.1)}")
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


def evaluate(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate model."""
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
    print("Physics Residual Model")
    print("=" * 60)

    # Test different residual scales
    residual_configs = {
        'small_residual': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'residual_scale': 0.01,
        },
        'medium_residual': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'residual_scale': 0.1,
        },
        'large_residual': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'residual_scale': 1.0,
        },
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for name, config in residual_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        model, state_std, action_std, delta_std = train_physics_residual(data, config, seed=43)

        print(f"\nEvaluating {name}...")
        results = evaluate(model, data, state_std, action_std, delta_std, horizons)

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
    print("SUMMARY: Physics Residual Results")
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
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best Physics':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    best_name = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_results = all_results[best_name]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        p_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(p_val):
            improvement = (v9_val - p_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {p_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'physics_residual',
        'configs': list(residual_configs.keys()),
        'results': {k: {'primary': v['primary']} for k, v in all_results.items()},
        'best_config': best_name,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP016_physics_residual.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

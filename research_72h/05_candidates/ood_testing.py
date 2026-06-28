"""OOD Testing for Best Model.

Tests the best model (noise-augmented no_noise) on out-of-distribution data.
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
    }


def train_model(data, config, seed=43):
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

    print(f"Training model (seed={seed})...")
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


def generate_ood_scenarios(data, state_std, n_scenarios=5):
    """Generate OOD scenarios by perturbing states."""
    episodes = data['episodes']
    test_eps = data['test_eps']

    ood_scenarios = []

    # Scenario 1: High speed (scale v by 1.5)
    for ep_idx in test_eps[:2]:
        ep = episodes[ep_idx]
        if ep['length'] >= 500:
            states = ep['obs'][:500].copy()
            states[:, 2] *= 1.5  # Scale velocity
            ood_scenarios.append({
                'name': 'high_speed',
                'states': states,
                'actions': ep['action'][:500],
            })

    # Scenario 2: Large initial heading error
    for ep_idx in test_eps[:2]:
        ep = episodes[ep_idx]
        if ep['length'] >= 500:
            states = ep['obs'][:500].copy()
            states[0, 1] = 0.8  # Large initial heading error
            ood_scenarios.append({
                'name': 'large_heading_error',
                'states': states,
                'actions': ep['action'][:500],
            })

    # Scenario 3: Large initial lateral error
    for ep_idx in test_eps[:2]:
        ep = episodes[ep_idx]
        if ep['length'] >= 500:
            states = ep['obs'][:500].copy()
            states[0, 0] = 0.4  # Large initial lateral error
            ood_scenarios.append({
                'name': 'large_lateral_error',
                'states': states,
                'actions': ep['action'][:500],
            })

    # Scenario 4: Aggressive steering
    for ep_idx in test_eps[:2]:
        ep = episodes[ep_idx]
        if ep['length'] >= 500:
            states = ep['obs'][:500].copy()
            actions = ep['action'][:500].copy()
            actions *= 2.0  # Double steering
            ood_scenarios.append({
                'name': 'aggressive_steering',
                'states': states,
                'actions': actions,
            })

    return ood_scenarios


def evaluate_ood(model, ood_scenarios, state_std, action_std, delta_std, horizons):
    """Evaluate model on OOD scenarios."""
    dt = 1.0 / 30.0

    results = {}
    for scenario in ood_scenarios:
        name = scenario['name']
        states = scenario['states']
        actions = scenario['actions']

        scenario_results = {}
        for h in horizons:
            n = min(h, len(actions))
            s_cur = states[0].copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                    a_norm = torch.FloatTensor([actions[step] / action_std]).unsqueeze(0)

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
                    if step + 1 < len(states):
                        step_err = np.abs(s_next - states[step + 1]) / state_std
                        step_errors.append(step_err)
                    s_cur = s_next
                except:
                    survived = False
                    break

            if step_errors:
                errors = np.array(step_errors[:min(len(step_errors), n)])
                nmae = np.mean(errors)
            else:
                nmae = float('nan')

            scenario_results[h] = {
                'nmae_mean': float(nmae) if not np.isnan(nmae) else float('nan'),
                'survival_rate': 1.0 if survived else 0.0,
            }

        results[name] = scenario_results

    return results


def main():
    print("=" * 60)
    print("OOD Testing for Best Model")
    print("=" * 60)

    config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training model...")
    model, state_std, action_std, delta_std = train_model(data, config, seed=43)

    print("\n3. Generating OOD scenarios...")
    ood_scenarios = generate_ood_scenarios(data, state_std)
    print(f"  Generated {len(ood_scenarios)} OOD scenarios")

    print("\n4. Evaluating on OOD scenarios...")
    horizons = [1, 10, 50, 100, 200, 500]
    ood_results = evaluate_ood(model, ood_scenarios, state_std, action_std, delta_std, horizons)

    # Print results
    print("\n" + "=" * 60)
    print("OOD Testing Results")
    print("=" * 60)

    for scenario_name, scenario_results in ood_results.items():
        print(f"\nScenario: {scenario_name}")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = scenario_results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Compare with in-distribution results
    print(f"\nComparison with in-distribution:")
    print(f"{'Scenario':<20} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10}")
    print("-" * 60)

    for scenario_name, scenario_results in ood_results.items():
        print(f"{scenario_name:<20} ", end="")
        for h in [50, 100, 200, 500]:
            print(f"{scenario_results[h]['nmae_mean']:<10.4f} ", end="")
        print()

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'ood_testing',
        'scenarios': [s['name'] for s in ood_scenarios],
        'results': ood_results,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP026_ood_testing.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

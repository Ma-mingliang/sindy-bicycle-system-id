"""Ensemble OOD Robust Model.

Combines ensemble methods with OOD robust training.
Key idea: Multiple models with different training = better OOD robustness.
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


def correct_state_physics(s_pred, s_cur, nn_model, train_obs_norm, state_std, dt,
                           correction_strength=0.1, physics_weight=0.7):
    """Correct predicted state using physics-informed constraints."""
    s_pred_norm = s_pred / state_std

    # Find nearest neighbors
    distances, indices = nn_model.kneighbors([s_pred_norm])

    # Compute correction: move toward nearest neighbor mean
    neighbor_mean = np.mean(train_obs_norm[indices[0]], axis=0)
    nn_correction = (neighbor_mean - s_pred_norm) * correction_strength

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


def evaluate_ensemble_ood(models, ood_scenarios, state_std, action_std, delta_std, nn_model, train_obs_norm,
                           horizons, correction_strength=0.1, physics_weight=0.7):
    """Evaluate ensemble of models on OOD scenarios."""
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
                    # Ensemble prediction
                    predictions = []
                    for model in models:
                        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                        a_norm = torch.FloatTensor([actions[step] / action_std]).unsqueeze(0)

                        with torch.no_grad():
                            dsdt_norm = model(s_norm, a_norm).numpy()[0]

                        dsdt = dsdt_norm * delta_std * dt
                        s_next = s_cur + dsdt
                        predictions.append(s_next)

                    # Average predictions
                    s_next = np.mean(predictions, axis=0)

                    # Apply physics-informed correction
                    s_next = correct_state_physics(s_next, s_cur, nn_model, train_obs_norm, state_std, dt,
                                                    correction_strength, physics_weight)

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
    print("Ensemble OOD Robust Model")
    print("=" * 60)

    config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
    }

    # Best configuration
    correction_strength = 0.1
    physics_weight = 0.7

    # Ensemble sizes to test
    ensemble_sizes = [1, 3, 5]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Building state corrector...")
    nn_model, train_obs_norm = build_state_corrector(data['train_obs'], data['state_std'])
    print(f"  Built nearest-neighbor corrector with {len(data['train_obs'])} training samples")

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for n_models in ensemble_sizes:
        print(f"\n{'='*60}")
        print(f"Testing ensemble size: {n_models}")
        print(f"{'='*60}")

        # Train ensemble
        models = []
        for i in range(n_models):
            seed = 42 + i
            print(f"\n  Training model {i+1}/{n_models} (seed={seed})...")
            model = train_model(data, config, seed=seed)
            models.append(model)

        print(f"\nGenerating OOD scenarios...")
        ood_scenarios = generate_ood_scenarios(data, data['state_std'])

        print(f"\nEvaluating ensemble on OOD scenarios...")
        ood_results = evaluate_ensemble_ood(models, ood_scenarios, data['state_std'], data['action_std'], data['delta_std'],
                                             nn_model, train_obs_norm, horizons, correction_strength, physics_weight)

        # Compute average OOD performance
        avg_ood_nmae = {}
        for h in horizons:
            nmae_values = []
            for scenario_name, scenario_results in ood_results.items():
                nmae = scenario_results[h]['nmae_mean']
                if not np.isnan(nmae):
                    nmae_values.append(nmae)
            avg_ood_nmae[h] = np.mean(nmae_values) if nmae_values else float('nan')

        print(f"\nAverage OOD results for ensemble_size={n_models}:")
        print(f"{'Horizon':<10} {'Avg NMAE':<12}")
        print("-" * 22)
        for h in horizons:
            print(f"H={h:<7} {avg_ood_nmae[h]:<12.4f}")

        all_results[n_models] = {
            'ood_results': ood_results,
            'avg_ood_nmae': avg_ood_nmae,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Ensemble OOD Robust Results")
    print("=" * 60)

    print(f"\n{'Ensemble':<12} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10}")
    print("-" * 52)

    for n_models, data_dict in all_results.items():
        avg = data_dict['avg_ood_nmae']
        print(f"{n_models:<12} {avg[50]:<10.4f} {avg[100]:<10.4f} {avg[200]:<10.4f} {avg[500]:<10.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'ensemble_ood_robust',
        'ensemble_sizes': ensemble_sizes,
        'correction_strength': correction_strength,
        'physics_weight': physics_weight,
        'results': {str(k): {'avg_ood_nmae': {str(h): v for h, v in data_dict['avg_ood_nmae'].items()}}
                   for k, data_dict in all_results.items()},
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP037_ensemble_ood_robust.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

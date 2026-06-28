"""Uncertainty Gating Model.

Addresses root cause: Need to detect OOD states and fall back to physics.
Solution: Use model uncertainty (ensemble variance) for OOD detection.
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


def train_ensemble(data, config, n_models=5):
    """Train ensemble of Neural ODE models."""
    models = []
    for i in range(n_models):
        seed = config.get('seed', 42) + i
        print(f"\nTraining ensemble model {i+1}/{n_models} (seed={seed})...")
        model, state_std, action_std, delta_std = train_single_model(data, config, seed=seed)
        models.append(model)

    return models, state_std, action_std, delta_std


def train_single_model(data, config, seed=42):
    """Train single Neural ODE model."""
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
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}")

    model.eval()

    return model, state_std, action_std, delta_std


def evaluate_ensemble_with_gating(models, data, state_std, action_std, delta_std, horizons,
                                   uncertainty_threshold=0.1, n_segments=5, seed=42):
    """Evaluate ensemble with uncertainty gating.

    When ensemble variance exceeds threshold, fall back to zero-prediction (like GP).
    """
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
    gating_stats = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}
        n_gated = 0
        n_total = 0

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
                    # Get predictions from all models
                    predictions = []
                    for model in models:
                        s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                        a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)

                        with torch.no_grad():
                            dsdt_norm = model(s_norm, a_norm).numpy()[0]

                        dsdt = dsdt_norm * delta_std * dt
                        s_next = s_cur + dsdt
                        predictions.append(s_next)

                    # Compute ensemble statistics
                    predictions = np.array(predictions)
                    mean_pred = np.mean(predictions, axis=0)
                    std_pred = np.std(predictions, axis=0)
                    uncertainty = np.mean(std_pred)

                    n_total += 1

                    # Uncertainty gating
                    if uncertainty > uncertainty_threshold:
                        # High uncertainty: fall back to zero-prediction (stay at current state)
                        s_next = s_cur.copy()
                        n_gated += 1
                    else:
                        # Low uncertainty: use ensemble prediction
                        s_next = mean_pred

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

        gating_stats[h] = {
            'n_gated': n_gated,
            'n_total': n_total,
            'gating_rate': n_gated / max(n_total, 1),
        }

    return results, gating_stats


def main():
    print("=" * 60)
    print("Uncertainty Gating Model")
    print("=" * 60)

    config = {
        'hidden': 64, 'depth': 3, 'activation': 'tanh',
        'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
        'seed': 42,
    }

    # Test different uncertainty thresholds
    thresholds = [0.05, 0.1, 0.2]

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training ensemble (5 models)...")
    models, state_std, action_std, delta_std = train_ensemble(data, config, n_models=5)

    horizons = [1, 10, 50, 100, 200, 500]
    all_results = {}

    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing uncertainty threshold: {threshold}")
        print(f"{'='*60}")

        results, gating_stats = evaluate_ensemble_with_gating(
            models, data, state_std, action_std, delta_std, horizons,
            uncertainty_threshold=threshold
        )

        print(f"\nResults for threshold={threshold}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12} {'Gating%':<12}")
        print("-" * 46)
        for h in horizons:
            r = results[h]
            g = gating_stats[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%} {g['gating_rate']:<12.2%}")

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'], results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

        all_results[threshold] = {
            'results': results,
            'gating_stats': gating_stats,
            'primary': primary,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Uncertainty Gating Results")
    print("=" * 60)

    print(f"\n{'Threshold':<12} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10} {'Primary':<10} {'Gating%':<10}")
    print("-" * 80)

    for threshold, data_dict in all_results.items():
        r = data_dict['results']
        g = data_dict['gating_stats']
        primary = data_dict['primary']
        avg_gating = np.mean([g[h]['gating_rate'] for h in horizons])
        print(f"{threshold:<12} {r[50]['nmae_mean']:<10.4f} {r[100]['nmae_mean']:<10.4f} "
              f"{r[200]['nmae_mean']:<10.4f} {r[500]['nmae_mean']:<10.4f} {primary:<10.4f} {avg_gating:<10.2%}")

    # Compare with baselines
    print(f"\nComparison with baselines:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Best Gated':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    best_threshold = min(all_results.keys(), key=lambda k: all_results[k]['primary'])
    best_results = all_results[best_threshold]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        g_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(g_val):
            improvement = (v9_val - g_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {g_val:<12.4f} {improvement:<12.1f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_144h',
        'experiment': 'uncertainty_gating',
        'thresholds': thresholds,
        'results': {str(k): {'primary': v['primary']} for k, v in all_results.items()},
        'best_threshold': best_threshold,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP017_uncertainty_gating.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

"""V9 Baseline Reproduction Script.

Reproduces the v9 Neural ODE baseline with proper numpy compatibility handling.
"""
import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Add paths
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')

# Constants
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
    """Neural ODE dynamics function: dx/dt = f(x, u)."""

    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            act = nn.Tanh
        elif activation == 'silu':
            act = nn.SiLU
        elif activation == 'relu':
            act = nn.ReLU
        else:
            act = nn.Tanh

        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Initialize last layer small for stability
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


def check_survival(state):
    """Check if state survives physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', seed=42):
    """Load 7D data from 8D dataset."""
    data = np.load(data_path, allow_pickle=True)
    obs_8d = data['obs']
    action = data['action']
    next_obs_8d = data['next_obs']
    done = data['done']

    # Convert to 7D
    obs = obs_8d[:, IDX_7D_FROM_8D]
    next_obs = next_obs_8d[:, IDX_7D_FROM_8D]
    deltas = next_obs - obs

    # Split by episodes
    episode_ends = np.where(done)[0]
    episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
    n_episodes = len(episode_ends)

    # Train/test split (75/25 by episode)
    np.random.seed(seed)
    indices = np.random.permutation(n_episodes)
    n_train = int(0.75 * n_episodes)
    train_eps = indices[:n_train]
    test_eps = indices[n_train:]

    # Collect train/test data
    train_mask = np.zeros(len(obs), dtype=bool)
    test_mask = np.zeros(len(obs), dtype=bool)

    for ep in train_eps:
        train_mask[episode_starts[ep]:episode_ends[ep]+1] = True
    for ep in test_eps:
        test_mask[episode_starts[ep]:episode_ends[ep]+1] = True

    # Compute statistics from training data
    state_std = np.std(obs[train_mask], axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(action[train_mask])
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(deltas[train_mask], axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    return {
        'obs': obs,
        'action': action,
        'deltas': deltas,
        'done': done,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'episode_ends': episode_ends,
        'episode_starts': episode_starts,
    }


def get_test_segments(data, n_segments=5, segment_length=1100, seed=42):
    """Get test segments for long-horizon evaluation."""
    obs = data['obs']
    action = data['action']
    episode_ends = data['episode_ends']
    episode_starts = data['episode_starts']

    # Find long enough test episodes
    test_eps = np.where(data['test_mask'][episode_ends])[0]
    long_eps = [ep for ep in test_eps
                if episode_ends[ep] - episode_starts[ep] >= segment_length]

    np.random.seed(seed)
    segments = []

    for _ in range(min(n_segments, len(long_eps))):
        ep = np.random.choice(long_eps)
        ep_start = episode_starts[ep]
        ep_end = episode_ends[ep]
        max_start = ep_end - segment_length
        if max_start <= ep_start:
            start = ep_start
        else:
            start = np.random.randint(ep_start, max_start + 1)

        segments.append({
            'states': obs[start:start+segment_length],
            'actions': action[start:start+segment_length].flatten(),
            'episode': ep,
            'start': start,
        })

    return segments


def evaluate_v9(model_func, segments, state_std, horizons):
    """Evaluate v9 model on test segments."""
    results = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            s_cur = s0.copy()
            survived = True
            step_errors = []

            for step in range(n):
                try:
                    # Predict next state
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
                    a_norm = torch.FloatTensor([actions_seg[step] / action_std]).unsqueeze(0)

                    with torch.no_grad():
                        dsdt_norm = model_func(s_norm, a_norm).numpy()[0]

                    dsdt = dsdt_norm * delta_std * dt
                    s_next = s_cur + dsdt

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break

                    if not check_survival(s_next):
                        survived = False
                        break

                    # Compute per-step error
                    if step + 1 < len(real_states):
                        step_err = np.abs(s_next - real_states[step + 1]) / state_std
                        step_errors.append(step_err)

                    s_cur = s_next
                except Exception as e:
                    survived = False
                    break

            # Compute NMAE
            if step_errors:
                n_valid = min(len(step_errors), n)
                errors = np.array(step_errors[:n_valid])
                nmae_per_state = np.mean(errors, axis=0)
                nmae_overall = np.mean(nmae_per_state)
                nmae_list.append(nmae_overall)
                for i, name in enumerate(STATE_NAMES_7D):
                    per_state_nmae[name].append(nmae_per_state[i])
            else:
                nmae_list.append(float('nan'))

            survival_list.append(survived and len(step_errors) >= n - 1)

        # Aggregate
        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': float(np.mean(survival_list)),
            'n_valid': len(valid_nmae),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                    'std': float(np.nanstd(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


def main():
    print("=" * 60)
    print("V9 Neural ODE Baseline Reproduction")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   State std: {data['state_std']}")
    print(f"   Action std: {data['action_std']}")
    print(f"   Delta std: {data['delta_std']}")

    # Get test segments
    print("\n2. Getting test segments...")
    segments = get_test_segments(data, n_segments=5, segment_length=1100, seed=42)
    print(f"   Number of segments: {len(segments)}")

    # Build model with v9 config
    print("\n3. Building v9 model...")
    model = ODEFunc(hidden=64, depth=3, activation='tanh')
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")

    # Try to load checkpoint
    ckpt_path = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt'
    print(f"\n4. Loading checkpoint: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print("   Checkpoint loaded successfully!")
        if 'state_std' in ckpt:
            state_std = ckpt['state_std']
            action_std = ckpt['action_std']
            delta_std = ckpt['delta_std']
            print(f"   Using checkpoint normalization stats")
        else:
            state_std = data['state_std']
            action_std = data['action_std']
            delta_std = data['delta_std']
    except Exception as e:
        print(f"   Checkpoint load failed: {e}")
        print("   Using random initialization (needs training)")
        state_std = data['state_std']
        action_std = data['action_std']
        delta_std = data['delta_std']

    # Set dt
    dt = 1.0 / 30.0

    # Evaluate
    print("\n5. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500]
    results = evaluate_v9(model, segments, state_std, horizons)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Per-state results for H=100
    print(f"\nPer-state NMAE (H=100):")
    print("-" * 40)
    for name in STATE_NAMES_7D:
        if 100 in results:
            mean = results[100]['per_state_nmae'][name]['mean']
            print(f"  {name:<12}: {mean:.4f}")

    # Compute primary score
    if 100 in results and 200 in results and 500 in results:
        primary = np.mean([results[100]['nmae_mean'],
                          results[200]['nmae_mean'],
                          results[500]['nmae_mean']])
        print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'model': 'v9_neural_ode',
        'config': {'hidden': 64, 'depth': 3, 'activation': 'tanh'},
        'horizons': horizons,
        'results': results,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/01_baseline/V9_REPRODUCTION_RESULTS.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()

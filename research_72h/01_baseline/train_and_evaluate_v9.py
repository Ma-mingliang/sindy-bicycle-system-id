"""V9 Neural ODE Training and Evaluation Script.

Trains the v9 Neural ODE from scratch and evaluates at multiple horizons.
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
    test_ep_indices = np.where(data['test_mask'][episode_ends])[0]
    long_eps = [ep for ep in test_ep_indices
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


def train_v9(data, config, seed=43):
    """Train v9 Neural ODE model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    # Build model
    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    # Prepare training data
    train_mask = data['train_mask']
    train_s = torch.FloatTensor(data['obs'][train_mask] / state_std)
    train_a = torch.FloatTensor(data['action'][train_mask].reshape(-1, 1) / action_std)
    train_dsdot = torch.FloatTensor(data['deltas'][train_mask] / (delta_std * dt))

    ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config['batch_size'], shuffle=True
    )

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs']
    )

    # Parse rollout curriculum
    curriculum = [int(x) for x in config['rollout_curriculum'].split(',')]

    # Training loop
    print(f"Training v9 model (seed={seed})...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        # Determine current rollout steps
        rollout_steps = 1
        for i, threshold in enumerate(curriculum):
            if epoch >= config['n_epochs'] * (i + 1) / (len(curriculum) + 1):
                rollout_steps = threshold

        for sb, ab, yb in loader:
            # Single-step loss
            pred = model(sb, ab)
            loss_single = nn.functional.mse_loss(pred, yb)

            # Multi-step rollout loss
            loss_multi = torch.tensor(0.0)
            if rollout_steps > 1 and len(sb) > rollout_steps + 1:
                n_roll = min(len(sb) - rollout_steps, 64)
                s_cur = sb[:n_roll].clone()
                for step in range(rollout_steps):
                    a_cur = ab[step:step + n_roll]
                    dsdt = model(s_cur, a_cur)
                    s_cur = s_cur + dsdt * dt
                    target = sb[step + 1:step + 1 + n_roll]
                    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
                loss_multi = loss_multi / rollout_steps

            # Consistency loss
            loss_consistency = torch.tensor(0.0)
            if config.get('lambda_consistency', 0) > 0:
                s_next_norm = sb + pred * dt
                theta_pred = s_next_norm[:, 3]
                theta_dot = sb[:, 4]
                theta_gt = sb[:, 3] + theta_dot * dt
                loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

            # Jacobian regularization
            loss_jacobian = torch.tensor(0.0)
            if config.get('lambda_jacobian', 0) > 0:
                s_req = sb[:min(32, len(sb))].requires_grad_(True)
                a_req = ab[:min(32, len(sb))].requires_grad_(True)
                dsdt = model(s_req, a_req)
                jac_norm = 0.0
                for i in range(STATE_DIM):
                    grad = torch.autograd.grad(
                        dsdt[:, i].sum(), s_req, create_graph=True
                    )[0]
                    jac_norm = jac_norm + grad.pow(2).sum()
                loss_jacobian = jac_norm / (STATE_DIM * min(32, len(sb)))

            loss = (loss_single
                    + config.get('lambda_multi', 0.3) * loss_multi
                    + config.get('lambda_consistency', 0.1) * loss_consistency
                    + config.get('lambda_jacobian', 0.01) * loss_jacobian)

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
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"rollout={rollout_steps}, time={elapsed:.1f}s")

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


def evaluate_v9(model, segments, state_std, action_std, delta_std, horizons):
    """Evaluate v9 model on test segments."""
    dt = 1.0 / 30.0
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
                        dsdt_norm = model(s_norm, a_norm).numpy()[0]

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
    print("V9 Neural ODE Training and Evaluation")
    print("=" * 60)

    # V9 best config
    config = {
        'hidden': 64,
        'depth': 3,
        'activation': 'tanh',
        'lr': 1e-3,
        'weight_decay': 0,
        'n_epochs': 200,
        'batch_size': 256,
        'rollout_curriculum': '1,5,10,20',
        'lambda_multi': 0.3,
        'lambda_consistency': 0.1,
        'lambda_jacobian': 0.01,
    }

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

    # Train model with multiple seeds
    seeds = [42, 43, 44]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed={seed}")
        print(f"{'='*60}")

        model, state_std, action_std, delta_std = train_v9(data, config, seed=seed)

        # Evaluate
        print(f"\nEvaluating seed={seed}...")
        horizons = [1, 10, 50, 100, 200, 500]
        results = evaluate_v9(model, segments, state_std, action_std, delta_std, horizons)

        # Print results
        print(f"\nResults for seed={seed}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        all_results[seed] = results

        # Save model
        model_path = f'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/v9_seed{seed}.pt'
        torch.save({
            'config': config,
            'state_std': state_std,
            'action_std': action_std,
            'delta_std': delta_std,
            'model_state': model.state_dict(),
            'seed': seed,
        }, model_path)
        print(f"Model saved to: {model_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Seed':<8} {'H=1':<10} {'H=10':<10} {'H=50':<10} {'H=100':<10} {'H=200':<10} {'H=500':<10}")
    print("-" * 68)
    for seed in seeds:
        r = all_results[seed]
        print(f"{seed:<8} ", end="")
        for h in horizons:
            print(f"{r[h]['nmae_mean']:<10.4f} ", end="")
        print()

    # Compute primary score
    print(f"\nPrimaryLongHorizonScore (mean of H=100,200,500):")
    for seed in seeds:
        r = all_results[seed]
        primary = np.mean([r[100]['nmae_mean'], r[200]['nmae_mean'], r[500]['nmae_mean']])
        print(f"  Seed {seed}: {primary:.4f}")

    # Save all results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'model': 'v9_neural_ode',
        'config': config,
        'seeds': seeds,
        'horizons': horizons,
        'results': {str(seed): all_results[seed] for seed in seeds},
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/01_baseline/V9_REPRODUCTION_RESULTS.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

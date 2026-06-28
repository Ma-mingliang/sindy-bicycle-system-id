"""V9 Neural ODE Training - BUG FIXED VERSION.

Fixes:
- CRITICAL-1: Multi-step rollout loss now uses contiguous trajectory segments
- CRITICAL-2: Consistency loss computed in physical space
- HIGH-3: Added validation-based early stopping
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
    """Load 7D data from 8D dataset with episode structure preserved."""
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

    # Build episode list
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

    # Compute statistics from training episodes
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


def sample_contiguous_windows(episodes, ep_indices, window_size, batch_size, state_std, action_std, delta_std, dt):
    """Sample contiguous trajectory windows for multi-step training."""
    states_list = []
    actions_list = []
    deltas_list = []

    attempts = 0
    max_attempts = batch_size * 10

    while len(states_list) < batch_size and attempts < max_attempts:
        attempts += 1
        # Pick a random episode
        ep_idx = np.random.choice(ep_indices)
        ep = episodes[ep_idx]
        ep_len = ep['length']

        if ep_len < window_size + 1:
            continue

        # Pick a random start
        start = np.random.randint(0, ep_len - window_size)
        end = start + window_size

        states_list.append(ep['obs'][start:end])
        actions_list.append(ep['action'][start:end])
        deltas_list.append(ep['deltas'][start:end])

    if len(states_list) == 0:
        return None, None, None

    # Pad to batch_size if needed
    while len(states_list) < batch_size:
        states_list.append(states_list[-1].copy())
        actions_list.append(actions_list[-1].copy())
        deltas_list.append(deltas_list[-1].copy())

    states = np.array(states_list[:batch_size])
    actions = np.array(actions_list[:batch_size])
    deltas = np.array(deltas_list[:batch_size])

    # Normalize
    states_t = torch.FloatTensor(states / state_std)
    actions_t = torch.FloatTensor(actions.reshape(batch_size, -1, 1) / action_std)
    deltas_t = torch.FloatTensor(deltas / (delta_std * dt))

    return states_t, actions_t, deltas_t


def train_v9_fixed(data, config, seed=43):
    """Train v9 Neural ODE model with BUG FIXES."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    # Split train into train/val (90/10)
    n_train = len(train_eps)
    n_val = max(1, n_train // 10)
    val_eps = train_eps[:n_val]
    actual_train_eps = train_eps[n_val:]

    # Build model
    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
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
    print(f"Training v9 FIXED model (seed={seed})...")
    start_time = time.time()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    patience_counter = 0

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        # Determine current rollout steps
        rollout_steps = 1
        for i, threshold in enumerate(curriculum):
            if epoch >= config['n_epochs'] * (i + 1) / (len(curriculum) + 1):
                rollout_steps = threshold

        # Training batches
        n_batches_per_epoch = 100
        for batch_idx in range(n_batches_per_epoch):
            # FIX CRITICAL-1: Sample contiguous windows for multi-step
            window_size = max(rollout_steps + 1, 2)
            states_t, actions_t, deltas_t = sample_contiguous_windows(
                episodes, actual_train_eps, window_size,
                config['batch_size'], state_std, action_std, delta_std, dt
            )

            if states_t is None:
                continue

            # Single-step loss (use first step of each window)
            sb = states_t[:, 0, :]  # (batch, 7)
            ab = actions_t[:, 0, :]  # (batch, 1)
            yb = deltas_t[:, 0, :]  # (batch, 7)

            pred = model(sb, ab)
            loss_single = nn.functional.mse_loss(pred, yb)

            # Multi-step rollout loss (FIX: now uses contiguous data)
            loss_multi = torch.tensor(0.0)
            if rollout_steps > 1 and window_size > rollout_steps:
                n_roll = min(config['batch_size'], len(states_t))
                s_cur = states_t[:n_roll, 0, :].clone()  # Starting states

                for step in range(rollout_steps):
                    a_cur = actions_t[:n_roll, step, :]
                    dsdt = model(s_cur, a_cur)
                    s_cur = s_cur + dsdt * dt
                    target = states_t[:n_roll, step + 1, :]
                    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
                loss_multi = loss_multi / rollout_steps

            # FIX CRITICAL-2: Consistency loss in physical space
            loss_consistency = torch.tensor(0.0)
            if config.get('lambda_consistency', 0) > 0:
                # Compute in physical space
                state_std_t = torch.FloatTensor(state_std)
                delta_std_t = torch.FloatTensor(delta_std)
                s_next_phys = sb * state_std_t + pred * delta_std_t * dt
                theta_pred_phys = s_next_phys[:, 3]
                theta_gt_phys = sb[:, 3] * state_std[3] + sb[:, 4] * state_std[4] * dt
                loss_consistency = nn.functional.mse_loss(
                    theta_pred_phys / state_std[3],
                    theta_gt_phys / state_std[3]
                )

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

        # Validation (FIX HIGH-3: add validation-based early stopping)
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            for _ in range(20):  # 20 validation batches
                states_t, actions_t, deltas_t = sample_contiguous_windows(
                    episodes, val_eps, 2,
                    config['batch_size'], state_std, action_std, delta_std, dt
                )
                if states_t is None:
                    continue

                sb = states_t[:, 0, :]
                ab = actions_t[:, 0, :]
                yb = deltas_t[:, 0, :]

                with torch.no_grad():
                    pred = model(sb, ab)
                    val_loss += nn.functional.mse_loss(pred, yb).item()
                n_val_batches += 1

            val_loss = val_loss / max(n_val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

            model.train()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, rollout={rollout_steps}, time={elapsed:.1f}s")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


def evaluate_v9(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate v9 model on test segments."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']

    # Get test segments
    np.random.seed(seed)
    segment_length = 1100
    segments = []

    for ep_idx in test_eps:
        ep = episodes[ep_idx]
        if ep['length'] >= segment_length:
            segments.append(ep)

    if len(segments) == 0:
        print("Warning: No test segments long enough!")
        return {}

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
                except Exception as e:
                    survived = False
                    break

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
    print("V9 Neural ODE Training - BUG FIXED VERSION")
    print("=" * 60)
    print("\nFixes applied:")
    print("  CRITICAL-1: Multi-step rollout now uses contiguous windows")
    print("  CRITICAL-2: Consistency loss computed in physical space")
    print("  HIGH-3: Added validation-based early stopping")

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
    print(f"   Train episodes: {len(data['train_eps'])}")
    print(f"   Test episodes: {len(data['test_eps'])}")

    # Train with multiple seeds
    seeds = [42, 43, 44]
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed={seed}")
        print(f"{'='*60}")

        model, state_std, action_std, delta_std = train_v9_fixed(data, config, seed=seed)

        # Evaluate
        print(f"\nEvaluating seed={seed}...")
        horizons = [1, 10, 50, 100, 200, 500]
        results = evaluate_v9(model, data, state_std, action_std, delta_std, horizons)

        # Print results
        print(f"\nResults for seed={seed}:")
        print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
        print("-" * 34)
        for h in horizons:
            r = results[h]
            print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

        all_results[seed] = results

        # Save model
        model_path = f'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/v9_fixed_seed{seed}.pt'
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
        'model': 'v9_neural_ode_fixed',
        'config': config,
        'seeds': seeds,
        'horizons': horizons,
        'bugfixes': [
            'CRITICAL-1: Multi-step rollout now uses contiguous windows',
            'CRITICAL-2: Consistency loss computed in physical space',
            'HIGH-3: Added validation-based early stopping',
        ],
        'results': {str(seed): all_results[seed] for seed in seeds},
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/01_baseline/V9_FIXED_REPRODUCTION_RESULTS.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

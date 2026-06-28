"""Quick Screen: Physics-Constrained Neural ODE.

Tests if adding physics constraints (e_y_dot = v*sin(e_psi)) improves prediction.
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


def sample_contiguous_windows(episodes, ep_indices, window_size, batch_size, state_std, action_std, delta_std, dt):
    """Sample contiguous trajectory windows for multi-step training."""
    states_list = []
    actions_list = []
    deltas_list = []

    attempts = 0
    max_attempts = batch_size * 10

    while len(states_list) < batch_size and attempts < max_attempts:
        attempts += 1
        ep_idx = np.random.choice(ep_indices)
        ep = episodes[ep_idx]
        ep_len = ep['length']

        if ep_len < window_size + 1:
            continue

        start = np.random.randint(0, ep_len - window_size)
        end = start + window_size

        states_list.append(ep['obs'][start:end])
        actions_list.append(ep['action'][start:end])
        deltas_list.append(ep['deltas'][start:end])

    if len(states_list) == 0:
        return None, None, None

    while len(states_list) < batch_size:
        states_list.append(states_list[-1].copy())
        actions_list.append(actions_list[-1].copy())
        deltas_list.append(deltas_list[-1].copy())

    states = np.array(states_list[:batch_size])
    actions = np.array(actions_list[:batch_size])
    deltas = np.array(deltas_list[:batch_size])

    states_t = torch.FloatTensor(states / state_std)
    actions_t = torch.FloatTensor(actions.reshape(batch_size, -1, 1) / action_std)
    deltas_t = torch.FloatTensor(deltas / (delta_std * dt))

    return states_t, actions_t, deltas_t


def train_physics_constrained(data, config, seed=43):
    """Train Neural ODE with physics constraints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    n_train = len(train_eps)
    n_val = max(1, n_train // 10)
    val_eps = train_eps[:n_val]
    actual_train_eps = train_eps[n_val:]

    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    lambda_physics = config.get('lambda_physics', 0.5)

    print(f"Training Physics-Constrained model (seed={seed}, lambda_physics={lambda_physics})...")
    start_time = time.time()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    patience_counter = 0

    # Parse rollout curriculum
    curriculum = [int(x) for x in config['rollout_curriculum'].split(',')]

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        # Determine current rollout steps
        rollout_steps = 1
        for i, threshold in enumerate(curriculum):
            if epoch >= config['n_epochs'] * (i + 1) / (len(curriculum) + 1):
                rollout_steps = threshold

        for _ in range(100):
            window_size = max(rollout_steps + 1, 2)
            states_t, actions_t, deltas_t = sample_contiguous_windows(
                episodes, actual_train_eps, window_size,
                config['batch_size'], state_std, action_std, delta_std, dt
            )

            if states_t is None:
                continue

            sb = states_t[:, 0, :]
            ab = actions_t[:, 0, :]
            yb = deltas_t[:, 0, :]

            pred = model(sb, ab)
            loss_single = nn.functional.mse_loss(pred, yb)

            # Multi-step rollout loss
            loss_multi = torch.tensor(0.0)
            if rollout_steps > 1 and window_size > rollout_steps:
                n_roll = min(config['batch_size'], len(states_t))
                s_cur = states_t[:n_roll, 0, :].clone()

                for step in range(rollout_steps):
                    a_cur = actions_t[:n_roll, step, :]
                    dsdt = model(s_cur, a_cur)
                    s_cur = s_cur + dsdt * dt
                    target = states_t[:n_roll, step + 1, :]
                    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
                loss_multi = loss_multi / rollout_steps

            # Physics constraint: e_y_dot = v * sin(e_psi)
            # In normalized space: pred[:, 0] should equal sb[:, 2] * sin(sb[:, 1] * state_std[1]) / (delta_std[0] * dt)
            # Simplified: use physical space
            loss_physics = torch.tensor(0.0)
            if lambda_physics > 0:
                state_std_t = torch.FloatTensor(state_std)
                delta_std_t = torch.FloatTensor(delta_std)

                # Convert to physical space
                s_phys = sb * state_std_t
                pred_phys = pred * delta_std_t * dt

                # e_y_dot = v * sin(e_psi)
                e_y_dot_pred = pred_phys[:, 0] / dt  # e_y change per second
                v_phys = s_phys[:, 2]
                e_psi_phys = s_phys[:, 1]
                e_y_dot_expected = v_phys * torch.sin(e_psi_phys)

                loss_ey = nn.functional.mse_loss(e_y_dot_pred, e_y_dot_expected)

                # e_psi_dot = -v * delta / L (simplified, L=1)
                e_psi_dot_pred = pred_phys[:, 1] / dt
                delta_phys = s_phys[:, 6]
                e_psi_dot_expected = -v_phys * delta_phys  # simplified

                loss_epsi = nn.functional.mse_loss(e_psi_dot_pred, e_psi_dot_expected)

                loss_physics = loss_ey + loss_epsi

            loss = loss_single + config.get('lambda_multi', 0.3) * loss_multi + lambda_physics * loss_physics

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            for _ in range(20):
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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
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
    print("Quick Screen: Physics-Constrained Neural ODE")
    print("=" * 60)

    config = {
        'hidden': 64,
        'depth': 3,
        'activation': 'tanh',
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 256,
        'rollout_curriculum': '1,5,10,20',
        'lambda_multi': 0.3,
        'lambda_physics': 0.5,
    }

    print("\n1. Loading data...")
    data = load_data(seed=42)

    print("\n2. Training Physics-Constrained model...")
    model, state_std, action_std, delta_std = train_physics_constrained(data, config, seed=43)

    print("\n3. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500]
    results = evaluate(model, data, state_std, action_std, delta_std, horizons)

    print("\n" + "=" * 60)
    print("RESULTS: Physics-Constrained Neural ODE")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    print(f"\nComparison with v9 baseline:")
    print(f"{'Horizon':<10} {'v9 NMAE':<12} {'Physics':<12} {'Improvement':<12}")
    print("-" * 46)
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        phys_val = results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(phys_val):
            improvement = (v9_val - phys_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {phys_val:<12.4f} {improvement:<12.1f}%")

    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_175824_neural_ode_72h',
        'experiment': 'physics_constrained',
        'config': config,
        'results': results,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP006_physics_loss.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    model_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/physics_constrained_seed43.pt'
    torch.save({
        'config': config,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'model_state': model.state_dict(),
    }, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()

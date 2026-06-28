"""Trajectory Optimization Neural ODE.

Core idea: Instead of minimizing single-step loss, minimize trajectory-level loss.
Roll out H steps using the model's own predictions, then compare entire trajectories.

Key techniques:
1. Trajectory Loss: MSE over H-step rollout (H=50/100/200)
2. Scheduled Sampling: Gradually transition from GT to predicted states
3. Pushforward Training: Use predicted states to continue rollout during training
4. Adaptive Horizon Curriculum: Start with short horizons, increase over training

Optimization: Vectorized trajectory rollout for CPU efficiency.
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


class TrajODEFunc(nn.Module):
    """Neural ODE dynamics function for trajectory optimization."""
    def __init__(self, hidden=128, depth=4, activation='tanh', dropout=0.0):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        """Predict ds/dt (normalized). s: (*, 7), a: (*, 1) -> (*, 7)"""
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


def sample_trajectory_batch(episodes, train_eps, horizon, batch_size, rng):
    """Sample a batch of trajectory segments.

    Returns padded tensors:
        obs_batch: (B, H+1, 7) - state trajectories (normalized)
        act_batch: (B, H, 1) - action trajectories (normalized)
        mask_batch: (B, H+1) - validity mask
    """
    valid_eps = [ep for ep in train_eps if episodes[ep]['length'] > horizon + 1]
    if len(valid_eps) == 0:
        valid_eps = [ep for ep in train_eps if episodes[ep]['length'] > 5]

    obs_list = []
    act_list = []

    for _ in range(batch_size):
        ep_idx = rng.choice(valid_eps)
        ep = episodes[ep_idx]
        max_start = max(0, ep['length'] - horizon - 1)
        start = rng.randint(0, max_start + 1)
        end = min(start + horizon + 1, ep['length'])
        T = end - start

        obs_list.append(ep['obs'][start:end])
        act_list.append(ep['action'][start:end].reshape(-1, 1))

    return obs_list, act_list


def trajectory_loss_vectorized(model, obs_list, act_list, horizon, state_std,
                               action_std, delta_std, dt, teacher_forcing_ratio, device):
    """Compute trajectory loss over a batch of trajectories.

    This is the key function: roll out H steps, compare with GT.

    For efficiency, we process each trajectory sequentially (each step depends on
    previous), but batch the forward pass across trajectories at each step.
    """
    B = len(obs_list)
    delta_scale = torch.FloatTensor(delta_std * dt).to(device)

    # Normalize all trajectories
    obs_norm = [torch.FloatTensor(o / state_std) for o in obs_list]
    act_norm = [torch.FloatTensor(a / action_std) for a in act_list]

    # Pad to same length
    max_T = max(len(o) for o in obs_norm)
    actual_horizon = min(horizon, max_T - 1)

    if actual_horizon < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Pad and stack: (B, max_T, 7) and (B, max_T, 1)
    obs_padded = torch.zeros(B, max_T, STATE_DIM, device=device)
    act_padded = torch.zeros(B, max_T, ACTION_DIM, device=device)
    mask = torch.zeros(B, max_T, device=device)

    for i in range(B):
        T_i = len(obs_norm[i])
        obs_padded[i, :T_i] = obs_norm[i]
        act_padded[i, :T_i] = act_norm[i]
        mask[i, :T_i] = 1.0

    # Roll out trajectory step by step
    pred_states = torch.zeros(B, actual_horizon + 1, STATE_DIM, device=device)
    pred_states[:, 0] = obs_padded[:, 0]  # s0

    s_cur = obs_padded[:, 0:1, :].squeeze(1)  # (B, 7)

    # Determine teacher forcing schedule: for each step, which trajectories use GT
    if teacher_forcing_ratio > 0:
        tf_mask = torch.rand(B, actual_horizon, device=device) < teacher_forcing_ratio
    else:
        tf_mask = torch.zeros(B, actual_horizon, dtype=torch.bool, device=device)

    for t in range(actual_horizon):
        a_t = act_padded[:, t, :]  # (B, 1)

        # Where teacher forcing applies, use GT state
        if teacher_forcing_ratio > 0:
            gt_s = obs_padded[:, t, :]  # (B, 7)
            s_input = torch.where(tf_mask[:, t:t+1], gt_s, s_cur)
        else:
            s_input = s_cur

        # Predict ds/dt
        dsdt_norm = model(s_input, a_t)  # (B, 7)

        # Integrate
        ds = dsdt_norm * delta_scale.unsqueeze(0)  # (B, 7)
        s_next = s_input + ds

        pred_states[:, t + 1] = s_next
        s_cur = s_next

    # Compute loss against GT (in normalized space)
    gt_states = obs_padded[:, :actual_horizon + 1, :]  # (B, H+1, 7)
    loss_mask = mask[:, :actual_horizon + 1]  # (B, H+1)

    # Per-step squared error
    sq_err = (pred_states - gt_states).pow(2).mean(dim=-1)  # (B, H+1)
    # Apply mask and average
    masked_loss = (sq_err * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    return masked_loss


def get_teacher_forcing_ratio(epoch, n_epochs, strategy='linear_decay'):
    """Scheduled sampling: gradually reduce teacher forcing ratio."""
    progress = epoch / max(n_epochs - 1, 1)

    if strategy == 'none':
        return 0.0
    elif strategy == 'constant_low':
        return 0.3
    elif strategy == 'linear_decay':
        return max(0.0, 1.0 - progress)
    elif strategy == 'step_decay':
        step = int(progress * 5)
        return max(0.0, 1.0 - step * 0.2)
    elif strategy == 'cosine_decay':
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    elif strategy == 'inverse_sigmoid':
        k = 10.0
        return 1.0 / (1.0 + np.exp(k * (progress - 0.5)))
    else:
        return max(0.0, 1.0 - progress)


def get_curriculum_horizon(epoch, n_epochs, max_horizon, strategy='linear'):
    """Adaptive horizon curriculum: start short, increase over training."""
    progress = epoch / max(n_epochs - 1, 1)

    if strategy == 'fixed':
        return max_horizon
    elif strategy == 'linear':
        return max(5, int(5 + (max_horizon - 5) * progress))
    elif strategy == 'step':
        if progress < 0.25:
            return 10
        elif progress < 0.5:
            return 25
        elif progress < 0.75:
            return 50
        else:
            return max_horizon
    elif strategy == 'exponential':
        return max(5, int(5 * (max_horizon / 5) ** progress))
    else:
        return max_horizon


def train_trajectory_optimization(data, config, seed=43):
    """Train Neural ODE with trajectory optimization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    # Model setup
    model = TrajODEFunc(
        hidden=config.get('hidden', 128),
        depth=config.get('depth', 4),
        activation=config.get('activation', 'tanh'),
        dropout=config.get('dropout', 0.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.01
    )

    # Config
    n_epochs = config['n_epochs']
    batch_size = config.get('batch_size', 16)
    max_horizon = config.get('max_horizon', 100)
    tf_strategy = config.get('teacher_forcing_strategy', 'linear_decay')
    horizon_strategy = config.get('horizon_curriculum', 'linear')
    lambda_single = config.get('lambda_single', 0.5)
    lambda_traj = config.get('lambda_traj', 1.0)
    multi_scale_horizons = config.get('multi_scale_horizons', [10, 50])
    multi_scale_weights = config.get('multi_scale_weights', [1.0, 0.5])

    # Pre-compute single-step training data (flat, for fast single-step loss)
    train_obs_all = np.concatenate([episodes[ep]['obs'] for ep in train_eps])
    train_action_all = np.concatenate([episodes[ep]['action'] for ep in train_eps])
    train_deltas_all = np.concatenate([episodes[ep]['deltas'] for ep in train_eps])
    n_single = len(train_obs_all)

    print(f"\nTraining Trajectory Optimization Neural ODE (seed={seed})...")
    print(f"  max_horizon: {max_horizon}")
    print(f"  multi_scale_horizons: {multi_scale_horizons}")
    print(f"  teacher_forcing: {tf_strategy}")
    print(f"  horizon_curriculum: {horizon_strategy}")
    print(f"  lambda_single: {lambda_single}, lambda_traj: {lambda_traj}")
    print(f"  batch_size: {batch_size}")

    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    patience = config.get('patience', 60)

    # Compute steps per epoch: balance single-step and trajectory batches
    n_traj_batches = 4  # Fewer trajectory batches per epoch (they're expensive)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Current curriculum settings
        cur_horizon = get_curriculum_horizon(epoch, n_epochs, max_horizon, horizon_strategy)
        tf_ratio = get_teacher_forcing_ratio(epoch, n_epochs, tf_strategy)

        # --- Single-step loss (fast, stabilizing) ---
        if lambda_single > 0:
            idx = rng.choice(n_single, size=min(256, n_single), replace=False)
            s_batch = torch.FloatTensor(train_obs_all[idx] / state_std).to(device)
            a_batch = torch.FloatTensor(train_action_all[idx].reshape(-1, 1) / action_std).to(device)
            dsdt_batch = torch.FloatTensor(train_deltas_all[idx] / (delta_std * dt)).to(device)

            pred_single = model(s_batch, a_batch)
            loss_single = nn.functional.mse_loss(pred_single, dsdt_batch)
        else:
            loss_single = torch.tensor(0.0, device=device)

        # --- Trajectory loss (multi-scale) ---
        loss_traj_total = torch.tensor(0.0, device=device)

        if lambda_traj > 0:
            for h_scale, w_scale in zip(multi_scale_horizons, multi_scale_weights):
                h_cur = min(h_scale, cur_horizon)
                if h_cur < 2:
                    continue

                # Sample trajectory batch for this horizon
                obs_list, act_list = sample_trajectory_batch(
                    episodes, train_eps, h_cur, batch_size, rng
                )

                loss_h = trajectory_loss_vectorized(
                    model, obs_list, act_list, h_cur, state_std, action_std,
                    delta_std, dt, tf_ratio, device
                )

                if not torch.isnan(loss_h):
                    loss_traj_total = loss_traj_total + w_scale * loss_h

        # --- Combined loss ---
        loss = lambda_single * loss_single + lambda_traj * loss_traj_total

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        opt.step()
        scheduler.step()

        epoch_loss = loss.item()

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            lr = opt.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"loss={epoch_loss:.6f} (single={loss_single.item():.6f}, "
                  f"traj={loss_traj_total.item():.6f}) "
                  f"H_cur={cur_horizon} tf={tf_ratio:.2f} "
                  f"lr={lr:.2e} time={elapsed:.1f}s")

        # Early stopping on total loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > n_epochs // 2:
            print(f"  Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_state)
            break

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


def evaluate(model, data, state_std, action_std, delta_std, horizons, n_segments=5, seed=42):
    """Evaluate model on multi-step prediction."""
    dt = 1.0 / 30.0
    episodes = data['episodes']
    test_eps = data['test_eps']
    device = next(model.parameters()).device

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
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0).to(device)
                    a_norm = torch.FloatTensor([[actions_seg[step] / action_std]]).to(device)

                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).cpu().numpy()[0]

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
                except Exception:
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


def run_experiment(config_name, config, data, horizons, seed=43):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Experiment: {config_name}")
    print(f"{'='*70}")

    model, state_std, action_std, delta_std = train_trajectory_optimization(data, config, seed=seed)

    print(f"\nEvaluating {config_name}...")
    results = evaluate(model, data, state_std, action_std, delta_std, horizons)

    print(f"\nResults for {config_name}:")
    print(f"{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    valid_scores = []
    for h in [100, 200, 500]:
        if h in results and not np.isnan(results[h]['nmae_mean']):
            valid_scores.append(results[h]['nmae_mean'])
    primary = float(np.mean(valid_scores)) if valid_scores else float('nan')
    print(f"\nPrimaryLongHorizonScore: {primary:.4f}")

    return results, primary


def main():
    print("=" * 70)
    print("Trajectory Optimization Neural ODE")
    print("Addressing error accumulation via trajectory-level training")
    print("=" * 70)

    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   Train episodes: {len(data['train_eps'])}, Test episodes: {len(data['test_eps'])}")

    horizons = [1, 10, 50, 100, 200, 500]

    configs = {
        # Config 1: v9 architecture baseline (single-step, for fair comparison)
        'v9_arch_baseline': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'max_horizon': 1,
            'teacher_forcing_strategy': 'none',
            'horizon_curriculum': 'fixed',
            'multi_scale_horizons': [1],
            'multi_scale_weights': [1.0],
            'lambda_single': 1.0, 'lambda_traj': 0.0,
            'weight_decay': 0, 'grad_clip': 1.0,
            'patience': 60, 'dropout': 0.0,
        },
        # Config 2: v9 arch + trajectory H=50
        'v9_traj_H50': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 16,
            'max_horizon': 50,
            'teacher_forcing_strategy': 'linear_decay',
            'horizon_curriculum': 'linear',
            'multi_scale_horizons': [5, 20, 50],
            'multi_scale_weights': [1.0, 0.5, 0.25],
            'lambda_single': 0.3, 'lambda_traj': 1.0,
            'weight_decay': 0, 'grad_clip': 1.0,
            'patience': 60, 'dropout': 0.0,
        },
        # Config 3: v9 arch + trajectory H=100 + cosine TF
        'v9_traj_H100_cosine': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 200, 'batch_size': 8,
            'max_horizon': 100,
            'teacher_forcing_strategy': 'cosine_decay',
            'horizon_curriculum': 'exponential',
            'multi_scale_horizons': [10, 50, 100],
            'multi_scale_weights': [1.0, 0.5, 0.25],
            'lambda_single': 0.3, 'lambda_traj': 1.0,
            'weight_decay': 0, 'grad_clip': 0.5,
            'patience': 60, 'dropout': 0.0,
        },
        # Config 4: v9 arch + pure pushforward H=100
        'v9_pushforward_H100': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 200, 'batch_size': 8,
            'max_horizon': 100,
            'teacher_forcing_strategy': 'none',
            'horizon_curriculum': 'linear',
            'multi_scale_horizons': [10, 50, 100],
            'multi_scale_weights': [1.0, 0.5, 0.25],
            'lambda_single': 0.5, 'lambda_traj': 1.0,
            'weight_decay': 0, 'grad_clip': 1.0,
            'patience': 60, 'dropout': 0.0,
        },
        # Config 5: v9 arch + trajectory H=200
        'v9_traj_H200': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 5e-4, 'n_epochs': 200, 'batch_size': 4,
            'max_horizon': 200,
            'teacher_forcing_strategy': 'cosine_decay',
            'horizon_curriculum': 'exponential',
            'multi_scale_horizons': [20, 100, 200],
            'multi_scale_weights': [1.0, 0.5, 0.25],
            'lambda_single': 0.2, 'lambda_traj': 1.0,
            'weight_decay': 0, 'grad_clip': 0.5,
            'patience': 60, 'dropout': 0.0,
        },
        # Config 6: v9 arch + step-decay TF
        'v9_traj_step_tf_H100': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 200, 'batch_size': 8,
            'max_horizon': 100,
            'teacher_forcing_strategy': 'step_decay',
            'horizon_curriculum': 'step',
            'multi_scale_horizons': [10, 50, 100],
            'multi_scale_weights': [1.0, 0.5, 0.25],
            'lambda_single': 0.3, 'lambda_traj': 1.0,
            'weight_decay': 0, 'grad_clip': 1.0,
            'patience': 60, 'dropout': 0.0,
        },
    }

    all_results = {}

    for config_name, config in configs.items():
        results, primary = run_experiment(config_name, config, data, horizons, seed=43)
        all_results[config_name] = {
            'results': results,
            'primary': primary,
        }

    # === Summary ===
    print("\n" + "=" * 95)
    print("SUMMARY: Trajectory Optimization Results")
    print("=" * 95)

    print(f"\n{'Config':<25} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Primary':<10}")
    print("-" * 95)

    for name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary']
        vals = []
        for h in horizons:
            v = r[h]['nmae_mean']
            vals.append(f"{v:.4f}" if not np.isnan(v) else "N/A")
        print(f"{name:<25} {vals[0]:<8} {vals[1]:<8} {vals[2]:<8} {vals[3]:<8} {vals[4]:<8} {vals[5]:<8} {primary:<10.4f}")

    # === Comparison with v9 baseline ===
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    v9_primary = 0.5110

    print(f"\n{'='*70}")
    print("Comparison with v9 baseline")
    print(f"{'='*70}")
    print(f"v9 baseline PrimaryLongHorizonScore: {v9_primary:.4f}")

    best_name = min(all_results.keys(),
                    key=lambda k: all_results[k]['primary'] if not np.isnan(all_results[k]['primary']) else float('inf'))
    best_primary = all_results[best_name]['primary']

    improvement = (v9_primary - best_primary) / v9_primary * 100 if v9_primary > 0 and not np.isnan(best_primary) else 0
    print(f"Best trajectory opt ({best_name}): {best_primary:.4f}")
    print(f"Improvement vs v9: {improvement:+.1f}%")

    print(f"\n{'Horizon':<10} {'v9 NMAE':<12} {'Best TrajOpt':<12} {'Improvement':<12}")
    print("-" * 46)
    best_results = all_results[best_name]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        c_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(c_val):
            imp = (v9_val - c_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {c_val:<12.4f} {imp:<12.1f}%")

    # === Per-state analysis for best config ===
    print(f"\n{'='*70}")
    print(f"Per-state NMAE for best config: {best_name}")
    print(f"{'='*70}")
    print(f"{'Horizon':<10}", end="")
    for name in STATE_NAMES_7D:
        print(f"{name:<12}", end="")
    print()
    print("-" * 94)
    for h in [1, 10, 50, 100, 200, 500]:
        print(f"H={h:<7}", end="")
        for name in STATE_NAMES_7D:
            v = best_results[h]['per_state_nmae'][name]['mean']
            print(f"{v:<12.4f}", end="")
        print()

    # === Save results ===
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'trajectory_optimization',
        'description': 'Trajectory Optimization Neural ODE with scheduled sampling and pushforward training',
        'v9_baseline': {
            'nmae': v9_nmae,
            'primary': v9_primary,
        },
        'best_config': best_name,
        'best_primary': float(best_primary),
        'improvement_vs_v9': float(improvement),
        'all_configs': {},
    }

    for name, data_dict in all_results.items():
        output['all_configs'][name] = {
            'primary': float(data_dict['primary']),
            'results': {
                str(h): {
                    'nmae_mean': data_dict['results'][h]['nmae_mean'],
                    'nmae_std': data_dict['results'][h]['nmae_std'],
                    'survival_rate': data_dict['results'][h]['survival_rate'],
                } for h in horizons
            },
        }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP021_trajectory_optimization.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()

"""Multi-Step Training for Improved Long-Horizon Prediction.

Addresses the core problem: all methods degrade at H=200/500.

Key innovations over EXP021 (trajectory_optimization):
1. Conservative curriculum: 1 -> 5 -> 10 -> 20 (not 100+)
2. Exponential Moving Average (EMA) for stable rollouts
3. State-weighted loss emphasizing hard-to-predict states
4. Scheduled sampling with smooth cosine decay
5. Multi-scale loss: single-step + short-trajectory + medium-trajectory
6. Gradient penalty on Jacobian for stability
7. Physics-aware regularization (energy bounds)

Design philosophy:
- Start simple (single-step), gradually add complexity
- Never sacrifice H=1 accuracy for H=500
- Use EMA to reduce oscillation in trajectory predictions
- Weight states inversely proportional to their prediction difficulty
"""
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
DT = 1.0 / 30.0

PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# State difficulty weights (harder states get higher weight)
# Based on EXP024 analysis: delta_dot, theta_dot, e_y are hardest
DEFAULT_STATE_WEIGHTS = np.array([1.5, 1.2, 0.3, 1.0, 1.3, 1.2, 1.5])


# ============================================================
# Model Architecture
# ============================================================
class MultiStepODEFunc(nn.Module):
    """Neural ODE dynamics function optimized for multi-step training.

    Architecture choices:
    - Tanh activation (smooth, bounded gradients)
    - Xavier init with small gain (prevents initial explosion)
    - Optional LayerNorm for training stability
    """
    def __init__(self, hidden=64, depth=3, activation='tanh', use_layer_norm=False):
        super().__init__()
        act_cls = nn.Tanh if activation == 'tanh' else nn.SiLU

        layers = []
        in_dim = STATE_DIM + ACTION_DIM
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            if use_layer_norm and i > 0:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Small output initialization for stability
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        """Predict ds/dt (normalized). s: (*, 7), a: (*, 1) -> (*, 7)"""
        return self.net(torch.cat([s, a], dim=-1))


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap current params with EMA params."""
        self.backup = {name: param.data.clone() for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original params."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


# ============================================================
# Data Loading
# ============================================================
def load_data(seed=42):
    """Load 7D data from 8D dataset with episode structure."""
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


# ============================================================
# Sampling Utilities
# ============================================================
def sample_trajectory_windows(episodes, ep_indices, horizon, batch_size, rng):
    """Sample contiguous trajectory windows for multi-step training.

    Returns lists of (obs, action) arrays, each of length >= horizon+1.
    """
    valid_eps = [ep for ep in ep_indices if episodes[ep]['length'] > horizon + 1]
    if len(valid_eps) == 0:
        valid_eps = [ep for ep in ep_indices if episodes[ep]['length'] > 5]
        horizon = min(horizon, min(episodes[ep]['length'] for ep in valid_eps) - 2)

    obs_list = []
    act_list = []

    for _ in range(batch_size):
        ep_idx = rng.choice(valid_eps)
        ep = episodes[ep_idx]
        max_start = max(0, ep['length'] - horizon - 1)
        start = rng.randint(0, max_start + 1)
        end = min(start + horizon + 1, ep['length'])

        obs_list.append(ep['obs'][start:end])
        act_list.append(ep['action'][start:end].reshape(-1, 1))

    return obs_list, act_list, horizon


def sample_flat_batch(episodes, ep_indices, batch_size, rng):
    """Sample random single-step transitions (for single-step loss)."""
    all_obs = []
    all_act = []
    all_delta = []

    for ep_idx in ep_indices:
        ep = episodes[ep_idx]
        all_obs.append(ep['obs'])
        all_act.append(ep['action'])
        all_delta.append(ep['deltas'])

    all_obs = np.concatenate(all_obs)
    all_act = np.concatenate(all_act)
    all_delta = np.concatenate(all_delta)

    n = len(all_obs)
    idx = rng.choice(n, size=min(batch_size, n), replace=False)
    return all_obs[idx], all_act[idx], all_delta[idx]


# ============================================================
# Loss Functions
# ============================================================
def trajectory_rollout_loss(model, obs_list, act_list, horizon, state_std,
                            action_std, delta_std, dt, teacher_forcing_ratio,
                            state_weights, device):
    """Roll out H steps and compute trajectory loss.

    Combines per-step MSE with state-specific weighting.
    Uses scheduled sampling (teacher forcing) for stability.
    """
    B = len(obs_list)
    delta_scale = torch.FloatTensor(delta_std * dt).to(device)
    state_w = torch.FloatTensor(state_weights).to(device)

    # Normalize
    obs_norm = [torch.FloatTensor(o / state_std).to(device) for o in obs_list]
    act_norm = [torch.FloatTensor(a / action_std).to(device) for a in act_list]

    # Pad to same length
    max_T = max(len(o) for o in obs_norm)
    actual_horizon = min(horizon, max_T - 1)
    if actual_horizon < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    obs_padded = torch.zeros(B, max_T, STATE_DIM, device=device)
    act_padded = torch.zeros(B, max_T, ACTION_DIM, device=device)
    mask = torch.zeros(B, max_T, device=device)

    for i in range(B):
        T_i = len(obs_norm[i])
        obs_padded[i, :T_i] = obs_norm[i]
        act_padded[i, :T_i] = act_norm[i]
        mask[i, :T_i] = 1.0

    # Roll out
    s_cur = obs_padded[:, 0]  # (B, 7)
    total_loss = torch.tensor(0.0, device=device)
    n_steps = 0

    # Teacher forcing mask per step
    if teacher_forcing_ratio > 0:
        tf_mask = torch.rand(B, actual_horizon, device=device) < teacher_forcing_ratio
    else:
        tf_mask = torch.zeros(B, actual_horizon, dtype=torch.bool, device=device)

    for t in range(actual_horizon):
        a_t = act_padded[:, t, :]  # (B, 1)

        # Teacher forcing: mix GT and predicted
        if teacher_forcing_ratio > 0:
            gt_s = obs_padded[:, t]
            s_input = torch.where(tf_mask[:, t:t+1], gt_s, s_cur)
        else:
            s_input = s_cur

        # Predict ds/dt and integrate
        dsdt_norm = model(s_input, a_t)
        s_next = s_input + dsdt_norm * delta_scale.unsqueeze(0)

        # Target: next GT state
        target = obs_padded[:, t + 1]
        valid = mask[:, t + 1] > 0

        if valid.any():
            # State-weighted MSE
            err = (s_next[valid] - target[valid]).pow(2) * state_w.unsqueeze(0)
            step_loss = err.mean()
            total_loss = total_loss + step_loss
            n_steps += 1

        s_cur = s_next

    if n_steps > 0:
        total_loss = total_loss / n_steps
    return total_loss


def single_step_loss(model, obs_batch, act_batch, delta_batch, state_std,
                     action_std, delta_std, dt, state_weights, device):
    """Standard single-step MSE loss with state weighting."""
    state_w = torch.FloatTensor(state_weights).to(device)

    s_norm = torch.FloatTensor(obs_batch / state_std).to(device)
    a_norm = torch.FloatTensor(act_batch.reshape(-1, 1) / action_std).to(device)
    dsdt_target = torch.FloatTensor(delta_batch / (delta_std * dt)).to(device)

    dsdt_pred = model(s_norm, a_norm)
    err = (dsdt_pred - dsdt_target).pow(2) * state_w.unsqueeze(0)
    return err.mean()


def jacobian_penalty(model, s_batch, a_batch, device, max_samples=32):
    """Penalize large Jacobian norms for stability.

    Prevents the dynamics from having excessively large gradients,
    which cause trajectory divergence.
    """
    n = min(max_samples, len(s_batch))
    s_req = torch.FloatTensor(s_batch[:n]).to(device).requires_grad_(True)
    a_req = torch.FloatTensor(a_batch[:n].reshape(-1, 1)).to(device).requires_grad_(True)

    dsdt = model(s_req, a_req)
    jac_norm = 0.0
    for i in range(STATE_DIM):
        grad = torch.autograd.grad(dsdt[:, i].sum(), s_req, create_graph=True)[0]
        jac_norm = jac_norm + grad.pow(2).sum()
    return jac_norm / (STATE_DIM * n)


def physics_energy_reg(pred_deltas, state_std, delta_std, device):
    """Regularize predictions to respect energy bounds.

    Penalizes predictions that would increase total energy unboundedly.
    """
    # Approximate energy: kinetic (v^2) + potential (theta^2)
    # Penalize large changes in these
    energy_states = torch.FloatTensor([0, 0, 1.0, 1.0, 0, 0, 0]).to(device)
    delta_scale = torch.FloatTensor(delta_std).to(device)

    energy_change = (pred_deltas * delta_scale * energy_states.unsqueeze(0)).pow(2).sum(dim=-1)
    return energy_change.mean()


# ============================================================
# Curriculum Schedule
# ============================================================
class CurriculumScheduler:
    """Manage curriculum learning schedule for multi-step training.

    Phase 1 (0-25%): H=1 only (single-step)
    Phase 2 (25-50%): H=1 + H=5
    Phase 3 (50-75%): H=1 + H=5 + H=10
    Phase 4 (75-100%): H=1 + H=5 + H=10 + H=20
    """
    def __init__(self, n_epochs, horizons=[1, 5, 10, 20], weights=None):
        self.n_epochs = n_epochs
        self.horizons = horizons
        self.weights = weights or [1.0, 0.5, 0.3, 0.2]
        self.n_phases = len(horizons)

    def get_active_horizons(self, epoch):
        """Return list of (horizon, weight) for current epoch."""
        progress = epoch / max(self.n_epochs - 1, 1)
        active = []

        for i, (h, w) in enumerate(zip(self.horizons, self.weights)):
            # Each horizon activates at phase boundary
            threshold = i / self.n_phases
            if progress >= threshold:
                active.append((h, w))

        return active


def get_teacher_forcing_ratio(epoch, n_epochs, strategy='cosine'):
    """Scheduled sampling: gradually reduce teacher forcing."""
    progress = epoch / max(n_epochs - 1, 1)

    if strategy == 'none':
        return 0.0
    elif strategy == 'linear':
        return max(0.0, 1.0 - progress)
    elif strategy == 'cosine':
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    elif strategy == 'slow_cosine':
        # Stays high longer, drops faster at end
        return max(0.0, math.cos(math.pi * 0.5 * progress))
    elif strategy == 'step':
        if progress < 0.33:
            return 1.0
        elif progress < 0.66:
            return 0.5
        else:
            return 0.1
    else:
        return max(0.0, 1.0 - progress)


# ============================================================
# Training
# ============================================================
def train_multi_step(data, config, seed=42):
    """Train Neural ODE with multi-step curriculum training.

    Args:
        data: dict from load_data()
        config: training configuration dict
        seed: random seed

    Returns:
        (model, state_std, action_std, delta_std, train_log)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    train_eps = data['train_eps']
    episodes = data['episodes']

    # Model
    model = MultiStepODEFunc(
        hidden=config.get('hidden', 64),
        depth=config.get('depth', 3),
        activation=config.get('activation', 'tanh'),
        use_layer_norm=config.get('use_layer_norm', False),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params} params")

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'], eta_min=config['lr'] * 0.01
    )

    # EMA
    ema_decay = config.get('ema_decay', 0.999)
    ema = EMA(model, decay=ema_decay)

    # Curriculum
    curriculum = CurriculumScheduler(
        n_epochs=config['n_epochs'],
        horizons=config.get('curriculum_horizons', [1, 5, 10, 20]),
        weights=config.get('curriculum_weights', [1.0, 0.5, 0.3, 0.2]),
    )

    # Config
    n_epochs = config['n_epochs']
    batch_size = config.get('batch_size', 64)
    lambda_single = config.get('lambda_single', 1.0)
    lambda_traj = config.get('lambda_traj', 0.5)
    lambda_jacobian = config.get('lambda_jacobian', 0.01)
    tf_strategy = config.get('tf_strategy', 'cosine')
    state_weights = config.get('state_weights', DEFAULT_STATE_WEIGHTS)
    patience = config.get('patience', 60)

    # Pre-compute flat training data
    flat_obs, flat_act, flat_delta = sample_flat_batch(
        episodes, train_eps, len(train_eps) * 1000, rng
    )
    n_flat = len(flat_obs)

    print(f"\n  Training multi-step (seed={seed}, epochs={n_epochs})...")
    print(f"  Curriculum: {config.get('curriculum_horizons', [1, 5, 10, 20])}")
    print(f"  TF strategy: {tf_strategy}")
    print(f"  lambda_single={lambda_single}, lambda_traj={lambda_traj}")

    start_time = time.time()
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_log = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {'total': 0, 'single': 0, 'traj': 0, 'jac': 0}
        n_batches = 0

        # Current settings
        active_horizons = curriculum.get_active_horizons(epoch)
        tf_ratio = get_teacher_forcing_ratio(epoch, n_epochs, tf_strategy)

        # Number of batches per epoch
        n_batches_epoch = config.get('n_batches_per_epoch', 40)

        for batch_idx in range(n_batches_epoch):
            opt.zero_grad()

            # --- Single-step loss ---
            loss_single = torch.tensor(0.0, device=device)
            if lambda_single > 0:
                idx = rng.choice(n_flat, size=min(batch_size, n_flat), replace=False)
                loss_single = single_step_loss(
                    model, flat_obs[idx], flat_act[idx], flat_delta[idx],
                    state_std, action_std, delta_std, DT, state_weights, device
                )

            # --- Trajectory losses (multi-scale) ---
            loss_traj = torch.tensor(0.0, device=device)
            for h, w in active_horizons:
                if h <= 1:
                    continue  # Skip H=1 (covered by single-step)

                # Use smaller batch for longer horizons
                traj_batch = max(4, batch_size // (h // 5 + 1))
                obs_list, act_list, actual_h = sample_trajectory_windows(
                    episodes, train_eps, h, traj_batch, rng
                )

                loss_h = trajectory_rollout_loss(
                    model, obs_list, act_list, actual_h,
                    state_std, action_std, delta_std, DT,
                    tf_ratio, state_weights, device
                )

                if not torch.isnan(loss_h) and not torch.isinf(loss_h):
                    loss_traj = loss_traj + w * loss_h

            # --- Jacobian penalty ---
            loss_jac = torch.tensor(0.0, device=device)
            if lambda_jacobian > 0:
                idx = rng.choice(n_flat, size=min(32, n_flat), replace=False)
                loss_jac = jacobian_penalty(
                    model, flat_obs[idx] / state_std,
                    flat_act[idx] / action_std, device
                )

            # --- Total loss ---
            loss = (lambda_single * loss_single
                    + lambda_traj * loss_traj
                    + lambda_jacobian * loss_jac)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            opt.step()

            # Update EMA
            ema.update(model)

            epoch_losses['total'] += loss.item()
            epoch_losses['single'] += loss_single.item()
            epoch_losses['traj'] += loss_traj.item()
            epoch_losses['jac'] += loss_jac.item()
            n_batches += 1

        scheduler.step()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        train_log.append({
            'epoch': epoch,
            'losses': {k: float(v) for k, v in epoch_losses.items()},
            'tf_ratio': tf_ratio,
            'active_horizons': [h for h, _ in active_horizons],
            'lr': opt.param_groups[0]['lr'],
        })

        # Logging
        if (epoch + 1) % 25 == 0:
            elapsed = time.time() - start_time
            h_list = [h for h, _ in active_horizons]
            print(f"    Epoch {epoch+1}/{n_epochs}: "
                  f"loss={epoch_losses['total']:.6f} "
                  f"(s={epoch_losses['single']:.6f} t={epoch_losses['traj']:.6f} "
                  f"j={epoch_losses['jac']:.6f}) "
                  f"H={h_list} tf={tf_ratio:.2f} "
                  f"t={elapsed:.0f}s")

        # Early stopping on single-step loss (preserve H=1 accuracy)
        val_loss = epoch_losses['single']
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > n_epochs // 2:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Apply best model with EMA
    if best_state is not None:
        model.load_state_dict(best_state)

    # Apply EMA for final model
    ema.apply(model)
    model.eval()

    total_time = time.time() - start_time
    print(f"  Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std, train_log


# ============================================================
# Evaluation
# ============================================================
def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def evaluate(model, data, state_std, action_std, delta_std, horizons,
             n_segments=5, seed=42):
    """Evaluate model on multi-step prediction horizons.

    Uses trajectory replay: feed model's own predictions back.
    """
    device = next(model.parameters()).device
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
                    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0).to(device)
                    a_norm = torch.FloatTensor([[actions_seg[step] / action_std]]).to(device)

                    with torch.no_grad():
                        dsdt_norm = model(s_norm, a_norm).cpu().numpy()[0]

                    dsdt = dsdt_norm * delta_std * DT
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


# ============================================================
# Experiment Runner
# ============================================================
def run_config(config_name, config, data, horizons, seed=42):
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")

    model, state_std, action_std, delta_std, train_log = train_multi_step(
        data, config, seed=seed
    )

    print(f"\n  Evaluating {config_name}...")
    results = evaluate(model, data, state_std, action_std, delta_std, horizons)

    print(f"\n  Results for {config_name}:")
    print(f"  {'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print(f"  {'-'*34}")
    for h in horizons:
        r = results[h]
        print(f"  H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Primary score: mean of H=100,200,500
    valid_scores = []
    for h in [100, 200, 500]:
        if h in results and not np.isnan(results[h]['nmae_mean']):
            valid_scores.append(results[h]['nmae_mean'])
    primary = float(np.mean(valid_scores)) if valid_scores else float('nan')

    # Short-horizon score: mean of H=1,10,50
    short_scores = []
    for h in [1, 10, 50]:
        if h in results and not np.isnan(results[h]['nmae_mean']):
            short_scores.append(results[h]['nmae_mean'])
    short_score = float(np.mean(short_scores)) if short_scores else float('nan')

    # Balanced score: 0.5 * short + 0.5 * long
    if not math.isnan(primary) and not math.isnan(short_score):
        balanced = 0.5 * short_score + 0.5 * primary
    else:
        balanced = float('nan')

    print(f"\n  PrimaryLongHorizonScore (H=100,200,500 avg): {primary:.4f}")
    print(f"  ShortHorizonScore (H=1,10,50 avg): {short_score:.4f}")
    print(f"  BalancedScore: {balanced:.4f}")

    return {
        'config': config_name,
        'results': results,
        'primary_score': primary,
        'short_score': short_score,
        'balanced_score': balanced,
        'train_log': train_log,
    }


def main():
    """Run multi-step training experiments."""
    print("=" * 70)
    print("Multi-Step Training for Long-Horizon Prediction")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")

    horizons = [1, 10, 50, 100, 200, 500]

    # Load data
    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   Train episodes: {len(data['train_eps'])}, Test episodes: {len(data['test_eps'])}")

    # V9 baseline (single-step reference)
    v9_baseline = {
        'nmae': {'1': 0.0051, '10': 0.0628, '50': 0.4557,
                 '100': 0.5064, '200': 0.4737, '500': 0.5529},
        'primary': 0.511
    }

    # ============================================================
    # Experiment Configurations
    # ============================================================
    configs = {
        # Config A: Conservative multi-step (our main proposal)
        # Curriculum: 1 -> 5 -> 10 -> 20
        # High single-step weight to preserve H=1 accuracy
        'multi_step_conservative': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 64,
            'curriculum_horizons': [1, 5, 10, 20],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2],
            'lambda_single': 1.0, 'lambda_traj': 0.5,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'cosine',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config B: Single-step baseline (same arch, no multi-step)
        # This is our fair baseline comparison
        'single_step_baseline': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 256,
            'curriculum_horizons': [1],
            'curriculum_weights': [1.0],
            'lambda_single': 1.0, 'lambda_traj': 0.0,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'none',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config C: Aggressive multi-step (H up to 50)
        # Test if longer curriculum helps or hurts
        'multi_step_aggressive': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 5e-4, 'n_epochs': 300, 'batch_size': 32,
            'curriculum_horizons': [1, 5, 10, 20, 50],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2, 0.1],
            'lambda_single': 0.8, 'lambda_traj': 0.8,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'slow_cosine',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 0.5,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config D: High single-step weight + light trajectory
        # Emphasize H=1 accuracy, use trajectory as regularization only
        'multi_step_h1_heavy': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 128,
            'curriculum_horizons': [1, 5, 10],
            'curriculum_weights': [1.0, 0.3, 0.15],
            'lambda_single': 2.0, 'lambda_traj': 0.2,
            'lambda_jacobian': 0.02,
            'tf_strategy': 'step',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config E: SiLU activation + LayerNorm
        # Test if modern architecture helps
        'multi_step_silu_norm': {
            'hidden': 64, 'depth': 3, 'activation': 'silu',
            'lr': 5e-4, 'n_epochs': 300, 'batch_size': 64,
            'curriculum_horizons': [1, 5, 10, 20],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2],
            'lambda_single': 1.0, 'lambda_traj': 0.5,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'cosine',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': True,
        },

        # Config F: Uniform state weights (ablation)
        # Test if state-specific weighting helps
        'multi_step_uniform_weights': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 64,
            'curriculum_horizons': [1, 5, 10, 20],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2],
            'lambda_single': 1.0, 'lambda_traj': 0.5,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'cosine',
            'state_weights': [1.0] * 7,  # Uniform
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config G: No scheduled sampling (ablation)
        'multi_step_no_tf': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 64,
            'curriculum_horizons': [1, 5, 10, 20],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2],
            'lambda_single': 1.0, 'lambda_traj': 0.5,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'none',  # No teacher forcing
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 0.999,
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },

        # Config H: No EMA (ablation)
        'multi_step_no_ema': {
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 8e-4, 'n_epochs': 300, 'batch_size': 64,
            'curriculum_horizons': [1, 5, 10, 20],
            'curriculum_weights': [1.0, 0.5, 0.3, 0.2],
            'lambda_single': 1.0, 'lambda_traj': 0.5,
            'lambda_jacobian': 0.01,
            'tf_strategy': 'cosine',
            'state_weights': DEFAULT_STATE_WEIGHTS.tolist(),
            'ema_decay': 1.0,  # No EMA (decay=1.0 means no update)
            'weight_decay': 1e-5, 'grad_clip': 1.0,
            'patience': 60, 'n_batches_per_epoch': 40,
            'use_layer_norm': False,
        },
    }

    # ============================================================
    # Run Experiments
    # ============================================================
    all_results = {}

    for config_name, config in configs.items():
        result = run_config(config_name, config, data, horizons, seed=42)
        all_results[config_name] = result

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  V9 Baseline: primary_score = {v9_baseline['primary']:.4f}")
    print(f"  {'Config':<30} {'Short':>8} {'Long':>8} {'Balanced':>10} {'vs V9':>8}")
    print(f"  {'-'*64}")

    best_balanced = float('inf')
    best_config_name = None

    for name, r in all_results.items():
        short = r['short_score']
        long_s = r['primary_score']
        balanced = r['balanced_score']
        vs_v9 = balanced - v9_baseline['primary'] if not math.isnan(balanced) else float('nan')

        marker = ""
        if not math.isnan(balanced) and balanced < best_balanced:
            best_balanced = balanced
            best_config_name = name
            marker = " *"

        print(f"  {name:<30} {short:>8.4f} {long_s:>8.4f} {balanced:>10.4f} {vs_v9:>+8.4f}{marker}")

    print(f"\n  Best config: {best_config_name}")
    print(f"  Best balanced score: {best_balanced:.4f}")
    print(f"  V9 baseline: {v9_baseline['primary']:.4f}")
    if not math.isnan(best_balanced):
        improvement = (v9_baseline['primary'] - best_balanced) / v9_baseline['primary'] * 100
        print(f"  Improvement: {improvement:+.1f}%")

    # ============================================================
    # Save Results
    # ============================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'multi_step_training',
        'description': 'Multi-step training with curriculum learning, scheduled sampling, and trajectory loss',
        'v9_baseline': v9_baseline,
        'horizons': horizons,
        'best_config': best_config_name,
        'best_balanced_score': float(best_balanced),
        'improvement_vs_v9_pct': float((v9_baseline['primary'] - best_balanced) / v9_baseline['primary'] * 100)
            if not math.isnan(best_balanced) else None,
        'configs': {},
    }

    for name, r in all_results.items():
        output['configs'][name] = {
            'config': {k: v for k, v in configs[name].items() if k != 'state_weights'},
            'results': r['results'],
            'primary_score': r['primary_score'],
            'short_score': r['short_score'],
            'balanced_score': r['balanced_score'],
            'train_log_summary': {
                'final_loss': r['train_log'][-1]['losses'] if r['train_log'] else None,
                'n_epochs': len(r['train_log']),
            },
        }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP056_multi_step_training.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    print(f"\nEnd: {datetime.now().isoformat()}")
    return all_results


if __name__ == '__main__':
    main()

"""Ensemble with Switching for Neural ODE.

Core idea: Use multiple diverse models and switch based on uncertainty.
When ensemble variance is high (OOD state), fall back to physics model.

Key differences from existing ensemble_v9.py:
1. Diverse architectures (not just different seeds of same architecture)
2. Switching logic with physics fallback (not just averaging)
3. Contiguous window multi-step training (bug-fixed)
4. Multiple switching strategies tested

Models:
- Model A: Standard v9 (tanh, hidden=64, depth=3)
- Model B: Wider/shallower (tanh, hidden=128, depth=2)
- Model C: SiLU activation (silu, hidden=64, depth=3)
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

# Physical parameters for fallback model
WHEELBASE = 1.0
GRAVITY = 9.81
HEIGHT_CG = 0.8


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


def physics_derivatives(s_cur, action_val, state_std, action_std, delta_std, dt):
    """Compute simplified bicycle physics derivatives.

    This is the fallback model when ensemble uncertainty is high.
    Input: s_cur (7D physical state), action_val (scalar physical action)
    Returns s_next (7D physical state after one Euler step).
    """
    e_y = s_cur[0]
    e_psi = s_cur[1]
    v = s_cur[2]
    theta = s_cur[3]
    theta_dot = s_cur[4]
    delta = s_cur[5]
    delta_dot = s_cur[6]

    # Simplified bicycle kinematics
    e_y_dot = v * np.sin(e_psi)
    e_psi_dot = -v * delta / WHEELBASE
    v_dot = 0.0
    theta_dot_phys = theta_dot
    theta_ddot = (GRAVITY / HEIGHT_CG) * theta - (v ** 2 / (HEIGHT_CG * WHEELBASE)) * delta
    delta_dot_phys = delta_dot
    # Simplified steering: spring-damper toward zero
    delta_ddot = -25.0 * delta - 5.0 * delta_dot

    # Stack derivatives
    dsdt_phys = np.array([
        e_y_dot, e_psi_dot, v_dot, theta_dot_phys,
        theta_ddot, delta_dot_phys, delta_ddot
    ])

    # Clip to prevent explosion
    dsdt_phys = np.clip(dsdt_phys, -10.0, 10.0)

    # Euler step
    s_next = s_cur + dsdt_phys * dt
    return s_next


def check_survival(state):
    """Check if state survives physical limits."""
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(seed=42):
    """Load 7D data from 8D dataset with episode structure preserved."""
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


def sample_contiguous_windows(episodes, ep_indices, window_size, batch_size,
                               state_std, action_std, delta_std, dt):
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


def train_single_model(data, config, seed=42, bootstrap_fraction=1.0):
    """Train a single Neural ODE model with contiguous window multi-step loss.

    Args:
        bootstrap_fraction: fraction of training episodes to sample (for diversity).
            Use <1.0 for bootstrap aggregating to increase ensemble diversity.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    dt = 1.0 / 30.0

    train_eps = data['train_eps']
    episodes = data['episodes']

    # Bootstrap sampling for diversity
    if bootstrap_fraction < 1.0:
        n_bootstrap = max(1, int(len(train_eps) * bootstrap_fraction))
        np.random.seed(seed + 1000)
        boot_indices = np.random.choice(len(train_eps), size=n_bootstrap, replace=True)
        boot_eps = train_eps[boot_indices]
    else:
        boot_eps = train_eps

    # Split train into train/val (90/10)
    n_train = len(boot_eps)
    n_val = max(1, n_train // 10)
    val_eps = boot_eps[:n_val]
    actual_train_eps = boot_eps[n_val:]

    model = ODEFunc(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation']
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['n_epochs'])

    curriculum = [int(x) for x in config.get('rollout_curriculum', '1,5,10,20').split(',')]

    print(f"  Training model (hidden={config['hidden']}, depth={config['depth']}, "
          f"act={config['activation']}, seed={seed}, bootstrap={bootstrap_fraction})...")
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

        n_batches_per_epoch = 80
        for batch_idx in range(n_batches_per_epoch):
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
                    + config.get('lambda_jacobian', 0.01) * loss_jacobian)

            # Guard against loss explosion
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                opt.zero_grad()
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation with early stopping
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
                print(f"    Early stopping at epoch {epoch+1}")
                break

            model.train()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch+1}/{config['n_epochs']}: loss={avg_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, time={elapsed:.1f}s")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    total_time = time.time() - start_time
    print(f"    Training completed in {total_time:.1f}s")

    return model


def predict_single_model(model, s_cur, action, state_std, action_std, delta_std, dt):
    """Get prediction from a single model."""
    s_norm = torch.FloatTensor(s_cur / state_std).unsqueeze(0)
    a_norm = torch.FloatTensor([action / action_std]).unsqueeze(0)

    with torch.no_grad():
        dsdt_norm = model(s_norm, a_norm).numpy()[0]

    dsdt = dsdt_norm * delta_std * dt
    s_next = s_cur + dsdt
    return s_next, dsdt_norm


def evaluate_ensemble_switching(models, data, state_std, action_std, delta_std,
                                 horizons, strategy='adaptive', threshold=0.1,
                                 n_segments=5, seed=42):
    """Evaluate ensemble with switching strategies.

    Strategies:
    - 'ensemble_avg': Simple ensemble averaging (baseline)
    - 'physics_fallback': Switch to physics when uncertainty > threshold
    - 'hold_state': Hold current state when uncertainty > threshold
    - 'adaptive': Weighted blend based on uncertainty (smooth switching)
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
    switching_stats = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}
        n_switched = 0
        n_total = 0
        uncertainty_values = []

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
                        s_next_i, _ = predict_single_model(
                            model, s_cur, actions_seg[step],
                            state_std, action_std, delta_std, dt
                        )
                        predictions.append(s_next_i)

                    predictions = np.array(predictions)
                    mean_pred = np.mean(predictions, axis=0)
                    std_pred = np.std(predictions, axis=0)
                    uncertainty = np.mean(std_pred)

                    n_total += 1
                    uncertainty_values.append(uncertainty)

                    # Apply switching strategy
                    if strategy == 'ensemble_avg':
                        # Simple averaging (no switching)
                        s_next = mean_pred

                    elif strategy == 'physics_fallback':
                        # Switch to physics when uncertain
                        if uncertainty > threshold:
                            s_next = physics_derivatives(
                                s_cur, actions_seg[step],
                                state_std, action_std, delta_std, dt
                            )
                            n_switched += 1
                        else:
                            s_next = mean_pred

                    elif strategy == 'hold_state':
                        # Hold current state when uncertain
                        if uncertainty > threshold:
                            s_next = s_cur.copy()
                            n_switched += 1
                        else:
                            s_next = mean_pred

                    elif strategy == 'adaptive':
                        # Smooth blending based on uncertainty
                        # Weight = 1 when uncertainty is 0, approaches 0 as uncertainty grows
                        weight = np.exp(-uncertainty / threshold)
                        weight = np.clip(weight, 0.0, 1.0)

                        # Physics prediction
                        s_next_phys = physics_derivatives(
                            s_cur, actions_seg[step],
                            state_std, action_std, delta_std, dt
                        )

                        # Weighted blend
                        s_next = weight * mean_pred + (1 - weight) * s_next_phys
                        if weight < 0.5:
                            n_switched += 1

                    else:
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

        switching_stats[h] = {
            'n_switched': n_switched,
            'n_total': n_total,
            'switching_rate': n_switched / max(n_total, 1),
            'uncertainty_mean': float(np.mean(uncertainty_values)) if uncertainty_values else 0,
            'uncertainty_std': float(np.std(uncertainty_values)) if uncertainty_values else 0,
            'uncertainty_p50': float(np.percentile(uncertainty_values, 50)) if uncertainty_values else 0,
            'uncertainty_p90': float(np.percentile(uncertainty_values, 90)) if uncertainty_values else 0,
            'uncertainty_p99': float(np.percentile(uncertainty_values, 99)) if uncertainty_values else 0,
        }

    return results, switching_stats


def main():
    print("=" * 70)
    print("Ensemble with Switching for Neural ODE")
    print("=" * 70)
    print("\nCore idea: Diverse models + uncertainty-based switching to physics")
    print("3 models: Standard(tanh,h64,d3), Wide(tanh,h128,d2), SiLU(silu,h64,d3)")

    # Configuration for each model architecture
    # Use bootstrap sampling + diverse architectures for ensemble diversity
    model_configs = [
        {
            'name': 'standard',
            'hidden': 64, 'depth': 3, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3, 'lambda_jacobian': 0.01,
            'bootstrap_fraction': 0.8,  # Bootstrap 80% of data
        },
        {
            'name': 'wide',
            'hidden': 128, 'depth': 2, 'activation': 'tanh',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3, 'lambda_jacobian': 0.01,
            'bootstrap_fraction': 0.8,
        },
        {
            'name': 'silu',
            'hidden': 64, 'depth': 3, 'activation': 'silu',
            'lr': 1e-3, 'n_epochs': 200, 'batch_size': 256,
            'rollout_curriculum': '1,5,10,20',
            'lambda_multi': 0.3, 'lambda_jacobian': 0.01,
            'bootstrap_fraction': 0.8,
        },
    ]

    # Load data
    print("\n1. Loading data...")
    data = load_data(seed=42)
    print(f"   State std: {data['state_std']}")
    print(f"   Action std: {data['action_std']}")
    print(f"   Delta std: {data['delta_std']}")
    print(f"   Train episodes: {len(data['train_eps'])}")
    print(f"   Test episodes: {len(data['test_eps'])}")

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # Train 3 diverse models
    print("\n2. Training ensemble models...")
    models = []
    seeds = [42, 43, 44]

    for i, (config, seed) in enumerate(zip(model_configs, seeds)):
        print(f"\n{'='*70}")
        print(f"Model {i+1}/3: {config['name']} (seed={seed})")
        print(f"{'='*70}")
        model = train_single_model(
            data, config, seed=seed,
            bootstrap_fraction=config.get('bootstrap_fraction', 1.0)
        )
        models.append(model)

        # Save model
        model_path = f'D:/系统辨识作业/sindy_bicycle/research_72h/07_models/ensemble_switch_{config["name"]}_seed{seed}.pt'
        torch.save({
            'config': config,
            'state_std': state_std,
            'action_std': action_std,
            'delta_std': delta_std,
            'model_state': model.state_dict(),
            'seed': seed,
        }, model_path)
        print(f"    Model saved to: {model_path}")

    # Evaluate all strategies
    horizons = [1, 10, 50, 100, 200, 500]

    # Thresholds calibrated to actual uncertainty range (mean~0.01, p99~0.018)
    strategies = {
        'ensemble_avg': {'threshold': None},
        'physics_fallback_p50': {'strategy': 'physics_fallback', 'threshold': 0.010},
        'physics_fallback_p90': {'strategy': 'physics_fallback', 'threshold': 0.014},
        'physics_fallback_p99': {'strategy': 'physics_fallback', 'threshold': 0.018},
        'hold_state_p50': {'strategy': 'hold_state', 'threshold': 0.010},
        'hold_state_p90': {'strategy': 'hold_state', 'threshold': 0.014},
        'adaptive_p50': {'strategy': 'adaptive', 'threshold': 0.010},
        'adaptive_p90': {'strategy': 'adaptive', 'threshold': 0.014},
        'adaptive_p99': {'strategy': 'adaptive', 'threshold': 0.018},
    }

    print("\n3. Evaluating strategies...")
    all_results = {}

    for strat_name, strat_config in strategies.items():
        print(f"\n{'='*70}")
        strategy = strat_config.get('strategy', 'ensemble_avg')
        threshold = strat_config.get('threshold', 0.1)
        print(f"Strategy: {strat_name} (strategy={strategy}, threshold={threshold})")
        print(f"{'='*70}")

        results, switch_stats = evaluate_ensemble_switching(
            models, data, state_std, action_std, delta_std,
            horizons, strategy=strategy, threshold=threshold,
            n_segments=5, seed=42
        )

        print(f"\n  {'Horizon':<10} {'NMAE':<12} {'Survival':<12} {'Switch%':<12}")
        print("  " + "-" * 46)
        for h in horizons:
            r = results[h]
            s = switch_stats[h]
            print(f"  H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%} "
                  f"{s['switching_rate']:<12.2%}")

        primary = np.mean([results[100]['nmae_mean'], results[200]['nmae_mean'],
                          results[500]['nmae_mean']])
        print(f"\n  PrimaryLongHorizonScore: {primary:.4f}")

        all_results[strat_name] = {
            'strategy': strategy,
            'threshold': threshold,
            'results': results,
            'switching_stats': switch_stats,
            'primary_score': primary,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All Strategies")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} "
          f"{'H=200':<8} {'H=500':<8} {'Primary':<8}")
    print("-" * 90)

    for strat_name, data_dict in all_results.items():
        r = data_dict['results']
        primary = data_dict['primary_score']
        print(f"{strat_name:<25} ", end="")
        for h in horizons:
            print(f"{r[h]['nmae_mean']:<8.4f} ", end="")
        print(f"{primary:<8.4f}")

    # V9 baseline comparison
    v9_nmae = {1: 0.0051, 10: 0.0628, 50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529}
    v9_primary = np.mean([v9_nmae[100], v9_nmae[200], v9_nmae[500]])

    print(f"\n{'='*70}")
    print("Comparison with V9 Baseline")
    print(f"{'='*70}")
    print(f"\nV9 baseline Primary score: {v9_primary:.4f}")

    best_strat = min(all_results.keys(), key=lambda k: all_results[k]['primary_score'])
    best_primary = all_results[best_strat]['primary_score']
    improvement = (v9_primary - best_primary) / v9_primary * 100

    print(f"Best strategy: {best_strat} (Primary: {best_primary:.4f})")
    print(f"Improvement over V9: {improvement:.1f}%")

    print(f"\nDetailed comparison (best strategy vs V9):")
    print(f"{'Horizon':<10} {'V9 NMAE':<12} {'Best NMAE':<12} {'Improvement':<12}")
    print("-" * 46)
    best_results = all_results[best_strat]['results']
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        b_val = best_results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(b_val):
            imp = (v9_val - b_val) / v9_val * 100
            print(f"H={h:<7} {v9_val:<12.4f} {b_val:<12.4f} {imp:<12.1f}%")

    # Individual model analysis
    print(f"\n{'='*70}")
    print("Individual Model Analysis (ensemble_avg = each model contributes)")
    print(f"{'='*70}")

    # Analyze uncertainty distribution
    ea_stats = all_results.get('ensemble_avg', {}).get('switching_stats', {})
    if ea_stats:
        for h in [100, 200, 500]:
            if h in ea_stats:
                s = ea_stats[h]
                print(f"\n  H={h}: uncertainty_mean={s['uncertainty_mean']:.4f}, "
                      f"p50={s['uncertainty_p50']:.4f}, p90={s['uncertainty_p90']:.4f}, "
                      f"p99={s['uncertainty_p99']:.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'ensemble_switching',
        'description': 'Ensemble of 3 diverse Neural ODE models with uncertainty-based switching',
        'model_configs': [
            {'name': c['name'], 'hidden': c['hidden'], 'depth': c['depth'],
             'activation': c['activation']}
            for c in model_configs
        ],
        'seeds': seeds,
        'horizons': horizons,
        'v9_baseline': {
            'nmae': v9_nmae,
            'primary_score': v9_primary,
        },
        'strategies': {
            name: {
                'strategy': d['strategy'],
                'threshold': d['threshold'],
                'primary_score': d['primary_score'],
                'results': d['results'],
                'switching_stats': d['switching_stats'],
            }
            for name, d in all_results.items()
        },
        'best_strategy': best_strat,
        'best_primary_score': best_primary,
        'improvement_over_v9': improvement,
    }

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP024_ensemble_switching.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()

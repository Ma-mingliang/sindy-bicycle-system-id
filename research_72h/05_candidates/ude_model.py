"""Universal Differential Equations (UDE) Model.

Core idea: UDE = Physics equations + NN residual
- Physics provides the known dynamics structure for ALL states
- NN learns only the residual (unmodeled dynamics)

Physics equations used:
  e_y_dot   = v * sin(e_psi)
  e_psi_dot = -v * delta / L
  v_dot     ≈ 0
  theta_dot = theta_dot  (state[4])
  delta_dot = delta_dot  (state[6])

The NN learns corrections for all 7 states, capturing:
  - Nonlinear coupling between states
  - Actuator dynamics (tau → delta_dot, theta_dot)
  - Higher-order effects not in simplified kinematics

Key difference from layered_prediction (EXP043):
  - Layered uses physics for e_y/e_psi, NN for theta/delta (separate)
  - UDE uses physics+NN for ALL states (joint), with NN learning residuals
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
    'delta': np.pi / 2, 'delta_dot': 10.0,
}

WHEELBASE = 1.0
DT = 1.0 / 30.0


# ---------------------------------------------------------------------------
# Physics model: known dynamics for all states
# ---------------------------------------------------------------------------

def physics_deltas(s, dt=DT):
    """Compute physics-based state deltas (in original space).

    Args:
        s: state vector [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
        dt: timestep

    Returns:
        physics_delta: 7D array of predicted state changes
    """
    e_y, e_psi, v, theta, theta_dot, delta, delta_dot = s

    phys = np.zeros(STATE_DIM)
    phys[0] = v * np.sin(e_psi) * dt          # e_y kinematics
    phys[1] = -v * delta / WHEELBASE * dt      # e_psi dynamics
    phys[2] = 0.0                               # v ≈ constant
    phys[3] = theta_dot * dt                    # theta kinematics
    phys[4] = 0.0                               # theta_dot dynamics (unknown)
    phys[5] = delta_dot * dt                    # delta kinematics
    phys[6] = 0.0                               # delta_dot dynamics (unknown)

    return phys


def physics_deltas_batch(s_batch, dt=DT):
    """Batch version of physics_deltas.

    Args:
        s_batch: (N, 7) array of states
    Returns:
        (N, 7) array of physics deltas
    """
    e_y = s_batch[:, 0]
    e_psi = s_batch[:, 1]
    v = s_batch[:, 2]
    theta_dot = s_batch[:, 4]
    delta = s_batch[:, 5]
    delta_dot = s_batch[:, 6]

    phys = np.zeros_like(s_batch)
    phys[:, 0] = v * np.sin(e_psi) * dt
    phys[:, 1] = -v * delta / WHEELBASE * dt
    phys[:, 2] = 0.0
    phys[:, 3] = theta_dot * dt
    phys[:, 4] = 0.0
    phys[:, 5] = delta_dot * dt
    phys[:, 6] = 0.0

    return phys


# ---------------------------------------------------------------------------
# NN residual model
# ---------------------------------------------------------------------------

class UDEModel(nn.Module):
    """UDE = Physics + NN Residual.

    The NN takes normalized (state, action) and predicts a correction
    to the physics-based dynamics in normalized delta space.
    """
    def __init__(self, hidden=128, depth=4, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Initialize final layer with small weights (residual should be small)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s_norm, a_norm):
        """Predict normalized residual correction.

        Args:
            s_norm: normalized state (batch, 7)
            a_norm: normalized action (batch, 1)
        Returns:
            residual_norm: normalized residual (batch, 7)
        """
        return self.net(torch.cat([s_norm, a_norm], dim=-1))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_survival(state):
    for i, name in enumerate(STATE_NAMES_7D):
        if name in PHYSICAL_LIMITS:
            if abs(state[i]) > PHYSICAL_LIMITS[name]:
                return False
    return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))


def load_data(seed=42):
    data = np.load(
        'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        allow_pickle=True,
    )
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

    # Compute physics deltas for training data
    train_phys = physics_deltas_batch(train_obs)
    # Residual = actual - physics
    train_residual = train_deltas - train_phys

    return {
        'episodes': episodes,
        'train_eps': train_eps,
        'test_eps': test_eps,
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
        'train_obs': train_obs,
        'train_action': train_action,
        'train_deltas': train_deltas,
        'train_residual': train_residual,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ude(data, config, seed=42):
    """Train UDE model: physics + NN residual."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    train_obs = data['train_obs']
    train_action = data['train_action']
    train_residual = data['train_residual']

    # Prepare normalized inputs
    train_s_norm = train_obs / state_std
    train_a_norm = train_action.reshape(-1, 1) / action_std
    # Target: residual normalized same as deltas
    train_residual_norm = train_residual / (delta_std * DT)

    X_s = torch.FloatTensor(train_s_norm)
    X_a = torch.FloatTensor(train_a_norm)
    Y = torch.FloatTensor(train_residual_norm)

    ds = torch.utils.data.TensorDataset(X_s, X_a, Y)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=config['batch_size'], shuffle=True,
    )

    model = UDEModel(
        hidden=config['hidden'],
        depth=config['depth'],
        activation=config['activation'],
    )

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config['n_epochs'],
    )

    print(f"Training UDE model (seed={seed})...")
    start_time = time.time()

    model.train()
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        n_batches = 0

        for sb, ab, yb in loader:
            # NN predicts normalized residual
            residual_pred = model(sb, ab)

            # Single-step loss on residual
            loss_single = nn.functional.mse_loss(residual_pred, yb)

            # Multi-step rollout loss (UDE prediction = physics + NN)
            loss_multi = torch.tensor(0.0)
            rollout_steps = 1
            curriculum = config.get('rollout_curriculum', [1, 5, 10, 20])
            for i, threshold in enumerate(curriculum):
                frac = (i + 1) / (len(curriculum) + 1)
                if epoch >= config['n_epochs'] * frac:
                    rollout_steps = threshold

            if rollout_steps > 1 and len(sb) > rollout_steps + 1:
                n_roll = min(len(sb) - rollout_steps, 64)
                s_cur_norm = sb[:n_roll].clone()
                # Convert to original space for physics
                s_cur_orig = s_cur_norm.numpy() * state_std

                for step in range(rollout_steps):
                    a_cur_norm = ab[step:step + n_roll]

                    # Physics prediction (in original space)
                    phys_delta = physics_deltas_batch(s_cur_orig)

                    # NN residual prediction (in normalized space)
                    with torch.no_grad():
                        res_norm = model(s_cur_norm, a_cur_norm).numpy()

                    # Combine: total delta = physics + NN residual * delta_std * dt
                    total_delta = phys_delta + res_norm * (delta_std * DT)
                    s_next_orig = s_cur_orig + total_delta

                    # Update for next step
                    s_cur_orig = s_next_orig
                    s_cur_norm = torch.FloatTensor(s_next_orig / state_std)

                    # Target: next normalized state from data
                    target = sb[step + 1:step + 1 + n_roll]
                    loss_multi = loss_multi + nn.functional.mse_loss(
                        s_cur_norm, target,
                    )

                loss_multi = loss_multi / rollout_steps

            lambda_multi = config.get('lambda_multi', 0.3)
            loss = loss_single + lambda_multi * loss_multi

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
            print(
                f"  Epoch {epoch+1}/{config['n_epochs']}: "
                f"loss={avg_loss:.6f}, rollout={rollout_steps}, "
                f"time={elapsed:.1f}s"
            )

    model.eval()
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return model, state_std, action_std, delta_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ude(model, data, state_std, action_std, delta_std,
                 horizons, n_segments=5, seed=42):
    """Evaluate UDE model with physics + NN residual."""
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
                    # Physics prediction (original space)
                    phys_delta = physics_deltas(s_cur)

                    # NN residual prediction (normalized space)
                    s_norm = torch.FloatTensor(
                        s_cur / state_std,
                    ).unsqueeze(0)
                    a_norm = torch.FloatTensor(
                        [actions_seg[step] / action_std],
                    ).unsqueeze(0)

                    with torch.no_grad():
                        res_norm = model(s_norm, a_norm).numpy()[0]

                    # Combine: total delta = physics + NN residual
                    total_delta = phys_delta + res_norm * delta_std * DT
                    s_next = s_cur + total_delta

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
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name]))
                    if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Universal Differential Equations (UDE) Model")
    print("=" * 60)
    print()
    print("Architecture: Physics equations + NN residual")
    print("  Physics: e_y_dot=v*sin(e_psi), e_psi_dot=-v*delta/L")
    print("  NN: learns residual corrections for all 7 states")
    print()

    config = {
        'hidden': 128,
        'depth': 4,
        'activation': 'tanh',
        'lr': 1e-3,
        'n_epochs': 200,
        'batch_size': 256,
        'rollout_curriculum': [1, 5, 10, 20],
        'lambda_multi': 0.3,
    }

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # 1. Load data
    print("1. Loading data...")
    data = load_data(seed=42)
    print(f"   Train episodes: {len(data['train_eps'])}")
    print(f"   Test episodes: {len(data['test_eps'])}")
    print(f"   State std: {data['state_std']}")
    print(f"   Delta std: {data['delta_std']}")
    print()

    # Show residual statistics
    residual = data['train_residual']
    print("   Residual statistics (actual - physics):")
    for i, name in enumerate(STATE_NAMES_7D):
        print(f"     {name:<12}: mean={np.mean(residual[:, i]):.6f}, "
              f"std={np.std(residual[:, i]):.6f}")
    print()

    # 2. Train
    print("2. Training UDE model...")
    model, state_std, action_std, delta_std = train_ude(data, config, seed=42)
    print()

    # 3. Evaluate
    print("3. Evaluating...")
    horizons = [1, 10, 50, 100, 200, 500, 1000]
    results = evaluate_ude(
        model, data, state_std, action_std, delta_std, horizons,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS: UDE Model")
    print("=" * 60)
    print(f"\n{'Horizon':<10} {'NMAE':<12} {'Survival':<12}")
    print("-" * 34)
    for h in horizons:
        r = results[h]
        print(f"H={h:<7} {r['nmae_mean']:<12.4f} {r['survival_rate']:<12.2%}")

    # Per-state results for key horizons
    for target_h in [1, 100, 500]:
        if target_h in results:
            print(f"\nPer-state NMAE (H={target_h}):")
            print("-" * 40)
            for name in STATE_NAMES_7D:
                mean = results[target_h]['per_state_nmae'][name]['mean']
                print(f"  {name:<12}: {mean:.4f}")

    # 4. Compare with v9 baseline
    print("\n" + "=" * 60)
    print("Comparison with v9 baseline")
    print("=" * 60)
    v9_nmae = {
        1: 0.0051, 10: 0.0628, 50: 0.4557,
        100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }
    print(f"\n{'Horizon':<10} {'v9 NMAE':<12} {'UDE NMAE':<12} {'Delta':<12} {'Status':<10}")
    print("-" * 56)
    for h in horizons:
        v9_val = v9_nmae.get(h, float('nan'))
        ude_val = results[h]['nmae_mean']
        if v9_val > 0 and not np.isnan(ude_val):
            delta = (v9_val - ude_val) / v9_val * 100
            status = "BETTER" if delta > 0 else "WORSE"
            print(f"H={h:<7} {v9_val:<12.4f} {ude_val:<12.4f} {delta:<12.1f}% {status:<10}")

    # 5. Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'run_id': '20260628_ude_model',
        'experiment': 'ude_model',
        'description': 'Universal Differential Equations: physics + NN residual',
        'architecture': {
            'type': 'UDE',
            'physics': [
                'e_y_dot = v * sin(e_psi)',
                'e_psi_dot = -v * delta / L',
                'v_dot = 0',
                'theta_dot = theta_dot (state[4])',
                'delta_dot = delta_dot (state[6])',
            ],
            'nn_residual': {
                'input_dim': STATE_DIM + ACTION_DIM,
                'output_dim': STATE_DIM,
                'hidden': config['hidden'],
                'depth': config['depth'],
                'activation': config['activation'],
            },
            'combination': 'total_delta = physics_delta + nn_residual * delta_std * dt',
        },
        'config': config,
        'results': {},
        'v9_baseline': v9_nmae,
    }

    for h in horizons:
        output['results'][str(h)] = {
            'nmae_mean': results[h]['nmae_mean'],
            'nmae_std': results[h]['nmae_std'],
            'survival_rate': results[h]['survival_rate'],
            'per_state_nmae': results[h]['per_state_nmae'],
        }

    # Compute primary score (mean of H=100,200,500)
    primary_scores = [
        results[h]['nmae_mean']
        for h in [100, 200, 500]
        if not np.isnan(results[h]['nmae_mean'])
    ]
    output['primary_score'] = float(np.mean(primary_scores)) if primary_scores else float('nan')

    output_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP046_ude_model.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Save model
    model_path = 'D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/ude_model.pt'
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
